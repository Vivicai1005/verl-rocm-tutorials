# 深入 verl Fully Async Training：从架构原理到 AMD ROCm 上的 DAPO 训练实践

从 `dapo_7b_math_fsdp2_4_4.sh`去理解verl fully async的工作流
```mermaid
flowchart TD
    A["dapo_7b_math_fsdp2_4_4.sh"]
    --> B["python -m verl.experimental.fully_async_policy.fully_async_main"]

    B --> C["fully_async_main.main()"]

    C --> D["main_ppo.run_ppo(config, FullyAsyncTaskRunner)"]

    D --> E["Ray<br/>FullyAsyncTaskRunner.run()"]

    E --> F["_initialize_components()"]

    F --> F1["FullyAsyncTrainer"]
    F1 --> F11["Training GPUs"]

    F --> F2["FullyAsyncRollouter"]
    F2 --> F21["vLLM Rollout GPUs"]

    F --> F3["MessageQueue"]

    F --> F4["CheckpointEngineManager"]
    F4 --> F41["NCCL weight sync"]

    E --> G["_run_training_loop()"]

    G --> R["FullyAsyncRollouter.fit()<br/><b>PRODUCER</b>"]
    G --> T["FullyAsyncTrainer.fit()<br/><b>CONSUMER</b>"]

    R --> R1["generate sample"]
    R1 --> MQ["MessageQueue"]

    MQ --> T1["get sample from MQ"]

    T --> T1
    T1 --> T2["assemble batch"]
    T2 --> T3["reward / logprob"]
    T3 --> T4["advantage"]
    T4 --> T5["actor update"]

    T5 -->|"every 4 local steps"| T6["NCCL parameter sync"]
    T6 --> T7["Rollouter new version"]

    T7 -.-> R
```

## FullyAsyncTaskRunner
```
dapo_7b_math_fsdp2_4_4.sh
        │
        ▼
python -m verl.experimental.fully_async_policy.fully_async_main
        │
        ▼
main(config)
        │
        ▼
run_ppo(config, FullyAsyncTaskRunner)
        │
        ▼
FullyAsyncTaskRunner.run(config)
```
<details>
<summary>fully_async_main.py中main()代码</summary>
    
```python 
@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")
```
</details>

```mermaid
flowchart TB

    Runner["FullyAsyncTaskRunner<br/>Orchestrator"]

    subgraph AsyncSystem["Fully Async Training System"]
        direction TB

        subgraph SampleFlow["Rollout Sample Path"]
            direction LR

            R["FullyAsyncRollouter<br/>Producer"]
            MQ["MessageQueue<br/>Buffer"]
            T["FullyAsyncTrainer<br/>Consumer"]

            R -->|"Rollout Samples"| MQ
            MQ -->|"Training Samples"| T
        end

        subgraph WeightFlow["Policy Weight Sync Path"]
            direction RL

            V["vLLM Rollout Replicas"]
            CE["CheckpointEngineManager"]

            CE -->|"Sync Weights"| V
        end

        SampleFlow -->|"Updated Policy"| WeightFlow
    end

    Runner -. "initialize & orchestrate" .-> AsyncSystem
```

Ray 是一个统计计算框架，旨在实现简单地从单机到大型分布式集群的扩展，提供构建和运行分布式应用的底层基础设置和一组核心原语。
Ray Task 是 Ray 中最基本的计算单元，代表一个无状态的远程函数。Ray Task的每次执行都是独立的，不保留之前的任何信息。就像调用一个普通函数，执行完就清除内部状态。我们调用一个 Ray Task 后，会立即返回得到一个Ray ObjectRef，而不是实际的结果。主程序可以继续执行其他操作，而 Ray Task 则在后台并行运行。我们需要使用ray.get() 来获取Task的实际结果。Ray Task非常适合并行执行大量独立、一次性的计算任务，譬如数据批处理、独立的模型推理等场景。

Fully Async的初始化主要由 `FullyAsyncTaskRunner._initialize_components()` 完成。 `main()`首先通过Hydra加载并合并训练配置，设置运行设备，然后调用通用的`run_ppo()` 初始化Ray,并启动一个 `FullyAsyncTaskRunner` Ray Actor,; TaskRunner 随后负责真正构建 Fully Async 训练系统。 

FullyAsyncRunner 初始化 `FullyAsyncTrainer` 和 `FullyAsyncRollouter`: Trainer 管理 Actor, Reference 等训练侧 Worker, 并负责消费rollout sample, 计算 advantage 和更新 policy; Rollouter 管理独立的 rollout GPU 和异步推理引擎，负责持续生成 trajectory。初始化 Trainer 前，会先通过 `create_role_worker_mapping()` 确定不同 Role 对应的 Worker 类型和 WorkerGroup 实现，并通过 ResourcePool 将 Trainer 的 Worker 分配到对应的 GPU 资源上。

Trainer 和 Rollouter 创建完成之后，`fully_async_main.py` 会进一步建立两条关键通信路径。第一条是 sample data path: 创建一个共享的 `MessageQueue`，并把同一个 `MessageQueueClient` 注入 Trainer 和 Rollouter。


## FullyAsyncRollouter

FullyAsyncRollouter的数据流如下：
```mermaid
sequenceDiagram
    autonumber

    participant R as FullyAsyncRollouter
    participant PQ as pending_queue
    participant ALM as FullyAsyncAgentLoopManager
    participant V as vLLM Replica
    participant MQ as MessageQueue

    loop Feed new rollout requests
        R->>R: _feed_samples()
        R->>PQ: put(RolloutSample)
    end

    loop Process rollout concurrently
        R->>PQ: get()
        PQ-->>R: RolloutSample

        R->>R: _processor_worker()
        R->>ALM: generate_sequences_single()

        ALM->>V: generate(prompt)
        V-->>ALM: generated tokens + log_probs

        ALM-->>R: DataProto
        R->>MQ: put_sample(RolloutSample)
    end
```
> [!TIP]
> **Producer-Consumer**
>
> 生产者-消费者模式 （Producer-Consumer Pattern） 是一种经典的并发设计模式，用于解决两个处理速率不一致的组件之间的数据传输问题。
>
> - **核心思想**：生产者 （Producer） 和消费者 （Consumer）并不直接通信，而是通过一个 缓冲区 （Buffer/Queue）进行解耦。
> - **Producer**：负责生成数据，并将其放入缓冲区。如果缓冲区满了，生产者必须等待或丢弃数据。
> - **Consumer**：负责从缓冲区取出数据进行处理。如果缓冲区空了，消费者必须等待。
> - **Buffer**：平滑了生产和消费的速率波动，允许两者并行工作，互不阻塞。

`FullyAsyncRollouter` 是 Fully Async Training 中的 rollout producer。它持续从训练数据集中读取prompt，将rollout request放入内部 `pending queue`



## FullyAsyncTrainer

## MessageQueue



```mermaid
sequenceDiagram
    autonumber

    participant D as Dataset
    participant R as Rollouter<br/>4 GPUs / vLLM
    participant MQ as MessageQueue
    participant T as Trainer<br/>4 GPUs / FSDP2
    participant CE as NCCL CheckpointEngine

    loop Fully Async Execution

        par Rollout continuously produces samples
            D->>R: 1 prompt
            Note over R: repeat n = 16
            R->>R: Generate 16 responses
            R->>MQ: Put 1 RolloutSample
        and Trainer continuously consumes samples
            loop Collect 128 RolloutSamples
                T->>MQ: get_sample()
                MQ-->>T: RolloutSample
            end

            Note over T: 128 prompt groups × 16 responses<br/>= 2048 trajectories

            T->>T: Assemble training batch
            T->>T: Reward
            T->>T: Log Prob
            T->>T: Advantage
            T->>T: Actor Update
        end

        Note over T: local_trigger_step += 1

        alt local_trigger_step == 4
            T->>T: current_param_version += 1

            T->>CE: Synchronize parameters
            CE->>R: Abort active generation
            CE->>R: NCCL broadcast weights
            CE->>R: Resume generation

            Note over R: partial_rollout=True<br/>unfinished responses continue

            T->>R: reset_staleness()
        end
    end
```
