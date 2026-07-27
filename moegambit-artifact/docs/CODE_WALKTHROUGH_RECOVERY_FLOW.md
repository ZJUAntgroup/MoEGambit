# MoEGambit 代码级流程讲解

本文假设读者已经知道 MoEGambit 的整体目标：MoE 训练中某个 rank fail-stop 后，尽量通过热替换和混合状态恢复避免整作业重启。本文进一步回答一个更具体的问题：这些流程在代码里到底是怎么串起来的？

阅读方式建议：

1. 先看每节的“调用链”。
2. 再打开对应文件，搜索文中的函数名。
3. 最后回到训练日志，看日志事件是否能对应到这些函数。

## 1. 总调用链

一次正常启动到故障恢复，主线可以压缩成下面这条链：

```text
moegambit-launch
  -> RuntimeConfig / FeatureSwitches
  -> MegatronAdapter.prepare_launch
  -> Megatron training.py
      -> setup_model_and_optimizer
      -> elastic_zero2_initialize
      -> maybe_initialize_moegambit_moe
      -> training loop
          -> train_step
              -> forward/backward
              -> moegambit_should_commit_optimizer
              -> elastic_zero2_wait_before_optimizer_step
              -> optimizer.step
              -> elastic_zero2_schedule_after_optimizer_step
          -> communication failure / injected fault
              -> RecoveryController.on_hard_rank_failure
              -> HotSparePool.allocate_spare
              -> ReplacementRegistry.announce_replacement
              -> ReplacementRegistry.announce_replacement_ready
              -> RecoveryController.at_safe_point
                  -> group repair / topology refresh
                  -> dense_sync_fn
                  -> expert_restore_fn
                  -> optional ZeRO-2 memory replica restore
                  -> reintegration barrier
              -> RecoveryController._finalize_reintegration
```

这条链里有三类代码：

| 类型 | 位置 | 作用 |
| --- | --- | --- |
| engine-neutral runtime | [`src/moegambit`](../src/moegambit) | 解析 feature、选择 adapter、准备环境变量，不依赖 Torch/Megatron。 |
| Megatron 训练 hook | [`training.py`](../src/Megatron-LM/megatron/training/training.py) | 在模型构建、训练循环、optimizer step 前后插入恢复逻辑。 |
| MoE recovery backend | [`megatron/core/transformer/moe`](../src/Megatron-LM/megatron/core/transformer/moe) | 管理故障状态机、热备、专家恢复、两阶段恢复和 reintegration。 |

## 2. 启动入口：runtime 先决定打开哪些能力

入口 CLI 是 [`src/moegambit/cli/launch.py`](../src/moegambit/cli/launch.py)。读者可以搜索 `def main`。

核心逻辑很短：

```python
config = runtime_config(args)
prepared = prepare_launch(command, config)
return run(command, config)
```

`runtime_config(args)` 来自 [`src/moegambit/cli/common.py`](../src/moegambit/cli/common.py)，它把 `--hot-swap` 和 `--zero2` 转成 `FeatureSwitches`：

```python
FeatureSwitches(
    hot_swap=args.hot_swap or inherited.features.hot_swap,
    zero2=args.zero2 or inherited.features.zero2,
)
```

这一步只决定能力开关，不做任何训练引擎相关操作。

## 3. FeatureSwitches 如何投影成环境变量

[`src/moegambit/runtime/config.py`](../src/moegambit/runtime/config.py) 的 `RuntimeConfig.project_environment(...)` 会把 canonical 环境变量和旧 elastic 兼容变量同时写进去：

```python
env.update({
    "MOEGAMBIT_HOT_SWAP": enabled(self.features.hot_swap),
    "MOEGAMBIT_ZERO2": enabled(self.features.zero2),
    "ELASTIC_HOT_SWAP_ENABLED": enabled(self.features.hot_swap),
    "ELASTIC_ZERO2_MEMORY_REPLICATION": enabled(self.features.zero2),
})
```

如果 `hot_swap` 没开，它还会清理一次性恢复变量，例如：

```text
ELASTIC_REBUILD_MODE
ELASTIC_REPLACEMENT_RANK
ELASTIC_RESUME_ITERATION
ELASTIC_RECOVERY_DESCRIPTOR
ELASTIC_PG_GENERATION
```

这个设计避免普通启动误继承上一次 replacement/rebuild 的环境。

## 4. MegatronAdapter 如何改写命令

runtime 不直接知道 Megatron 的参数。Megatron 细节在 [`src/moegambit_megatron/__init__.py`](../src/moegambit_megatron/__init__.py) 的 `MegatronAdapter.prepare_launch(...)`。

关键逻辑是：

```python
if not request.features.hot_swap:
    command = _remove_flags(command, hot_swap_flags, valued_hot_swap_flags)

if request.features.zero2 and "--use-distributed-optimizer" not in command:
    command += ("--use-distributed-optimizer",)

environment.update({
    "MOEGAMBIT_MEGATRON_ADAPTER": "1",
    "ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER": "1" if zero2 else "0",
})
```

所以：

1. `--no-hot-swap` 会剥掉一部分热替换相关 Megatron flags。
2. `--zero2` 会自动确保 Megatron 启用 distributed optimizer。
3. runtime 层只传 feature，Megatron adapter 才知道具体要加哪个 Megatron flag。

## 5. Megatron 训练初始化：三个关键 hook

训练主文件是 [`training.py`](../src/Megatron-LM/megatron/training/training.py)。在模型和优化器创建后，有三个关键 hook：

```python
model, optimizer, opt_param_scheduler = setup_model_and_optimizer(...)

elastic_zero2_initialize(
    model,
    optimizer,
    initial_step=int(args.iteration),
    start_transport=not _elastic_rebuild,
)

if _elastic_rebuild:
    elastic_replacement_sync_params(model, optimizer, opt_param_scheduler)

maybe_initialize_moegambit_moe(
    model, args,
    optimizer=optimizer,
    opt_param_scheduler=opt_param_scheduler,
)
```

这三步分别做：

| 函数 | 做什么 |
| --- | --- |
| `elastic_zero2_initialize(...)` | 如果启用 ZeRO-2 memory replication，建立 optimizer shard host replica manager。 |
| `elastic_replacement_sync_params(...)` | replacement/rebuild 模式下，从健康 peer 或 ZeRO-2 holder 恢复状态。 |
| `maybe_initialize_moegambit_moe(...)` | 初始化 MoEGambit 的 controller、directory、registry、barrier、callbacks。 |

注意 `_elastic_rebuild`：普通 rank 会直接训练；replacement rank 会先同步状态，再进入训练循环。

## 6. maybe_initialize_moegambit_moe 做了什么

函数位置：[`moegambit_integration.py`](../src/Megatron-LM/megatron/core/transformer/moe/moegambit_integration.py)，搜索 `def maybe_initialize_moegambit_moe`。

它是 Megatron 后端的总装配函数，按顺序做这些事：

1. 检查 `args.moe_moegambit_enable`，没开就返回。
2. 读取当前并行拓扑：EP group、DP group、world size、num experts、num layers。
3. 初始化 expert directory、replacement registry、group rebuild、topology manager、reintegration barrier。
4. 初始化 deferred optimizer loader 和 two-phase recovery coordinator。
5. 创建并配置 `RecoveryController`。
6. 调 `_wire_recovery_callbacks(...)`，把实际恢复动作注册进 controller。
7. 初始化 hard failure detector、iteration invalidator、optimizer commit guard。
8. 初始化 async recovery worker 和 fault injection。
9. 初始化 hot spare pool。

最重要的是第 6 步：`RecoveryController` 本身只负责状态机，不直接知道怎么复制参数、怎么读 checkpoint。实际动作通过 callback 注入：

```python
ctrl.register_callbacks(
    dense_sync_fn=dense_sync_fn,
    expert_restore_fn=expert_restore_fn,
    checkpoint_restart_fn=checkpoint_restart_fn,
    expert_peer_sync_fn=expert_peer_sync_fn,
    force_checkpoint_restart_fn=force_checkpoint_restart_fn,
)
```

这种结构让控制面和状态恢复实现解耦：controller 决定“什么时候做”，callback 决定“怎么做”。

## 7. 正常训练时 optimizer step 如何被保护

`train_step(...)` 里 optimizer 更新前有两道保护。

第一道是 MoEGambit commit guard：

```python
if not moegambit_should_commit_optimizer():
    moegambit_mark_optimizer_skipped(reason="iteration_invalidated")
    return ...
```

如果故障发生在 forward/backward/optimizer 中间，本轮 iteration 会被标记 invalidated。此时不能执行 `optimizer.step()`，否则某些 rank 更新了参数、某些 rank 没更新，训练状态就不一致。

第二道是 ZeRO-2 memory replication 的 safe-point snapshot：

```python
elastic_zero2_wait_before_optimizer_step(args.curr_iteration)
update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
...
elastic_zero2_schedule_after_optimizer_step(args.curr_iteration + 1)
```

含义是：

1. `optimizer.step()` 修改本地 optimizer shard 之前，先确保当前 step 的旧 shard 已经复制到 ring holder。
2. `optimizer.step()` 成功后，调度下一版本 snapshot。
3. 即使 loss-scale skip，没有真正更新参数，也会调度 snapshot，因为外层 iteration 已经前进，恢复版本号也要前进。

## 8. 故障如何进入 RecoveryController

故障有两类入口：

1. 训练循环捕获 NCCL/通信异常。
2. fault injection 或 hard failure detector 主动触发。

训练循环中可以搜索 `caught communication error in train_step`。大致逻辑是：

```python
try:
    train_step(...)
except RuntimeError as exc:
    if is_comm_error and args.moe_moegambit_enable:
        elastic_on_nccl_error(exc)
        if elastic_check_pause():
            ...
```

MoEGambit 模块内部，hard failure detector 会注册：

```python
detector.register_callbacks(
    on_hard_failure_fn=_RECOVERY_CONTROLLER.on_hard_rank_failure,
    on_iteration_invalid_fn=invalidator.invalidate,
)
```

真正进入状态机的函数是 [`RecoveryController.on_hard_rank_failure(...)`](../src/Megatron-LM/megatron/core/transformer/moe/recovery_controller.py)。

它会做这些状态修改：

1. 创建或更新 `FaultRecord`。
2. 把受影响 experts 标成 recovering。
3. 如果是 mid-iteration，把当前 iteration 标记 invalidated。
4. 调 optimizer commit block callback。
5. 调 `enter_waiting_fn` 通知外部系统准备 replacement。
6. 把 phase 从 `HEALTHY_TRAINING` 切到 `PENDING_GROUP_REPAIR`。

简化后的代码形态是：

```python
record = FaultRecord(
    failed_rank=failed_rank,
    fault_type="hard",
    expert_ids=expert_ids,
    mid_iteration=mid_iteration,
)
self._active_faults[failed_rank] = record

if expert_ids:
    self._expert_tracker.mark_recovering(expert_ids, step=step)

if mid_iteration:
    self._iteration_invalidated = True

self._optimizer_commit_block_fn(step=step, failed_rank=failed_rank)
self._enter_waiting_fn(failed_rank=failed_rank, step=step)
self._transition_to(RecoveryPhase.PENDING_GROUP_REPAIR, ...)
```

## 9. 热备 rank 如何被分配

热备池在 [`hot_spare_pool.py`](../src/Megatron-LM/megatron/core/transformer/moe/hot_spare_pool.py)，搜索 `allocate_spare`。

关键点：`allocate_spare(...)` 只做控制面分配，不把 spare 加进任何 NCCL group。

```python
available = self.available_spare_ranks
spare_rank = available[0]
slot.state = SpareState.ALLOCATED
slot.assigned_to_failed_rank = failed_rank
self._notify_spare_fn(spare_rank=spare_rank, failed_rank=failed_rank, step=step)
```

这段代码有三个重要含义：

1. spare 初始状态是 `STANDBY`，分配后变成 `ALLOCATED`。
2. 通知通过 control channel，不通过训练 NCCL collective。
3. 真正的 group rebuild 必须等 safe point。

## 10. ReplacementRegistry 如何约束生命周期

替换 rank 生命周期在 [`replacement_registry.py`](../src/Megatron-LM/megatron/core/transformer/moe/replacement_registry.py)。

三个最重要的 API：

```python
announce_replacement(...)
announce_replacement_ready(...)
mark_integrated(...)
```

状态流转是：

```text
NOT_PRESENT
  -> BOOTSTRAPPING
  -> READY_FOR_REPAIR
  -> INTEGRATED
```

含义分别是：

| 状态 | 可以做什么 | 不可以做什么 |
| --- | --- | --- |
| `BOOTSTRAPPING` | replacement 进程启动、读环境、准备模型。 | 不能参加训练 collectives。 |
| `READY_FOR_REPAIR` | 等 controller 在 safe point 做 group repair。 | 仍不能跑下一轮 forward。 |
| `INTEGRATED` | 成为完整训练 rank。 | 不再是旁路 replacement。 |

`mark_integrated(...)` 的注释明确要求：只能在 safe point 内调用，在下一次 forward pass 前完成。

## 11. safe point 如何触发真正恢复

`RecoveryController.at_safe_point(step)` 是恢复动作的入口。它会：

1. 清理上一轮 invalidation。
2. 如果 phase 是 `REINTEGRATED`，先尝试 `_finalize_reintegration(...)`。
3. 如果 phase 是 `SAFE_POINT_REPAIR`，调用 `_execute_safe_point_repair(...)`。

简化结构：

```python
if self._phase == RecoveryPhase.REINTEGRATED:
    self._finalize_reintegration(step=step)

if self._phase == RecoveryPhase.SAFE_POINT_REPAIR:
    repaired = self._execute_safe_point_repair(step=step)
```

safe point repair 内部可以理解为三段：

| 阶段 | 内容 |
| --- | --- |
| Phase A | process group repair、expert directory refresh、dispatch topology refresh。 |
| Phase B | 选择 checkpoint restart 或 hybrid recovery，并执行状态恢复。 |
| Phase C | 设置 reintegration barrier，转入 `REINTEGRATED`。 |

Phase B 的选择由 gap-aware policy 和运行时条件决定。结果通常是两条路径之一：

```text
CHECKPOINT_RESTART
HYBRID_RECOVERY
```

## 12. checkpoint restart 路径

函数：`RecoveryController._execute_checkpoint_restart_path(...)`。

这条路径调用注册的 `checkpoint_restart_fn(...)`：

```python
self._checkpoint_restart_fn(
    failed_rank=failed_rank,
    replacement_rank=replacement_rank,
    step=step,
    decision=decision,
)
```

语义是：dense、expert、optimizer、scheduler 都从 checkpoint 恢复。它不会使用 dense-from-peer / expert-from-checkpoint 的混合逻辑。

如果 checkpoint restart callback 失败，controller 会 fallback 到 hybrid recovery：

```python
except Exception:
    self._execute_hybrid_recovery_path(...)
```

这就是为什么日志里可能看到先选择 checkpoint restart，失败后又进入 hybrid。

## 13. hybrid recovery 路径

函数：`RecoveryController._execute_hybrid_recovery_path(...)`。

这条路径最核心的两步是：

```python
self._dense_sync_fn(
    failed_rank=failed_rank,
    replacement_rank=replacement_rank,
    step=step,
)

self._expert_restore_fn(
    failed_rank=failed_rank,
    replacement_rank=replacement_rank,
    step=step,
    expert_ids=ready_record.expert_ids,
)
```

所以读 hybrid recovery 不要只读 controller。真正的数据搬运在 `moegambit_integration.py` 里注册的两个 callback。

## 14. dense_sync_fn：为什么 dense 可以从 DP peer 拉

函数位置：[`moegambit_integration.py`](../src/Megatron-LM/megatron/core/transformer/moe/moegambit_integration.py)，搜索 `def dense_sync_fn`。

它的职责是恢复 dense/shared/router 这类 DP replicated state。注释里写得很清楚：

```text
Expert parameters are intentionally skipped.
They will be recovered from checkpoint in expert_restore_fn.
```

代码里的核心判断来自 [`dense_param_sync.py`](../src/Megatron-LM/megatron/core/transformer/moe/dense_param_sync.py)：通过 Megatron 参数的 `param.allreduce` 识别 dense-like 参数。直观规则是：

```text
param.allreduce == True  -> dense/shared/router, DP peers 应一致
param.allreduce == False -> local expert, 不能随便从 DP peer 复制
```

所以 dense path 的数据来源是“当前健康 DP peer”，它恢复的是更接近当前 iteration 的状态，而不是旧 checkpoint。

非 ZeRO-2 optimizer 下，dense optimizer state 也可以从健康 DP peer 同步。ZeRO-2 下 optimizer state 已经分片，不能用这个 callback 直接复制完整 optimizer state。

## 15. expert_restore_fn：为什么 expert 从 checkpoint 或 EDP peer 恢复

函数位置：[`moegambit_integration.py`](../src/Megatron-LM/megatron/core/transformer/moe/moegambit_integration.py)，搜索 `def expert_restore_fn`。

它大致做这些事：

1. 根据 `failed_rank` 和 `expert_ids` 构造 restore plan。
2. 读取 checkpoint manifest。
3. 构造 `load_fn`。
4. 调 `StaleExpertRestoreCoordinator.execute_restore(...)` 写回 expert 权重。
5. 通知 two-phase coordinator：weights restored。
6. 如果需要恢复 expert optimizer state，提交 deferred optimizer load。
7. 运行 unified post-recovery convergence。

简化结构：

```python
plan = coordinator.build_restore_plan(...)
load_fn = _build_expert_load_fn(checkpoint_dir=checkpoint_dir, model=model)

result = coordinator.execute_restore(
    model=target_model,
    plan=plan,
    load_fn=load_fn,
    directory=directory,
    barrier=barrier,
    step=step,
)

if result.success:
    _two_phase_coord.on_weights_restored(...)
    opt_loader.submit_from_restore_plan(plan, step=step)
    _two_phase_coord.on_optimizer_submitted(...)
```

这就是“weights first, optimizer later”的具体实现。

## 16. two-phase recovery 和 optimizer barrier

两阶段状态机在 [`two_phase_recovery.py`](../src/Megatron-LM/megatron/core/transformer/moe/two_phase_recovery.py)：

```text
NOT_STARTED
  -> WEIGHTS_LOADING
  -> WEIGHTS_READY
  -> OPTIMIZER_PENDING
  -> FULLY_RECOVERED
  -> COMPLETED
```

专家权重恢复后，expert 可以进入 `WEIGHTS_READY`，但 optimizer state 没恢复前不能被更新。这个限制通过 `OptimizerUpdateBarrier` 和 [`deferred_optimizer_load.py`](../src/Megatron-LM/megatron/core/transformer/moe/deferred_optimizer_load.py) 实现。

`deferred_optimizer_load.py` 的核心流程是：

```text
submit_from_restore_plan
  -> 生成 optimizer state load request
poll_and_finalize
  -> 执行 load_fn
  -> finalize_loaded
  -> unblock barrier
  -> on_optimizer_loaded
```

这让 safe point 里的关键路径更短：先把 expert weights 补上，optimizer state 可以随后完成，但更新被 barrier 保护。

## 17. ZeRO-2 memory replication：optimizer shard 如何被备份

ZeRO-2 相关恢复主要在两个文件：

| 文件 | 作用 |
| --- | --- |
| [`zero2_memory_checkpoint.py`](../src/Megatron-LM/megatron/training/zero2_memory_checkpoint.py) | 管理 optimizer shard host-memory snapshot 和 TCP ring 复制。 |
| [`elastic_client.py`](../src/Megatron-LM/megatron/training/elastic_client.py) | 把 snapshot 初始化、调度、quiesce、恢复接进训练流程。 |

`elastic_zero2_initialize(...)` 会检查是否启用：

```python
if not _zero2_memory_replication_enabled():
    return None

if not args.use_distributed_optimizer:
    raise RuntimeError("PHOENIX replication requires --use-distributed-optimizer")
```

然后创建 `Zero2MemoryReplicaManager`。这个 manager 每个 rank 有：

```text
两个 local staging buffers
两个 peer receive buffers
一个 outgoing TCP connection
一个 incoming TCP connection
```

ring 关系由 `ring_neighbors(...)` 决定：

```text
rank i 的 optimizer shard
  -> 复制到 DP ring 里的 next(i)
```

`schedule_snapshot(step)` 做的是：

1. 收集 optimizer tensor refs 和 scalar refs。
2. 生成 manifest 和 manifest hash。
3. 把 GPU optimizer shard 拷到 pinned CPU buffer。
4. 后台 sender thread 把 snapshot 发给 holder。
5. holder 收完后返回 ack。

这解释了为什么 `train_step(...)` 会在 optimizer step 前调用：

```python
elastic_zero2_wait_before_optimizer_step(args.curr_iteration)
```

它要保证旧版本 optimizer shard 已经安全复制到 holder，然后本 rank 才能修改本地 shard。

## 18. ZeRO-2 restore：replacement 如何拿回 shard

replacement 同步参数时会进入 [`elastic_client.py`](../src/Megatron-LM/megatron/training/elastic_client.py) 的 `_sync_params_to_new_rank(...)`。

这个函数先恢复 dense model params：

```python
_sync_tensor_peer_chunked(param.data, sync_src_rank, replacement_rank, ...)
```

然后根据是否启用 ZeRO-2 memory replication 选择 optimizer state 来源：

```python
if not zero2_memory_enabled:
    optimizer_summary = _sync_non_expert_optimizer_state_peer(...)
else:
    optimizer_summary = _restore_zero2_optimizer_from_memory_peer(...)
```

`_restore_zero2_optimizer_from_memory_peer(...)` 的参与方只有两个：

```text
holder_rank      -> 拥有 failed logical rank 的 host-memory replica
replacement_rank -> 需要恢复 optimizer shard
```

holder 发送 snapshot header 和 raw buffers；replacement 接收后调用：

```python
apply_optimizer_snapshot(
    snapshot,
    tensor_refs,
    scalar_refs,
    restore_expert=restore_expert,
)
```

`apply_optimizer_snapshot(...)` 会重新计算本地 manifest，并要求和 snapshot 完全一致：

```python
if manifest_hash != snapshot.manifest_hash or manifest != snapshot.manifest:
    raise RuntimeError("optimizer memory snapshot manifest mismatch")
```

这就是 fail closed：只要 replacement 的 optimizer layout 和 holder 保存的 layout 对不上，就拒绝恢复，避免把 optimizer shard 写错位置。

## 19. reintegration：什么时候算恢复完成

safe point repair 成功后，controller 会设置 reintegration barrier，然后转到 `REINTEGRATED`：

```python
self._reintegration_barrier.begin_reintegration(...)
self._reintegration_barrier.mark_precondition(..., "replacement_ready")
self._reintegration_barrier.mark_precondition(..., "groups_repaired")
self._reintegration_barrier.mark_precondition(..., "directory_refreshed")
self._reintegration_barrier.mark_precondition(..., "topology_refreshed")
self._transition_to(RecoveryPhase.REINTEGRATED, ...)
```

下一次 safe point 或 post-step hook 会进入 `_finalize_reintegration(...)`。

这个函数做五件事：

1. 检查 barrier 是否允许 reintegrate。
2. 执行 barrier reintegration。
3. 把 recovered experts 标为 `HEALTHY`。
4. 把 `FaultRecord` 从 active 移到 completed。
5. phase 回到 `HEALTHY_TRAINING`。

如果还有其他 fault record 已经 ready，但没 repair 完，它会回到 `SAFE_POINT_REPAIR` 继续处理。

## 20. 一次完整恢复的代码路径

把上面内容合起来，一次 hybrid hot swap 可以按这个路径读：

```text
1. training.py
   train_step 抛 NCCL RuntimeError

2. training.py / elastic_client.py
   elastic_on_nccl_error 通知 watcher

3. moegambit_integration.py
   hard failure detector callback 调 RecoveryController.on_hard_rank_failure

4. recovery_controller.py
   创建 FaultRecord
   标记 experts RECOVERING
   invalidated 当前 iteration
   phase: HEALTHY_TRAINING -> PENDING_GROUP_REPAIR

5. hot_spare_pool.py
   allocate_spare: STANDBY -> ALLOCATED

6. replacement_registry.py
   announce_replacement: NOT_PRESENT -> BOOTSTRAPPING
   announce_replacement_ready: BOOTSTRAPPING -> READY_FOR_REPAIR

7. recovery_controller.py
   on_replacement_ready 让 phase 进入 SAFE_POINT_REPAIR

8. recovery_controller.py
   at_safe_point -> _execute_safe_point_repair

9. recovery_controller.py
   Phase A: group repair / directory refresh / topology refresh

10. recovery_controller.py
    Phase B: _execute_hybrid_recovery_path

11. moegambit_integration.py
    dense_sync_fn: dense/shared/router 从健康 DP peer 拉

12. moegambit_integration.py
    expert_restore_fn: expert weights 从 checkpoint 或 EDP peer 恢复

13. deferred_optimizer_load.py
    expert optimizer state deferred load，barrier 阻止提前更新

14. elastic_client.py / zero2_memory_checkpoint.py
    如果启用 ZeRO-2 memory replica，non-expert optimizer shard 从 holder 拉回

15. recovery_controller.py
    Phase C: reintegration barrier
    phase: SAFE_POINT_REPAIR -> REINTEGRATED

16. recovery_controller.py
    _finalize_reintegration
    experts -> HEALTHY
    phase: REINTEGRATED -> HEALTHY_TRAINING
```

## 21. 读者最容易误解的三个点

### 21.1 热替换不是立刻换 NCCL group

`HotSparePool.allocate_spare(...)` 明确不把 spare 加进 NCCL group。真正的 group repair 必须在 safe point。原因是 NCCL collective 顺序要求极强，不能让某些 rank 先看到新 group、某些 rank 还在旧 group。

### 21.2 dense sync 和 expert restore 不是同一种恢复

dense/shared/router 参数是 DP replicated，所以可以从健康 DP peer 拉当前状态。expert 参数按 expert ownership 分布，不能随便复制；它要么从 checkpoint 恢复，要么在 EDP>1 时从对应 expert DP peer 恢复。

### 21.3 ZeRO-2 memory replica 不是 checkpoint

`zero2_memory_checkpoint.py` 保存的是运行中 optimizer shard 的 host-memory snapshot，走 TCP ring。它不是磁盘 checkpoint，也不是 `torch_dist` checkpoint。它服务的是 hot swap 时快速恢复 failed logical rank 的 optimizer shard。

## 22. 建议断点或日志观察点

如果要调试一次恢复，建议按这些函数下断点或过滤日志：

| 位置 | 看什么 |
| --- | --- |
| `MegatronAdapter.prepare_launch` | feature 是否正确投影到命令和环境变量。 |
| `training.py` 的 `elastic_zero2_initialize` | ZeRO-2 replica manager 是否启动。 |
| `maybe_initialize_moegambit_moe` | callbacks 是否注册。 |
| `RecoveryController.on_hard_rank_failure` | fault record、expert ids、mid_iteration 是否正确。 |
| `HotSparePool.allocate_spare` | spare 是否被分配，是否只走 control channel。 |
| `ReplacementRegistry.announce_replacement_ready` | replacement 是否进入 ready 状态。 |
| `RecoveryController.at_safe_point` | 是否进入 `_execute_safe_point_repair`。 |
| `dense_sync_fn` | dense 参数数量、source DP rank、optimizer sync 是否符合预期。 |
| `expert_restore_fn` | restore plan、checkpoint path、restored experts。 |
| `_restore_zero2_optimizer_from_memory_peer` | holder rank、manifest hash、restore scope。 |
| `_finalize_reintegration` | barrier 是否放行，experts 是否回到 HEALTHY。 |

## 23. 读完这份文档后应该能回答的问题

1. `--hot-swap` 和 `--zero2` 是在哪一层变成环境变量的？
2. Megatron adapter 为什么会自动添加 `--use-distributed-optimizer`？
3. replacement rank 是什么时候同步参数的？
4. hard failure 为什么会导致 optimizer step 被跳过？
5. 为什么 dense 参数可以从 DP peer 拉，而 expert 参数不行？
6. two-phase recovery 里的 `WEIGHTS_READY` 和 `OPTIMIZER_PENDING` 有什么区别？
7. ZeRO-2 optimizer shard 是如何在 host memory 里复制到 holder rank 的？
8. replacement rank 如何验证拿到的 optimizer snapshot 没写错位置？
9. reintegration barrier 阻止了哪些过早恢复？

如果这些问题能顺着函数名找到答案，就说明已经从“理解整体逻辑”进入了“能读懂实现路径”的阶段。
