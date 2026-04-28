# BSR-MoE: Bounded Staleness Recovery for Mixture-of-Experts

BSR-MoE 是嵌入 Megatron-LM 训练循环的在线故障恢复系统。当 MoE 训练中某个 rank 发生故障时，系统无需停机重启，在降级模式下继续训练，同时异步完成故障 rank 的替换与专家参数恢复，最终在安全点（迭代边界）执行进程组修复和重集成。

## 目录

- [系统架构](#系统架构)
- [模块清单](#模块清单)
- [恢复流程](#恢复流程)
- [状态机](#状态机)
- [安全点修复序列](#安全点修复序列)
- [Soft Failure 与 Hard Failure](#soft-failure-与-hard-failure)
- [迭代回滚与重放](#迭代回滚与重放)
- [PP > 1 扩展](#pp--1-扩展)
- [降级模式策略](#降级模式策略)
- [配置项](#配置项)
- [训练循环集成](#训练循环集成)
- [故障注入测试](#故障注入测试)
- [当前限制](#当前限制)

---

## 系统架构

```
训练循环 (training.py)
    │
    ├── bsr_before_iteration(step)  ──→  RecoveryController.before_iteration()
    │                                         │
    │                                         ├── _execute_safe_point_repair()
    │                                         │     ├── 1. replacement_integrate_fn
    │                                         │     ├── 2. group_rebuild_request_fn
    │                                         │     ├── 3. group_rebuild_execute_fn  → SafePointGroupRepairer
    │                                         │     ├── 4. group_rebuild_finish_fn
    │                                         │     ├── 5. topology_refresh_fn       → DispatchTopologyManager
    │                                         │     ├── 6. dense_sync_fn             → DenseParamRecoveryCoordinator
    │                                         │     ├── 7. expert_restore_fn         → StaleExpertRestoreCoordinator
    │                                         │     └── 7b. pipeline_stage_repair_fn → PipelineStageRepairer (PP>1)
    │                                         │
    │                                         └── _finalize_reintegration()
    │                                               └── ReintegrationBarrier
    │
    ├── forward() / backward()
    │     └── Router 读取 ExpertHealthMask 排除不健康专家
    │
    ├── OptimizerCommitGuard → IterationInvalidator
    │     └── 若迭代被 invalidate 则跳过 optimizer.step()
    │
    └── bsr_after_iteration(step)
          ├── IterationInvalidator.end_iteration()
          ├── OptimizerCommitGuard.end_iteration()
          └── DeferredOptimizerLoader.poll_and_finalize()
```

核心设计原则：

- **回调驱动**：RecoveryController 不直接依赖 Megatron 内部实现，所有操作通过注册的回调函数执行
- **安全点修复**：所有破坏性操作（进程组销毁/重建、P2P 重绑定）仅在迭代边界执行
- **全局单例**：每个模块提供 `get_*()` / `clear_*()` 函数
- **纯 Python 可测试**：所有模块可在无 `torch.distributed` 初始化的环境下通过 stub 测试

---

## 模块清单

### 编排层

| 模块 | 文件 | 说明 |
|------|------|------|
| RecoveryController | `recovery_controller.py` | 核心状态机，协调整个恢复生命周期 |
| BSR 集成层 | `bsr_integration.py` | 连接所有模块到 Megatron 训练循环 |

### 故障检测与隔离

| 模块 | 文件 | 说明 |
|------|------|------|
| HardFailureDetector | `hard_failure_detector.py` | 检测集合通信故障 |
| RankQuarantineRegistry | `rank_quarantine.py` | Rank 级别隔离注册表 |
| ExpertHealthMask | `expert_health.py` | Per-layer 布尔掩码，路由器读取 |
| ExpertHealthManager | `expert_health_manager.py` | Per-expert 四态生命周期管理 |

### 替换与修复

| 模块 | 文件 | 说明 |
|------|------|------|
| ReplacementRegistry | `replacement_registry.py` | 替换 rank 生命周期注册表 |
| GroupRebuildCoordinator | `group_rebuild.py` | 进程组重建协调 |
| SafePointGroupRepairer | `safe_point_group_repair.py` | 安全点 NCCL 组销毁/重建/重绑定 |
| ActiveExpertDirectory | `expert_directory.py` | (layer, expert) → host_rank 映射 |
| DispatchTopologyManager | `dispatch_topology_refresh.py` | Token 分发拓扑刷新 |

### 参数恢复

| 模块 | 文件 | 说明 |
|------|------|------|
| DenseParamRecoveryCoordinator | `dense_param_sync.py` | 从健康 DP peer 拉取稠密参数 |
| StaleExpertRestoreCoordinator | `stale_expert_restore.py` | 从 checkpoint 恢复专家权重 |
| DeferredOptimizerLoader | `deferred_optimizer_load.py` | 异步加载专家优化器状态 |

### 迭代控制

| 模块 | 文件 | 说明 |
|------|------|------|
| IterationInvalidator | `iteration_invalidator.py` | 标记当前迭代无效 |
| RollbackReplayManager | `iteration_rollback.py` | 迭代边界快照、回滚、重放 |
| OptimizerCommitGuard | `optimizer_commit_guard.py` | 阻止无效迭代的优化器提交 |

### 降级与重集成

| 模块 | 文件 | 说明 |
|------|------|------|
| DegradedModePolicy | `degraded_mode_policy.py` | 三阈值降级续训策略引擎 |
| ReintegrationBarrier | `reintegration_barrier.py` | 5 前置条件门控的重集成屏障 |

### PP > 1 扩展

| 模块 | 文件 | 说明 |
|------|------|------|
| PipelineRollbackCoordinator | `pipeline_rollback.py` | 跨 stage 协调回滚 |
| PipelineStageRepairer | `pipeline_stage_repair.py` | PP 组重建 + P2P 通信器重绑定 |

### 测试

| 模块 | 文件 | 说明 |
|------|------|------|
| FaultInjectionFramework | `fault_injection_framework.py` | 确定性故障注入与指标采集 |

---

## 恢复流程

### Soft Failure（rank 存活但不可信）

```
故障检测 → 隔离 rank → 标记专家 UNAVAILABLE → 路由器屏蔽 → 降级训练
```

仅涉及 `RankQuarantineRegistry` + `ExpertHealthMask`。不需要替换 rank，不需要进程组修复。训练以降低的专家容量继续。

### Hard Failure（rank 无法参与集合通信）

```
故障检测 → 隔离 + 标记 UNAVAILABLE → 等待替换 rank → 替换 rank 引导
→ 安全点修复（7 步） → 重集成 → 恢复正常训练
```

Hard failure 复用 soft failure 的隔离和健康掩码机制，然后继续推进到进程组修复。

### 专家状态流

```
HEALTHY → UNAVAILABLE → STALE_RUNNABLE → FULLY_RECOVERED → HEALTHY
           (故障)        (checkpoint恢复)   (优化器加载)      (重集成)
```

- **UNAVAILABLE**：路由器排除，不参与计算
- **STALE_RUNNABLE**：参与前向/反向，但优化器更新被 `OptimizerUpdateBarrier` 阻止
- **FULLY_RECOVERED**：优化器状态已加载，等待重集成屏障放行

---

## 状态机

### RecoveryPhase（8 个状态）

| 状态 | 值 | 含义 |
|------|---|------|
| `HEALTHY_TRAINING` | 0 | 正常训练 |
| `DEGRADED_ISOLATION` | 1 | Rank 被隔离（软故障），降级训练 |
| `PENDING_GROUP_REPAIR` | 2 | 硬故障，等待替换 rank |
| `WAITING_FOR_REPLACEMENT` | 3 | 替换 rank 已分配，正在引导 |
| `SAFE_POINT_REPAIR` | 4 | 替换 rank 就绪，等待安全点修复 |
| `REINTEGRATED` | 5 | 修复已执行，等待最终化 |
| `PIPELINE_REBINDING` | 6 | PP 组重建 + P2P 重绑定（PP>1） |
| `ROLLBACK_PENDING` | 7 | PP>1 迭代中故障，等待协调回滚 |

### 转换图

```
HEALTHY_TRAINING ──┬── on_rank_quarantined() ──▶ DEGRADED_ISOLATION
                   │                                │
                   ├── on_hard_rank_failure() ──▶ PENDING_GROUP_REPAIR
                   │                                ▲
                   └── on_pipeline_stage_failure()   │
                       (mid_iteration) ──▶ ROLLBACK_PENDING ──┘

DEGRADED_ISOLATION ──┬── on_hard_rank_failure() ──▶ PENDING_GROUP_REPAIR
                     └── (resolved) ──▶ HEALTHY_TRAINING

PENDING_GROUP_REPAIR ── on_replacement_assigned() ──▶ WAITING_FOR_REPLACEMENT
                                                          │
WAITING_FOR_REPLACEMENT ── on_replacement_ready() ──▶ SAFE_POINT_REPAIR
                                                          │
SAFE_POINT_REPAIR ── before_iteration() ──┬── (PP=1) ──▶ REINTEGRATED
                                          └── (PP>1) ──▶ PIPELINE_REBINDING
                                                              │
PIPELINE_REBINDING ──────────────────────────────────▶ REINTEGRATED
                                                          │
REINTEGRATED ── before_iteration() ──▶ HEALTHY_TRAINING
```

---

## 安全点修复序列

在 `before_iteration()` 中，当状态为 `SAFE_POINT_REPAIR` 时执行 7+1 步修复：

| 步骤 | 回调 | 操作 |
|------|------|------|
| 1 | `replacement_integrate_fn` | 在 ReplacementRegistry 中标记替换 rank 为 INTEGRATED |
| 2 | `group_rebuild_request_fn` | 计算新的 EP/DP 组 rank 列表（failed → replacement） |
| 3 | `group_rebuild_execute_fn` | 销毁旧 NCCL 组 → 创建新组 → 重绑定模块引用 → barrier 验证 |
| 4 | `group_rebuild_finish_fn` | 更新 GroupRebuildCoordinator 内部状态 |
| 5 | `topology_refresh_fn` | 刷新 expert→rank 映射、健康掩码、分发拓扑 |
| 6 | `dense_sync_fn` | 从健康 DP peer broadcast 拉取稠密/路由器/共享专家参数 |
| 7 | `expert_restore_fn` | 从 checkpoint 加载专家权重，标记 STALE_RUNNABLE |
| 7b | `pipeline_stage_repair_fn` | PP 组重建 + P2P 通信器重绑定（仅 PP>1） |

步骤 6 和 7 体现了统一恢复语义：
- **稠密参数**：所有 DP rank 持有相同副本，从任意健康 peer 拉取（零陈旧性）
- **MoE 专家参数**：仅存在于特定 EP rank，只能从 checkpoint 恢复（有界陈旧性）

---

## Soft Failure 与 Hard Failure

### Soft Failure

- 入口：`RecoveryController.on_rank_quarantined()`
- 行为：隔离 rank + 标记专家 UNAVAILABLE + 路由器屏蔽
- 状态：停留在 `DEGRADED_ISOLATION`
- 不需要替换 rank，不需要进程组修复
- NCCL 集合通信不受影响（故障 rank 仍然参与）

### Hard Failure

- 入口：`RecoveryController.on_hard_rank_failure()`
- 行为：复用 soft failure 的隔离 + 健康标记，然后推进到 `PENDING_GROUP_REPAIR`
- 需要替换 rank 和完整的安全点修复序列
- 若 `mid_iteration=True`，当前迭代被标记无效，跳过优化器提交

---

## 迭代回滚与重放

当硬故障发生在迭代中间时：

1. **IterationInvalidator** 标记当前迭代无效
2. **OptimizerCommitGuard** 阻止 `optimizer.step()` 提交
3. **RollbackReplayManager** 回滚到上一个快照（恢复 iteration、consumed_samples、数据迭代器位置）
4. 训练循环 `continue`，不递增 iteration
5. 下一次 `before_iteration()` 在安全点执行修复
6. 修复完成后重放被跳过的迭代

```python
# 训练循环中的集成
bsr_snapshot_iteration(iteration, consumed_train_samples, ...)
bsr_before_iteration(iteration)

try:
    loss, ... = train_step(...)
except RuntimeError as e:
    if is_nccl_error(e):
        bsr_report_hard_failure(...)

if bsr_is_current_iteration_invalid():
    bsr_rollback_iteration(args, data_iterators, ...)
    continue  # 不递增 iteration

if bsr_is_replay_pending():
    bsr_complete_replay(data_iterators)

bsr_after_iteration(iteration)
```

---

## PP > 1 扩展

### Pipeline-safe Rollback

PP > 1 时，多个 stage 同时执行不同 microbatch。若某个 stage 故障：

1. `on_pipeline_stage_failure()` → 状态转入 `ROLLBACK_PENDING`
2. `PipelineRollbackCoordinator.initiate_rollback()` 协调所有 stage 同步回滚
3. 清除所有 stage 的梯度缓冲区
4. `invalidate_inflight_microbatches()` 标记所有在途 microbatch 无效
5. 转入 `PENDING_GROUP_REPAIR`，后续流程与 PP=1 相同

### Pipeline Stage Repair

安全点修复序列的步骤 7b，`PipelineStageRepairer` 执行 6 步：

1. 销毁旧 PP 组（`_PIPELINE_MODEL_PARALLEL_GROUP` 等 6 个组变量）
2. 用新 rank 列表创建新 PP 组
3. 更新 `_PIPELINE_GLOBAL_RANKS`（prev/next rank 缓存）
4. 重建 P2P 通信器（send/recv）
5. 重建包含 PP 维度的复合组
6. Barrier 验证

状态路径：`SAFE_POINT_REPAIR → PIPELINE_REBINDING → REINTEGRATED`

---

## 降级模式策略

`DegradedModePolicy` 通过三个阈值决定是否允许降级续训：

| 阈值 | 公式 | 默认值 | 含义 |
|------|------|--------|------|
| 容量阈值 τ_c | C = healthy / total | 0.5 | 低于此值不允许降级续训 |
| 迭代预算 T_max | degraded_steps > T_max | 1000 | 超过则必须等待恢复 |
| 陈旧度阈值 S_max | S = step - ckpt_step | 500 | 超过则强制恢复 |

附加观测指标：**SLT**（Stale-Load Token ratio）= 路由到陈旧专家的 token 数 / 总路由 token 数。

---

## 配置项

### Megatron 命令行参数

| 参数 | 说明 |
|------|------|
| `--moe-bsr-enable` | 启用 BSR-MoE |
| `--moe-bsr-health-mask` | 启用专家健康掩码 |
| `--moe-bsr-rank-quarantine` | 启用 rank 隔离 |
| `--moe-bsr-expert-directory` | 启用专家目录 |
| `--moe-bsr-replacement-protocol` | 启用替换协议 |
| `--moe-bsr-group-rebuild` | 启用进程组重建 |
| `--moe-bsr-dispatch-topology-refresh` | 启用分发拓扑刷新 |
| `--moe-bsr-dense-param-sync` | 启用稠密参数同步 |
| `--moe-bsr-stale-expert-restore` | 启用专家恢复 |
| `--moe-bsr-recovery-controller` | 启用恢复控制器 |
| `--moe-bsr-deferred-optimizer-load` | 启用延迟优化器加载 |
| `--moe-bsr-degraded-mode-policy` | 启用降级模式策略 |
| `--moe-bsr-reintegration-barrier` | 启用重集成屏障 |
| `--moe-bsr-fault-injection` | 启用故障注入 |
| `--moe-bsr-degraded-tau-c FLOAT` | 容量阈值（默认 0.5） |
| `--moe-bsr-degraded-t-max INT` | 最大降级迭代数（默认 1000） |
| `--moe-bsr-degraded-s-max INT` | 最大陈旧度（默认 500） |
| `--moe-bsr-dispatch-quarantine-assert` | 启用分发隔离断言 |
| `--moe-bsr-dispatch-sanitize` | 启用分发清洗 |

### 故障注入环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BSR_FAULT_INJECT_TYPE` | quarantine | 故障类型：`quarantine` / `hard_failure` |
| `BSR_FAULT_INJECT_RANK` | 0 | 目标 rank |
| `BSR_FAULT_INJECT_STEP` | 50 | 注入步骤 |
| `BSR_FAULT_REPLACEMENT_STEP` | 60 | 替换就绪步骤 |
| `BSR_FAULT_REPLACEMENT_RANK` | -1 | 替换 rank（-1 自动分配） |

---

## 训练循环集成

### 初始化

```python
from megatron.core.transformer.moe.bsr_integration import (
    maybe_initialize_bsr_moe,
    bsr_before_iteration,
    bsr_after_iteration,
)

# 模型/优化器初始化之后、训练循环之前
maybe_initialize_bsr_moe(model, args)
```

### 训练循环

```python
for step in range(num_steps):
    bsr_before_iteration(step)       # 安全点钩子（可触发修复）
    bsr_snapshot_iteration(step, consumed_train_samples, ...)

    try:
        loss, ... = train_step(...)
    except RuntimeError as e:
        if is_nccl_error(e):
            bsr_report_hard_failure(failed_rank=-1, step=step, mid_iteration=True)

    if bsr_is_current_iteration_invalid():
        bsr_rollback_iteration(args, data_iterators, ...)
        bsr_after_iteration(step)
        continue

    if bsr_is_replay_pending():
        bsr_complete_replay(data_iterators)

    bsr_after_iteration(step)
    iteration += 1
```

### Checkpoint 集成

```python
# 保存前注入 BSR 元数据
bsr_pre_save_checkpoint(checkpoint_dir, step)

# 加载后恢复 BSR 状态
bsr_post_load_checkpoint(checkpoint_dir, step)

# 保存 manifest 侧车文件（bsr_manifest.json）
bsr_save_manifest(checkpoint_dir, step)
```

---

## 故障注入测试

### 运行脚本配置

```bash
# 启用 soft failure 注入（不破坏 NCCL 通信）
export BSR_FAULT_INJECT_TYPE=quarantine
export BSR_FAULT_INJECT_STEP=50
export BSR_FAULT_INJECT_RANK=0

# 启用 hard failure 注入（需要真实替换 rank）
export BSR_FAULT_INJECT_TYPE=hard_failure
export BSR_FAULT_INJECT_STEP=50
export BSR_FAULT_REPLACEMENT_STEP=60
```

### 单元测试

```bash
cd Megatron-LM
# 运行所有 BSR-MoE 单元测试
python -m pytest tests/unit_tests/transformer/moe/test_safe_point_repair.py -v
python -m pytest tests/unit_tests/transformer/moe/test_pp_e2e_recovery.py -v
python -m pytest tests/unit_tests/transformer/moe/test_e2e_fault_injection.py -v
```

### 端到端故障注入框架

`fault_injection_framework.py` 提供确定性故障注入和指标采集：

```python
from megatron.core.transformer.moe.fault_injection_framework import (
    FaultInjector, FaultScenario, FaultType,
    RecoveryMetricsCollector, run_simulated_training_loop,
    build_pp1_hard_failure_scenario,
)

# 构建 PP=1 hard failure 场景
scenarios = build_pp1_hard_failure_scenario(
    fault_step=10, replacement_step=15, failed_rank=2,
)

# 运行模拟训练循环并采集指标
result = run_simulated_training_loop(config, scenarios)
print(result.metrics[0].format_report())
```

采集的指标包括：`time_to_detect`、`time_to_replacement`、`time_to_group_repair`、`time_to_resume`、`degraded_steps`、`invalidated_steps`、`rollback_count`、`replay_count`。

---

## 当前限制

1. **单故障串行恢复**：一次只处理一个 FaultRecord，后续故障排队等待
2. **无自动故障检测**：调用方必须显式调用 `on_rank_quarantined()` / `on_hard_rank_failure()`
3. **无自动替换 rank 调度**：替换 rank 的分配和引导由外部调度器负责
4. **无备用 rank 池**：当前没有预留 spare rank 的机制，hard failure 注入在真实训练中需要外部提供替换进程
5. **快照不含模型参数**：RollbackReplayManager 仅保存标量元数据
6. **进程组修复是全局阻塞的**：修复期间所有 rank 暂停训练
7. **PP 拓扑固定**：PipelineStageRepairer 假设 PP 组大小不变
8. **数据迭代器回滚依赖外部支持**：假设数据迭代器支持位置保存/恢复
