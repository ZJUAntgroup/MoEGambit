# MoEGambit Restart-in-Place Recovery

## 概述

Restart-in-Place（原地重启）是 MoEGambit 容错恢复栈的扩展模块，用于模拟 GPU 进程原地重启后的状态恢复。核心思路：**不替换物理节点，而是在同一 rank 上用 NaN 哨兵值填充所有张量来模拟内存丢失，然后走完整的恢复流程验证系统能否正确恢复**。

### 与传统 hard_failure 模式的区别

| 特性 | hard_failure | restart_in_place |
|------|-------------|-----------------|
| replacement_rank | 可以是新 rank | 必须 == failed_rank |
| 进程组重建 (A2-A5) | 需要 | 跳过 |
| 张量状态 | 不操作 | NaN 哨兵填充 |
| 验证门 | 无 | 恢复后必须通过 NaN/Inf 检查 |
| 适用场景 | 真实节点替换 | 进程原地重启、CI 测试 |

## 架构

```
故障注入 (step N)
    │
    ▼
invalidate_rank_tensors()          ← 所有 param.data + optimizer state 填 NaN
    │
    ▼
RecoveryController.on_hard_rank_failure(restart_in_place=True)
    │
    ├─ 自动: replacement_rank = failed_rank
    ├─ 自动: on_replacement_assigned + on_replacement_ready
    └─ 跳过: group_rebuild, topology_refresh (rank 不变)
    │
    ▼
gap-aware 路径选择
    │
    ├─ gap <= threshold → CHECKPOINT_RESTART (全量 checkpoint 恢复)
    └─ gap >  threshold → HYBRID_RECOVERY
    │                       ├─ dense params: 从 DP peer broadcast
    │                       ├─ expert params: 从 checkpoint 加载
    │                       └─ optimizer states: 分别恢复
    ▼
verify_recovery()                  ← NaN/Inf/shape/dtype 全量检查
    │
    ├─ PASS → HEALTHY (可重新参与训练)
    └─ FAIL → RecoveryVerificationError (阻止重入)
```

## 状态机

```
HEALTHY → INVALIDATED → WEIGHTS_RESTORING → WEIGHTS_READY
    → OPTIMIZER_RESTORING → FULLY_RECOVERED → (verify) → HEALTHY
```

每个状态的权限矩阵：

| 状态 | forward | backward | optimizer step |
|------|---------|----------|---------------|
| HEALTHY | 允许 | 允许 | 允许 |
| INVALIDATED | 阻止 | 阻止 | 阻止 |
| WEIGHTS_RESTORING | 阻止 | 阻止 | 阻止 |
| WEIGHTS_READY | 允许 | 阻止 | 阻止 |
| OPTIMIZER_RESTORING | 允许 | 阻止 | 阻止 |
| FULLY_RECOVERED | 允许 | 允许 | 阻止 |

## 文件清单

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `restart_in_place.py` | ~480 | 核心模块：NaN 注入、验证门、RestartInPlaceCoordinator |
| `test_restart_in_place.py` | ~700 | 18 个单元测试，CPU 张量，无需 GPU |

### 修改文件

| 文件 | 修改点 | 说明 |
|------|--------|------|
| `dense_param_sync.py` | 3 处 | fail-closed 语义 + `verify_synced_params()` |
| `recovery_controller.py` | 8 处 | `restart_in_place=True` 快速路径，跳过 A2-A5 |
| `stale_expert_restore.py` | 1 处 | dry-run 检测 NaN 哨兵并报告失败 |
| `moegambit_integration.py` | 3 处 | 注入类型支持、invalidate_tensor_fn 回调注册 |
| `arguments.py` | 1 处 | `--moe-moegambit-restart-in-place` 参数 |
| `transformer_config.py` | 1 处 | `moe_moegambit_restart_in_place` 配置字段 |

## 使用方法

### 1. 命令行参数

在 `torchrun` 命令中添加：

```bash
--moe-moegambit-fault-injection \
--moe-moegambit-restart-in-place \
```

### 2. 环境变量

```bash
# 注入类型（设置 --moe-moegambit-restart-in-place 后自动覆盖为 restart_in_place）
export MOEGAMBIT_FAULT_INJECT_TYPE=restart_in_place

# 在哪个 rank 注入故障
export MOEGAMBIT_FAULT_INJECT_RANK=0

# 在第几步注入故障
export MOEGAMBIT_FAULT_INJECT_STEP=50
```

### 3. 启动脚本

```bash
# 使用默认 restart_in_place 模式
bash run_moe.sh

# 或切换回传统 hard_failure 模式
MOEGAMBIT_FAULT_INJECT_TYPE=hard_failure bash run_moe.sh
```

### 4. 运行单元测试

```bash
cd Megatron-LM
python3 -m pytest tests/unit_tests/transformer/moe/test_restart_in_place.py \
    -x -v --noconftest
```

## API 参考

### restart_in_place.py

```python
# NaN 哨兵注入
invalidate_rank_tensors(model, optimizer=None) -> dict
invalidate_dense_params_only(model, optimizer=None, classification=None) -> dict
invalidate_expert_params_only(model, optimizer=None, classification=None) -> dict

# 恢复验证
verify_recovery(model, optimizer=None, check_optimizer=True) -> VerificationResult
# 失败时抛出 RecoveryVerificationError

# 协调器
coord = RestartInPlaceCoordinator(gap_threshold=100, checkpoint_iteration=0)
coord.inject_fault_and_invalidate(model, optimizer, step=N)
path = coord.execute_recovery(model, optimizer, ...)  # "CHECKPOINT_RESTART" | "HYBRID_RECOVERY"
result = coord.verify_and_reintegrate(model, optimizer, step=N+1)
```

### dense_param_sync.py (新增)

```python
# 同步后验证（检查 NaN/Inf）
ok, msg = verify_synced_params(model, classification=None)

# fail-closed: require_nonempty_state=True 时，空 optimizer state 会抛 RuntimeError
_sync_optimizer_states_for_dense(..., require_nonempty_state=True)
```

### recovery_controller.py (扩展)

```python
# 注册 NaN 注入回调
ctrl.register_callbacks(invalidate_tensor_fn=my_fn)

# 触发原地重启故障
ctrl.on_hard_rank_failure(failed_rank=0, step=100, restart_in_place=True)
# 自动: replacement_rank=0, 跳过 group rebuild / topology refresh

# 查询模式
ctrl.restart_in_place_mode  # bool
```

## 测试覆盖

| 测试类 | 测试数 | 覆盖场景 |
|--------|--------|----------|
| TestInvalidation | 3 | 全量/仅 dense/仅 expert NaN 注入 |
| TestVerification | 3 | 健康模型通过、NaN 失败、空 optimizer 失败 |
| TestCheckpointRestartPath | 1 | 小 gap → checkpoint 全量恢复 |
| TestHybridRecoveryPath | 2 | 大 gap → hybrid 恢复 + optimizer 正确性 |
| TestVerificationGate | 1 | 部分恢复（expert 仍 NaN）被阻止 |
| TestDenseParamSyncFailClosed | 2 | 空 optimizer state 抛异常 + override |
| TestVerifySyncedParams | 2 | 健康参数通过 + NaN 参数失败 |
| TestRecoveryControllerRestartInPlace | 2 | 跳过 group rebuild + invalidate 回调 |
| TestStaleExpertRestoreDryRunNaN | 1 | dry-run 检测 NaN 哨兵 |
| TestEndToEndRestartInPlace | 1 | 端到端：controller + coordinator + 真实张量 |

## 风险与注意事项

1. **NaN 哨兵 vs 零值**：使用 `float('nan')` 而非 `0.0`，因为零是合法参数值，NaN 不是。
2. **fail-closed 语义**：同步 0 个参数或空 optimizer state 会抛 RuntimeError，防止静默恢复失败。
3. **仅限单机模拟**：restart_in_place 模式下 replacement_rank == failed_rank，不涉及真实的进程替换。生产环境中需要 torchrun elastic 配合。
4. **optimizer state 时序**：optimizer 必须至少执行过一次 `step()` 才有 state，否则验证会报 "empty optimizer state"。
5. **fused_a2a.py 无需修改**：DeepEP all-to-all 通信层与 restart-in-place 无直接耦合，NaN 注入发生在参数层面，不影响通信原语。
