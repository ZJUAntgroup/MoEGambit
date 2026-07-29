# Phase C 实现与验证记录

## 实现结论

Phase C 已建立一条不依赖 torch 或训练框架的完整恢复控制链：

```text
signed ControlClient
        ↓
ControlRequestProcessor / RecoveryCoordinatorService
        ↓ frozen RecoveryPlan + digest
WatcherRecoveryCoordinator
        ↓
RecoveryRuntime / RecoveryExecutor
        ↓
FrameworkAdapter 四类窄协议
```

测试中的 survivor 首先提交故障事实并取得冻结计划；replacement 使用另一个
签名客户端按 epoch 和 plan digest 获取同一计划。两者分别执行恢复、进入
provisional，并在同一个 post-recovery step 得到控制面确认后提交。

## 已实现模块

### 控制协议和 watcher

- `control/protocol.py`
  - protocol version；
  - job、attempt、recovery epoch 和 sender 身份；
  - request id 和响应关联；
  - HMAC-SHA256 与时间窗口；
  - 有界 message id 和消息大小。
- `control/watcher.py`
  - newline-delimited TCP 服务；
  - HMAC 验证；
  - request id 幂等缓存；
  - 相同 message id 携带不同内容时拒绝；
  - 显式 bind interface，禁止默认 wildcard。
- `control/state_store.py`
  - monotonic epoch 写入；
  - compare-and-set；
  - 有界等待和超时；
  - 同 epoch 完整性检查。
- `control/service.py`
  - 每个 job/attempt/epoch 冻结一个 RecoveryPlan；
  - 完整 state catalog、capability、拓扑和版本进入请求摘要；
  - 同 epoch 不同事实按 split-brain 拒绝；
  - 较新 epoch 冻结后拒绝旧 epoch；
  - commit step 和 plan digest 一致性检查；
  - checkpoint relaunch 请求留存审计记录。

### 训练侧运行时

- `runtime/client.py`
  - 有界请求与响应；
  - HMAC、job、attempt、epoch 和 request id 全部校验；
  - survivor 提交事实和 replacement 获取冻结计划使用相同契约。
- `runtime/lifecycle.py`
  - `clean → recovering → provisional → committed` 状态机；
  - nested recovery、旧 epoch 和未完成迭代提交均拒绝。
- `runtime/executor.py`
  - 按 quiesce、prepare、group rebuild/rebind、state restore、transient reset、resume、warmup 顺序执行；
  - plan digest、protocol、epoch、topology generation、manifest 和状态验证任一不一致即失败。
- `runtime/runtime.py`
  - lifecycle hooks；
  - typed failure classification；
  - survivor 与 replacement 两种入口；
  - provisional epoch 的二阶段提交；
  - checkpoint relaunch fallback；
  - lazy `moegambit.initialize` 公共入口。

### Agent 和进程生命周期

- `agent/worker_supervisor.py`
  - argv 启动且 `shell=False`；
  - 每个 logical rank 只有一个存活 worker；
  - generation 计数；
  - TERM/KILL 有界退出。
- `agent/node_agent.py`
  - 生成标准 distributed 环境；
  - 清理继承的 torch elastic rendezvous 状态；
  - replacement 保留 logical rank；
  - recovery epoch 和 frozen plan digest 注入 replacement 环境；
  - 部分启动失败时回收已启动 worker。
- `cli/launch.py`
  - 提供不经过 shell 的轻量节点启动入口；
  - 一个 worker 退出不会隐式改变其他 worker 的 logical rank。

### 观测和回退

- `observability/events.py` 提供稳定 JSON RecoveryRecord；
- `observability/metrics.py` 提供结果计数、epoch gauge 和阶段延迟；
- `runtime/fallback.py` 将 checkpoint relaunch 明确建模为外部动作；当前进程提出 fallback 后仍返回“未完成进程内恢复”，不会谎报恢复成功。

## 验证范围

Phase C 单元和端到端测试覆盖：

- HMAC 正确、篡改、过期和 protocol mismatch；
- 请求/响应关联和 message id 幂等；
- message id 携带不同内容；
- 消息大小上限；
- monotonic epoch 与 CAS；
- state catalog 进入冻结摘要；
- 同 epoch split-brain；
- 旧 epoch、错误 plan digest 和错误 commit step；
- fake adapter 完整执行顺序；
- topology generation、manifest 和状态验证失败；
- survivor/replacement 取得同一签名计划并在同一步提交；
- checkpoint fallback 不冒充进程内恢复；
- NodeAgent logical rank replacement；
- control、agent 和公共 initialize 在无 torch 环境导入；
- 真实本机 TCP 回环 HMAC 请求/响应。

## 明确限制

以下内容没有被当时 Phase C 的通过结果覆盖。后续 Phase H 已补齐其中的通用代码，
详见 `PHASE_H_CAPABILITY_CLOSURE.md`；本节保留为 Phase C 历史边界，不能继续用来
描述当前 `runtime` HEAD：

1. 通用计划服务当前只支持单个 failed rank 和 `replicated` peer state。
   `sharded` 与 `unique` state 当时会 fail closed。Phase H 已加入显式 candidate
   inventory、policy、deterministic planner 和 resolver；adapter 不提供证据时仍
   fail closed。
2. `InMemoryControlStore` 只适用于单 watcher 和单进程测试。跨 watcher HA
   需要映射到平台存储或具备一致性的外部 store。Phase H 已实现 SQLite 单机
   持久化/CAS；跨主机 HA 仍需平台 store。
3. checkpoint fallback 已完成签名请求、审计与运行时语义，但旧 launcher
   的实际退出/relaunch 接线要在后续兼容迁移时完成。Phase H 已接通新 NodeAgent
   冷重启与 ACK；旧 launcher 通过显式开关进入新路径。
4. NodeAgent 已具备启动和 replacement 原语，但尚未替换 main 中的旧
   `elastic_launcher.py`；旧路径 shim 和 Megatron 接入属于 Phase D。Phase H
   已增加 `--moegambit-runtime` 显式 shim；不带开关仍保留旧 launcher 行为。
5. 没有进行 GPU、NCCL、Megatron 或真实多机验证。因此这些能力仍是
   Experimental，不能标记为生产 Supported。
6. ZeRO-2 测试需要 PyTorch；当前本地无 torch 环境不执行该测试。
