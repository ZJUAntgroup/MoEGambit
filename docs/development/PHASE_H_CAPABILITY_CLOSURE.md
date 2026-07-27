# Phase H：非 DeepSpeed 通用能力闭环

## 结论

本阶段补齐了设计文档中在 Phase B–F 之后仍缺少的五组通用能力：

1. adapter 提供候选状态源，核心 policy 决策并冻结确定性来源；
2. optimizer 内存复制的 buffer、TCP transport、版本和 locator 从 Megatron
   内嵌实现提取到通用包；
3. watcher 的 frozen plan、commit、failure 和 relaunch directive 可以持久化，
   watcher 重启后不会丢失；
4. checkpoint fallback 从“只记录请求”接到 NodeAgent 的真实冷重启；
5. `launch`、`watcher`、`doctor` CLI 和 Generic DDP checkpoint 示例形成可执行入口。

这些结果不等同于生产环境 Supported。当前本机缺少 PyTorch/CUDA/NCCL，
Megatron 多机热替换与 optimizer TCP 张量环网仍需在目标集群完成故障矩阵。

## 1. 状态源与恢复策略

新增 `moegambit.policy`：

- `StateSourceCandidateProvider`：adapter 描述每个 `StateRef` 当前有哪些 peer、
  memory replica 或 checkpoint 来源；
- `RecoveryPolicy`：只读取标准化事实，不接触 framework tensor；
- `PeerOrCheckpointPolicy`：优先完整且同版本的 live state，否则使用完整 checkpoint；
- `MoeHybridPolicy`：non-unique state 使用当前 live source，unique expert 使用
  checkpoint，并计算可审计的 expert staleness density；
- `DeterministicStateSourcePlanner`：在 policy 选定 mode 后，为每个 identity 冻结
  一个稳定 locator；
- `StateSourceResolver`：执行端通过显式注入把 locator 解析为 adapter 可消费对象。

恢复请求摘要现在覆盖：catalog、candidate sources、locator、version、checkpoint
step、exposure history 和 capabilities。修改 memory holder、peer rank 或 checkpoint
位置会改变摘要；同一 epoch 的参与者不能执行不同来源。

兼容路径只会自动推断 replicated peer state。sharded 与 unique 状态没有 adapter
证据时仍然 fail closed，不会根据 rank 数量猜测所有权。

## 2. 通用 optimizer memory replication

新增 `moegambit.replication`：

- adapter 只枚举 `OptimizerTensorRef` / `OptimizerScalarRef`；
- 通用 manager 管理两个本地 staging slot 和两个 peer receive slot；
- TCP ring 不使用 NCCL/c10d，不改变训练 collective 顺序；
- endpoint connection 支持 bounded retry 和 timeout；
- header、segment 数量和总 payload 有显式上限；
- 每个 payload segment 携带 SHA-256，接收后校验；
- ACK 同时核对 step 与 manifest hash；
- snapshot 带 generation、owner、holder 和 committed step；
- `MemoryReplicaLocator` 可稳定序列化进 `RecoveryPlan.state_sources`。

Megatron 原路径 `megatron.training.zero2_memory_checkpoint` 变为兼容导入层，
实际实现由 `moegambit.replication` 提供。旧 `zero2-memory:*` rendezvous key 保留，
避免滚动迁移期间破坏 watcher 端点发现。

## 3. 持久化 ControlStore

新增 `SQLiteControlStore`，提供：

- monotonic epoch；
- 同 epoch 仅允许完全相同的幂等写，不允许 last-writer-wins；
- transaction 内 compare-and-set；
- watcher 重启后重新加载 frozen assignment；
- 多个 watcher 进程竞争同一 SQLite 文件时，只能冻结一个同 epoch 计划；
- commit、failure、fallback audit 和 relaunch ACK 的 CAS 追加。

默认 watcher CLI 使用 SQLite，单元测试可显式选择 in-memory store。

边界必须明确：SQLite 解决单机多进程和 watcher 重启持久化，不是跨主机 HA
共识系统。跨主机 active/standby watcher 需要实现同一个 `ControlStore` Protocol，
映射到平台 KV、etcd、Redis/Valkey 或其他具备线性化 CAS 的服务。不要把 SQLite
数据库放到不保证 POSIX locking/WAL 语义的共享文件系统上，并声称获得 HA。

## 4. checkpoint relaunch 闭环

冷重启流程为：

```text
训练循环确认 checkpoint 已完整落盘
  -> runtime.record_checkpoint(locator, resume_step)
  -> in-process recovery 失败
  -> watcher 冻结 RelaunchDirective
  -> NodeAgent heartbeat 获取 directive
  -> 校验 parent attempt 和 command digest
  -> 终止旧 attempt 的本节点全部 worker
  -> 生成 next_attempt_id
  -> 注入 checkpoint locator/step，并可追加 framework CLI 参数
  -> 启动新 worker
  -> 向旧 attempt scope 回传 ACK
  -> 后续 heartbeat 切换到新 attempt scope
```

系统不会把失败时的 `at_step` 自动当成 checkpoint。只有在原子 rename 或等价的
durability barrier 之后调用 `record_checkpoint`，该 checkpoint 才有资格成为重启
目标。缺少 locator/step 的 fallback 请求会被 watcher 拒绝。

`elastic_launcher.py --moegambit-runtime ...` 是新 NodeAgent 的兼容入口；不带该
开关时仍走旧 launcher，避免在目标集群验证前静默改变现有实验启动语义。

## 5. CLI 与接入示例

安装包提供：

- `moegambit-launch`：节点 worker ownership、logical rank replacement 和 checkpoint
  cold relaunch；
- `moegambit-watcher`：HMAC control service、policy、persistent store 和 rendezvous
  assignment；
- `moegambit-doctor`：只读检查配置、adapter 依赖、SQLite 路径和安全边界。

`examples/generic_ddp/train_loop.py` 展示两类 timing：

- iteration boundary / optimizer before / optimizer after / commit / error；
- checkpoint 原子发布后调用 `record_checkpoint`，以及 cold attempt 启动前加载
  NodeAgent 注入的 checkpoint。

## 6. 已验证范围

本阶段测试覆盖：

- peer、memory replica、checkpoint、hybrid source selection；
- mixed version fallback 和 missing source abort；
- source locator 进入 request/plan digest；
- SQLite same-epoch split-brain、CAS、watcher restart 和 concurrent writer；
- relaunch directive 持久化、heartbeat、command digest 和 ACK；
- NodeAgent 全 worker cold relaunch、attempt/env/argv 传递；
- bounded transport framing、payload digest、stable memory locator；
- package/control/agent/replication 在无 torch 环境 lazy import；
- 原 Phase B–E 非 torch 单元测试回归。

PyTorch optimizer tensor ring 测试在安装 torch 的环境执行；没有 torch 时明确 skip，
不是伪造通过。

2026-07-28 在当前开发机完成的最终复核结果：

- `python3 -m pytest -q -rs`：`147 passed, 2 skipped`；
- skip 1：未安装 PyTorch，`test_zero2_memory_checkpoint.py` 不收集真实 tensor ring；
- skip 2：当前执行沙箱不允许测试创建 loopback listener；其余基于同一
  `ControlRequestProcessor` 的签名回环和 client/server 契约测试通过；
- `compileall` 与 `git diff --check` 通过；
- 使用符合 `pyproject.toml` 的 setuptools 82.0.1 构建出
  `moegambit-0.1.0.dev0-py3-none-any.whl`；
- wheel 包含 `LICENSE`、`LEGAL.md`、policy、replication、三个 CLI 和完整
  Megatron adapter，不包含 `DeepSpeed/` 或 `deepspeed_adapter/`；
- wheel 安装到隔离目录后，核心、policy、replication、control 和三个 CLI
  均可导入/启动 help，且没有隐式导入 torch。

系统自带 setuptools 58 不能解析本项目 PEP 621 metadata，会错误生成
`UNKNOWN-0.0.0`；这不是可发布 artifact。构建环境必须遵守
`pyproject.toml` 中 `setuptools>=77` 的要求。

## 7. 未完成或仍需实机证明

以下内容不能从本阶段本机测试推导为完成：

1. Megatron adapter 仍使用 `MegatronRecoveryDriver` 桥接原热备状态机。通用 policy
   和 resolver 已可供 adapter 使用，但尚未在目标集群证明“拆掉桥接后由通用
   RecoveryExecutor 分阶段执行”与原行为等价。
2. 同一 recovery epoch 仍只接受一个 failed logical rank；多 rank 同时故障未实现。
3. SQLite 不提供跨主机 watcher HA。
4. Generic DDP NCCL、Megatron TP/PP/EP、ZeRO-2 memory replica 和连续故障矩阵
   仍需 GPU 集群验证。
5. DeepSpeed 目录和同事 adapter 本阶段未修改；最终 system 分支仍需解决第二份
   顶层 `moegambit` 包身份。

因此当前正确标签仍是：Generic DDP 与 Megatron adapter 为 Experimental；未跑完
目标实机矩阵前不得标为 Supported。
