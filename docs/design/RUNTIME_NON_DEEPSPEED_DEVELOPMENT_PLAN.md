# MoEGambit 非 DeepSpeed 运行时二次开发方案

> 状态：Proposed，作为 `runtime` 分支开发基线
>
> 日期：2026-07-27
>
> 开发基线：`origin/main@1c63067`
>
> 架构参考：`origin/generalize@42adebd`
>
> 本文范围：通用核心、控制面、Megatron 适配、Generic DDP、公共接入示例、测试与发布工程

## 1. 结论与开发决策

本分支采用以下产品结构：

> **通用核心包负责恢复规则和执行流程，框架适配器负责操作框架内部对象，显式接入层负责在训练循环的正确时机调用运行时。**

本轮开发确认以下决策：

1. `runtime` 从最新 `origin/main` 创建，保留 main 上已有实验、日志和 DeepSpeed 工作。
2. `origin/generalize` 作为非 DeepSpeed 实现和提交拆分的主要参考，但不整体 merge 或批量 cherry-pick。
3. `src/moegambit` 是本轮新增的通用核心包；核心不得 import Megatron 或 DeepSpeed。
4. Megatron 私有通信组、模型、optimizer、MoE 路由与 checkpoint 语义只能出现在 `adapters/megatron`。
5. Generic DDP 作为通用契约的独立验收实现，证明公共接口不是 Megatron API 的改名。
6. 本轮不修改 `DeepSpeed/`、`deepspeed_adapter/` 及 DeepSpeed 实机脚本，不评价或重构同事当前方案。
7. 由于 main 现有 `deepspeed_adapter/moegambit` 与计划中的 `src/moegambit` 同名，最终集成前必须单独评审包名与依赖关系；本轮只登记风险，不擅自处理同事目录。
8. 所有快速恢复能力均 fail closed：计划、拓扑、状态版本或 adapter 能力不满足时，不得继续执行不确定训练。

## 2. 当前基线

### 2.1 main 已有能力

main 的 Megatron 路径已经实现一套内嵌式恢复系统：

- 根目录 `elastic_launcher.py` 负责启动训练 worker 和节点控制代理；
- 根目录 `elastic_watcher.py` 负责心跳、故障处理、replacement 和 fallback；
- `Megatron-LM/megatron/training/elastic_client.py` 负责训练侧通信、group rebuild、rebind 和状态传输；
- `zero2_memory_checkpoint.py` 负责 optimizer host/TCP 内存副本；
- `moegambit_integration.py` 和 MoE 模块负责 quarantine、rollback、expert restore 与 reintegration；
- `training.py`、`parallel_state.py`、`router.py`、`token_dispatcher.py` 等文件包含显式恢复钩子。

这些代码已有纵向功能，但框架无关逻辑与 Megatron 私有逻辑混合，无法独立安装或复用。

### 2.2 generalize 可复用成果

generalize 已验证以下架构方向：

- `pyproject.toml` + `src/moegambit` 可安装包；
- lazy import，不安装 torch 也可 import 包和控制面；
- `TopologySpec`、`StateCatalog`、`StateVersion`、`RecoveryPlan`；
- 四个窄 adapter Protocol；
- versioned control protocol、HMAC、request id、epoch 与幂等处理；
- `RecoveryRuntime`、`RecoveryExecutor` 和 checkpoint fallback；
- Megatron legacy 路径迁入 adapter，旧路径保留 forwarding shim；
- Generic DDP wrapper/reducer 替换和 peer state restore；
- wheel、插件发现、文档和不依赖 PyTorch 的测试。

本轮应复用这些已经完成的设计与测试，不再从零重新定义第二套核心。

## 3. 范围与非目标

### 3.1 本轮范围

本分支负责：

- Python distribution、公共 API、配置和 capability；
- launcher、node agent、worker supervisor；
- watcher 控制协议、服务、协调器和状态存储；
- recovery epoch、不可变计划、执行器、fallback；
- topology、group manifest、store 和 PyTorch compat；
- state identity、placement、version、source 和 transfer；
- framework-neutral optimizer replication transport；
- recovery policy、事件、指标和 RecoveryRecord；
- MegatronAdapter 及 Megatron 训练循环接入；
- Generic DDP reference adapter 和故障验收程序；
- CLI、doctor、package build、文档和测试。

### 3.2 明确不在本轮实施

- 不修改 `DeepSpeed/`；
- 不修改 `deepspeed_adapter/`；
- 不实现或移植 DeepSpeedAdapter；
- 不修改 `test_deepspeed_*` 实机脚本和日志；
- 不决定 DeepSpeed 最终是内置 adapter 还是独立插件；
- 不实现动态扩缩容或改变 world size；
- 不重新实现 NCCL、AllReduce、AllToAll 或 MoE dispatch kernel；
- 不承诺任意 PyTorch/NCCL 版本可安全重建 ProcessGroup；
- 不在机械迁移阶段顺便改变原 Megatron 恢复语义。

## 4. 目标架构

```text
User Training Loop / Framework Integration
        │
        │ lifecycle hooks
        ▼
RecoveryRuntime
├── lifecycle / recovery epoch
├── frozen RecoveryPlan
├── RecoveryExecutor
├── policy / fallback
└── observability
        │
        │ FrameworkAdapter contract
        ▼
Framework Adapter
├── MegatronAdapter
└── GenericDDPAdapter
        │
        ▼
Megatron / torch.nn.parallel.DistributedDataParallel

Control Plane and Node Agent operate beside the runtime:

Watcher Service ↔ Runtime Client ↔ Node Agent ↔ Worker Processes
```

### 4.1 依赖方向

依赖必须保持单向：

```text
integration → adapter → core
```

同时运行时调用表现为：

```text
integration → RecoveryRuntime → adapter → framework objects
```

约束如下：

- `control/`、`agent/`、`policy/`、`observability/` 模块级不得 import torch；
- 核心目录不得 import Megatron；
- Megatron 私有字段只允许位于 `adapters/megatron/`；
- 训练循环不得自行推导恢复计划；
- adapter 不得自行生成与其他 rank 不同的恢复策略；
- PyTorch 私有 c10d 操作集中在版本敏感 compat 层。

## 5. 公共契约

### 5.1 TopologySpec

统一描述当前分布式拓扑：

```python
TopologySpec(
    world_size,
    rank,
    logical_axes,       # dp/tp/pp/ep/cp/etp
    coordinates,
    groups,
    generation,
    manifest_hash,
)
```

每个 `GroupSpec` 至少记录：

```text
name
ranks
backend
purpose
creation_ordinal
```

所有 rank 必须对全局 group manifest 达成一致，创建顺序不一致时快速失败。

### 5.2 StateCatalog 与 StateVersion

adapter 将框架状态转换为稳定的 `StateRef`：

```text
identity
kind
placement       replicated / sharded / unique
owner
version
shape / dtype
tags
tensor 或 scalar accessor
```

状态版本使用：

```text
committed_step
optimizer_generation
recovery_epoch
```

禁止恢复出以下不一致组合：

```text
parameters@k+1 + optimizer@k
```

### 5.3 RecoveryPlan

控制面为所有参与者冻结同一个恢复计划：

```text
protocol_version
recovery_epoch
failed_ranks
replacements
resume_step
mode
topology_generation
group_manifest_hash
state_sources
policy_evidence
timeout_budget
```

`state_sources` 必须明确每份状态来自：

```text
peer
checkpoint
local
memory replica
reinitialize
```

source 的 kind、locator、version 和 metadata 必须进入计划摘要。

### 5.4 FrameworkAdapter

避免定义一个巨型万能接口，继续采用四个窄协议：

```text
TopologyAdapter
StateAdapter
OptimizerAdapter
TrainingAdapter
```

职责分别是：

| 协议 | 职责 |
|---|---|
| TopologyAdapter | inspect、prepare rebuild、rebuild、rebind、validate |
| StateAdapter | catalog、load replacement base、restore、validate state |
| OptimizerAdapter | local state refs、before/after step、rebind |
| TrainingAdapter | progress、quiesce、resume、reset transients、warmup |

### 5.5 训练生命周期 API

框架接入层应收敛到：

```text
initialize_runtime
iteration_boundary
before_optimizer_step
after_optimizer_step
on_distributed_error
commit_iteration
```

这些钩子必须显式存在，因为核心包无法可靠推断 gradient accumulation、optimizer commit 和 iteration safe point。

## 6. 原版组件到目标目录的映射

| 原组件 | 目标位置 | 处理方式 |
|---|---|---|
| `elastic_launcher.py` | `agent/node_agent.py` + `cli/launch.py` | 迁移实现，旧文件保留 shim |
| `elastic_watcher.py` 网络协议 | `control/protocol.py`、`control/service.py` | 抽出通用部分 |
| watcher replacement 生命周期 | `agent/worker_supervisor.py` | 从 watcher 拆出 |
| watcher MoE 决策 | `policy/moe_hybrid.py` | 只消费标准化事实 |
| `elastic_client.py` watcher client | `runtime/client.py` | 去掉 Megatron import |
| `elastic_client.py` epoch/执行流程 | `runtime/` | 用 RecoveryPlan 驱动 |
| `elastic_client.py` c10d/store | `distributed/` | 公共数据结构和 compat |
| `elastic_client.py` Megatron rebind | `adapters/megatron/` | 保留框架专属实现 |
| `zero2_memory_checkpoint.py` transport | `replication/` | 通用 transport/buffer |
| optimizer tensor 枚举 | Megatron optimizer adapter | 不放入通用 transport |
| MoE integration 和恢复模块 | `adapters/megatron/moe/` | 迁移并保留行为 |
| `training.py` 大量调用 | `adapters/megatron/integration.py` | 收敛为生命周期 facade |
| `parallel_state.py` manifest 数据 | `distributed/` | 抽出通用结构 |
| Megatron group getter/alias | Megatron topology adapter | 继续框架专属 |

## 7. generalize 参考和迁移规则

### 7.1 建议复用的目录

```text
src/moegambit/agent/
src/moegambit/cli/
src/moegambit/control/
src/moegambit/runtime/
src/moegambit/distributed/
src/moegambit/state/
src/moegambit/replication/
src/moegambit/policy/
src/moegambit/observability/

src/moegambit/adapters/base.py
src/moegambit/adapters/registry.py
src/moegambit/adapters/generic_ddp.py
src/moegambit/adapters/megatron/

examples/generic_ddp/
tests/unit/
docs/
```

### 7.2 不直接迁移的内容

```text
src/moegambit/adapters/deepspeed/
examples/deepspeed/
任何只验证 generalize 受限 DeepSpeed 骨架的测试
```

### 7.3 迁移方法

- 不整体 merge `origin/generalize`；
- 不批量 cherry-pick 所有提交；
- 优先按职责目录迁移；
- 混合修改多个框架的提交必须重新拆分；
- 每个迁移提交必须单独通过当时版本的测试；
- 迁移代码与当前 main 冲突时，以当前 main 已验证行为为基线，不能仅为匹配 generalize 删除后续修复。

## 8. 分阶段开发计划

### Phase A：分支与基线冻结

状态：已完成。

- 从 `origin/main@1c63067` 创建 `runtime`；
- 记录 generalize 参考提交 `42adebd`；
- 不修改 DeepSpeed 所有权目录；
- 建立本设计文档。

退出条件：开发基线和范围获得项目组确认。

### Phase B：包骨架与公共模型

- 添加 `pyproject.toml` 和 `src/moegambit`；
- 建立 lazy public API；
- 添加 config、capabilities、error taxonomy；
- 迁移 TopologySpec、StateCatalog、StateVersion、RecoveryPlan；
- 迁移 adapter Protocol 和 registry；
- 添加 package/lazy import 单元测试。

退出条件：无 torch 环境可以安装并 import 核心包。

### Phase C：控制面、agent 与恢复执行器

状态：已完成（框架无关源码、无 torch 单测和本机回环控制链路）。

- 迁移 launcher、node agent、worker supervisor；
- 建立 versioned protocol 和 HMAC；
- 建立 watcher service、coordinator、state store；
- 建立 RecoveryLifecycle 和 RecoveryExecutor；
- 建立 checkpoint fallback；
- 建立结构化事件、metrics 和 RecoveryRecord。

退出条件：fake adapter 可以端到端执行同一冻结计划，旧 epoch 和 split-brain 被拒绝。

实现证据和能力边界见 `docs/development/PHASE_C_IMPLEMENTATION.md`。Phase C 的通用计划服务当前只接受单 rank fail-stop 和 replicated peer state；sharded/unique state、Megatron 行为迁移及目标集群验证分别属于后续阶段，不能据此标记为生产 Supported。

### Phase D：Megatron 适配与行为保持

- 迁移原 elastic client 到 `adapters/megatron`；
- 迁移 MoE integration 和恢复模块；
- 原路径替换为兼容 alias；
- 实现 Megatron topology/state/optimizer/training adapter；
- 将训练循环调用收敛到稳定 facade；
- 更新 launcher、watcher 和 replacement 脚本的 PYTHONPATH/CLI；
- 保持原 hybrid、ZeRO-2、rollback、pipeline repair 语义。

退出条件：非 adapter 核心代码不 import Megatron；原 Megatron 测试与启动路径保持可用。

### Phase E：Generic DDP conformance

- 提供 `RebindableModel`；
- 重建 DDP wrapper/reducer；
- 实现参数与 optimizer peer restore；
- 支持 AdamW lazy slot materialization；
- 提供 CPU/Gloo 与 GPU/NCCL 故障程序。

退出条件：在无 Megatron 环境完成至少一次真实 logical rank replacement。

### Phase F：发布工程和文档

- 完成 CLI `launch|watcher|doctor`；
- 构建 wheel 并验证 LICENSE/LEGAL；
- 整理设计、实施状态、限制和实机测试说明；
- 建立兼容矩阵和 support level；
- 清理内部路径、token 和不可发布配置。

退出条件：全新环境可复现安装与示例，不依赖开发者机器隐式文件。

### Phase G：未来系统集成门槛

本阶段不在本轮实施，只登记后续必须解决的问题：

- 统一 `src/moegambit` 与 `deepspeed_adapter/moegambit` 的包身份；
- DeepSpeed 实现依赖公共契约，而不是携带第二份核心；
- 明确 DeepSpeed 节点级 epoch relaunch 与进程内 rebuild 的 capability 区别；
- 在 system 分支重新运行双方全部单测和实机矩阵。

## 9. 测试与验收

### 9.1 每个提交的最低要求

- `git diff --check` 通过；
- Python 文件 compileall 通过；
- 当前提交可运行的非 torch 测试全部通过；
- 不引入 DeepSpeed 目录修改；
- 不把日志、缓存、模型、checkpoint 和 `.DS_Store` 纳入提交。

### 9.2 核心与契约测试

- protocol version、HMAC、message size；
- epoch、幂等、旧消息拒绝；
- RecoveryPlan digest 包含 state source；
- group manifest 和创建顺序；
- state identity、placement、version；
- RecoveryExecutor 阶段顺序和失败回退；
- adapter capability 和 lazy discovery；
- control/agent 无 torch import。

### 9.3 Megatron 验收

- 禁用 MoEGambit 时训练语义不变；
- 单 rank fail-stop 与备用 rank 接管；
- PP/EP/DP group rebuild 和 rebind；
- dense/non-expert peer restore；
- expert/checkpoint restore；
- ZeRO-2 memory replica；
- 第一个恢复后 collective 和 optimizer step；
- 连续两次故障和恢复中再故障；
- 参数、optimizer、loss、scheduler、RNG 和数据位置对齐。

### 9.4 Generic DDP 验收

- 两进程 Gloo fail-stop/replacement；
- DDP reducer 真正替换；
- 参数和 AdamW state 完整恢复；
- 恢复后继续多个 step；
- 所有 rank 最终参数摘要一致；
- NCCL 版本在目标 GPU 环境重复验证。

## 10. 支持等级

| 等级 | 判定条件 |
|---|---|
| Supported | 目标版本和集群完成故障矩阵，允许生产试用 |
| Experimental | 源码和正确性测试完成，但目标规模/版本覆盖不足 |
| Detection-only | 只分类故障并触发外部 checkpoint relaunch |
| Unsupported | 启动前拒绝，不允许静默进入错误路径 |

源码存在、adapter 可 import 或 mock 单测通过，均不能单独升级为 Supported。

## 11. 提交和评审策略

建议按以下逻辑提交，不把全部重构压成一个大提交：

1. `docs: define non-DeepSpeed runtime development plan`
2. `build: scaffold installable moegambit package`
3. `feat(core): add topology, state and recovery contracts`
4. `refactor(agent): extract process launcher and supervisor`
5. `feat(control): add versioned recovery coordination`
6. `feat(runtime): execute frozen recovery plans`
7. `feat(infra): add distributed, replication and observability utilities`
8. `refactor(megatron): relocate MoE recovery modules`
9. `refactor(megatron): relocate elastic recovery client`
10. `feat(megatron): add adapter and training facade`
11. `feat(ddp): add replacement conformance adapter`
12. `feat(examples): add non-DeepSpeed recovery examples`
13. `docs: record implementation and validation status`

每个提交应能独立检出、编译和运行当时适用的测试。

## 12. 风险和处理原则

### 12.1 两份 moegambit 包

风险：`src/moegambit` 与 `deepspeed_adapter/moegambit` 同时位于 PYTHONPATH 时，导入结果由路径顺序决定。

本轮措施：不修改或删除同事目录；默认 core 测试只暴露 `src`，DeepSpeed 测试使用单独进程和显式 profile；wheel 只从 `src` 收集包；核心发现竞争包根目录时 fail closed。具体约定见 `docs/development/PACKAGE_ISOLATION.md`。

限制：这些措施消除当前开发和发布过程中的随机路径选择，但第二份同名包仍然存在。最终合并 main 前，必须迁移并删除 `deepspeed_adapter/moegambit`，或将其改为唯一包名；在此之前不得同时把两个根目录加入 PYTHONPATH。

### 12.2 当前 main 与 generalize 行为差异

风险：generalize 分叉后 main 已继续修复运行时和实机路径。

措施：迁移时逐文件比较，以当前 main 行为为基线，将 generalize 的结构应用到最新实现，而不是用旧文件整体覆盖。

### 12.3 PyTorch c10d 私有接口

风险：ProcessGroup registry、group count 和重初始化行为跨版本不稳定。

措施：集中到 compat 层、版本矩阵、doctor 预检、manifest、外部同步和 fail closed。

### 12.4 大型兼容模块仍存在

风险：Megatron `elastic_client` 和 MoE integration 迁入 adapter 后仍然很大。

措施：本轮先完成依赖边界和行为保持；后续再在 adapter 内渐进拆分，不在一次迁移中同时改架构和恢复语义。

## 13. 完成定义

本轮非 DeepSpeed 开发完成，至少需要满足：

- `runtime` 分支包含可安装的唯一非 DeepSpeed 核心包；
- 核心目录不 import Megatron；
- watcher/control/agent 可在无 torch 环境运行；
- RecoveryRuntime 能执行冻结计划，而不是固定返回 detection-only；
- Megatron 原热备路径通过 adapter 和公共 facade 接入；
- Generic DDP 完成真实 replacement 验证；
- wheel、CLI、lazy import 和非 torch 测试通过；
- Megatron 目标集群故障矩阵有可复核结果；
- 所有未验证能力准确标记为 Experimental、Detection-only 或 Unsupported；
- 未修改同事负责的 DeepSpeed 目录；
- 最终集成前已登记并解决两份 `moegambit` 包冲突。

## 14. 下一步

设计文档评审通过后，按 Phase B 开始迁移包骨架和公共契约。第一批代码提交不触碰 Megatron 恢复语义，也不触碰 DeepSpeed 目录，先建立可安装、可测试、依赖方向明确的核心基础。
