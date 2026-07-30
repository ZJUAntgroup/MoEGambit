# MoEGambit 统一恢复架构

## 1. 目标

Megatron 和 DeepSpeed 共用一份可安装的 `moegambit` 包。框架原生仓库只保留
必要的生命周期钩子和兼容转发，不再各自维护 watcher、wire protocol、恢复策略
或 optimizer replica transport。

本次整合的基线为本地可用的 `origin/main@50b2507`。远端 SSH 在开发时不可达，
因此没有声称包含该提交之后的远端变更。

## 2. 目录与依赖方向

```text
src/moegambit/
├── core/                  # 纯 Python 决策模型、状态机、事件
├── runtime/               # 恢复编排、watcher client、hot spare、c10d 协议
├── interfaces/            # EngineAdapter 启动协议
├── adapters/
│   ├── megatron/          # Megatron 对象、训练循环和 MoE 语义
│   ├── deepspeed/         # DeepSpeed engine/group/ZeRO/checkpoint 语义
│   └── generic_ddp.py
├── distributed/           # 通用 topology model 和 c10d compatibility
├── replication/           # 通用 optimizer memory replication
├── control/               # 认证控制面和 frozen recovery plan
└── cli/                   # launch、watch、watcher、doctor
```

静态依赖只能向内：

```text
framework hook -> framework adapter -> interfaces/runtime/core
```

`core` 不允许导入 torch、Megatron 或 DeepSpeed。`runtime` 可以在函数内部延迟
导入 torch distributed compatibility，但不得持有框架 engine。框架对象只能进入
各自 adapter。

## 3. 两层 adapter contract

`interfaces.EngineAdapter` 负责识别训练命令、投影 feature 环境变量、注入框架
源码或插件路径和选择 watcher；它不接触模型、optimizer 或 process group。

`adapters.base.FrameworkAdapter` 组合 `TopologyAdapter`、`StateAdapter`、
`OptimizerAdapter` 和 `TrainingAdapter`。统一 runtime 只通过这些协议读取
topology/state 并执行 frozen `RecoveryPlan`。成熟的旧恢复序列可以暂时通过
`RecoveryDriver` 兼容桥接，但策略和状态源仍由通用层定义。

## 4. 公共恢复流程

```text
failure
  -> classify
  -> freeze recovery epoch and plan
  -> quiesce and prove collectives drained
  -> rebuild topology in deterministic order
  -> rebind framework objects
  -> restore version-compatible state
  -> warmup and validate
  -> provisional resume
  -> commit first complete iteration
```

任何 topology digest、state version、source locator 或 capability 不一致都
fail closed，并进入 checkpoint relaunch 或 abort，不允许继续不确定训练。

## 5. 兼容边界

- `megatron.training.elastic_client` 和 Megatron MoE 旧模块保留 forwarding shim；
- DeepSpeed 原生目录从 `moegambit.adapters.deepspeed` 接入 engine hook；
- 根目录 `elastic_launcher.py` 和 `elastic_watcher.py` 仅按 `--adapter` 分发；
- Megatron 旧实现位于 `adapters/megatron/compat_launcher.py` 和
  `compat_watcher.py`，DeepSpeed 同名入口转发到通用 `runtime.hot_spare`；
- `moegambit.runtime.hot_spare`、`distributed`、`zero2_replica` 等已验证模块名
  保留，但实现只存在于根包；
- `moegambit.runtime.config` 仅转发根 `moegambit.config`，防止再次出现两份配置；
- 仓库运行时 `PYTHONPATH` 顺序固定为 `src`、框架源码。

## 6. CLI

- `moegambit-launch` 带 engine 选项时走 `EngineAdapter`，带 node-agent 参数时
  走通用 worker supervisor；
- `moegambit-watch` 选择框架 watcher；
- `moegambit-watcher` 运行带认证和持久 epoch 的通用控制服务；
- `moegambit-doctor` 检查配置和能力。

## 7. 验证要求

1. 只存在一个可导入的 `moegambit/__init__.py`；
2. 不安装 torch 时可以导入 core、interfaces、control 和 CLI parser；
3. Megatron shim 指向 `src/moegambit/adapters/megatron`；
4. DeepSpeed adapter 与通用 runtime 能在同一解释器导入；
5. DeepSpeed 多机脚本 dry-run 生成原有 rank-in-process hybrid 命令；
6. 通用、Megatron boundary 和 DeepSpeed adapter 测试同时通过。
