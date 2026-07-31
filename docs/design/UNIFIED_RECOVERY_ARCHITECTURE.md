# MoEGambit 统一恢复架构

## 1. 目标

Megatron 和 DeepSpeed 共用一份可安装的 `moegambit` 包。框架原生仓库只保留
必要的生命周期钩子，不再维护 forwarding shim、watcher、wire protocol、恢复
策略或 optimizer replica transport。



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

### 4.1 训练 step 事务

两套 adapter 共用 `core.step_transaction.StepTransaction`。状态机分别记录
forward、backward、optimizer-before、optimizer-during、optimizer-after、
step-committed、checkpoint-before、checkpoint-commit 和
checkpoint-committed。恢复决策遵守以下不变量：

1. optimizer-before 及以前只清理瞬态梯度、回退数据并 replay；
2. optimizer-during 必须先从相同 committed step 的完整 model/optimizer
   副本恢复，禁止只修改 iteration；
3. optimizer-after 只有在对应 optimizer peer replica 得到 ACK 后才能转为
   step-committed；
4. 无法证明完整恢复时进入 checkpoint relaunch 或 abort；
5. checkpoint 只有在 shard/manifest 校验完成并原子写入完成标记后才可用于
   restart，未完成目录永远不参与最新 checkpoint 选择。

## 5. 兼容边界

- Megatron 原生文件只允许
  `from moegambit.adapters.megatron.hooks import megatron_hooks`，不允许导入
  adapter 实现函数，也不在 `megatron/` 下保留 MoEGambit 模块或 forwarding
  shim；
- Megatron 的恢复参数注册、训练恢复策略、选择性进程组重建状态和首个 MoE
  collective 诊断均由 adapter 持有；原生文件只提交生命周期事件和框架对象；
- DeepSpeed 原生目录只从 `moegambit.adapters.deepspeed.hooks` 接入 engine、
  launcher、process-group 和 checkpoint hook；
- 根目录 `elastic_launcher.py` 和 `elastic_watcher.py` 仅按 `--adapter` 分发；
- Megatron 旧实现位于 `adapters/megatron/compat_launcher.py` 和
  `compat_watcher.py`，DeepSpeed 同名入口转发到通用 `runtime.hot_spare`；
- `moegambit.runtime.hot_spare`、`distributed`、`zero2_replica` 等已验证模块名
  保留；optimizer 内存副本的唯一实现位于 `moegambit.replication`，
  `runtime.zero2_replica` 只提供兼容导出，Megatron 和 DeepSpeed 不再维护两套
  ring transport、snapshot 和 restore 实现；
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
3. Megatron 原生树只导入公开 `megatron_hooks` 单例，`TransformerConfig`
   不包含 `moe_moegambit_*` 恢复策略字段；
4. DeepSpeed adapter 与通用 runtime 能在同一解释器导入；
5. DeepSpeed 多机脚本 dry-run 生成原有 rank-in-process hybrid 命令；
6. 通用、Megatron boundary 和 DeepSpeed adapter 测试同时通过。
