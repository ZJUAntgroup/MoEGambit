# Phase D：Megatron 适配与行为保持

## 实现结果

Phase D 将当前分支上最新版 Megatron 热备恢复实现迁入
`src/moegambit/adapters/megatron/`，没有回退到 `generalize` 分支中较早的实现。
迁移覆盖 elastic client、MoE integration，以及 hybrid recovery、ZeRO-2 内存副本、
iteration rollback、pipeline repair、replacement/reintegration 等恢复模块。

原来的 `megatron.training.elastic_client` 和
`megatron.core.transformer.moe.*` 路径保留为模块别名。新旧路径加载的是同一个模块对象，
因此不会形成两套全局状态，也保留现有测试、patch point 和启动命令的兼容性。

## 新的职责边界

- `moegambit` 通用核心负责恢复生命周期、计划、epoch、执行和失败回退规则；
- `moegambit.adapters.megatron` 负责读取 Megatron 拓扑、枚举参数与优化器状态、
  重建和 rebind 通信组，并桥接已有的热备恢复状态机；
- `megatron.training.training` 只在正确时机调用稳定 facade：初始化、iteration safe point、
  optimizer step 前后、分布式异常和 iteration commit。

适配器由 `TopologyAdapter`、`StateAdapter`、`OptimizerAdapter`、`TrainingAdapter`
四个窄组件组成。成熟的原有恢复序列暂由 `MegatronRecoveryDriver` 桥接，避免在这次目录
重构中重写破坏性通信流程。

## 行为保持

适配器明确声明以下能力：静态 world 热备替换、选择性和完整通信组重建、参数 peer restore、
ZeRO-2 optimizer memory replication、MoE 状态分类、两阶段 optimizer restore，以及
DP/TP/PP/EP/ETP/CP 拓扑轴。

源码启动脚本的 `PYTHONPATH` 已同时包含 `src` 与 `Megatron-LM`，replacement worker
沿用同一环境，因此能加载新包和旧兼容路径。DeepSpeed 与 `deepspeed_adapter` 未修改。

## 已验证内容

- 非 Megatron adapter 的核心源码不导入 Megatron；
- 导入适配器定义不会提前导入 torch 或 Megatron；
- 迁移后的 elastic/MoE 实现仍包含关键恢复入口；
- 每个迁移模块都有旧路径 alias；
- 训练循环已接入全部稳定生命周期钩子；
- 旧控制面源码测试与 Phase B/C 单元测试继续通过；
- wheel 能包含完整的 Megatron adapter 包。

## 尚未由本机证明的内容

当前本机没有 PyTorch/CUDA/NCCL，因此测试通过不等于已经证明多机 GPU 热替换成功。
发布为 Supported 前仍必须执行：单 rank fail-stop、replacement worker 加入、ZeRO-2
副本恢复、EP unique expert checkpoint 恢复、PP rollback/replay，以及恢复后持续训练和
loss 连续性检查。未完成这些实机用例前，Megatron adapter 的支持等级保持 experimental。
