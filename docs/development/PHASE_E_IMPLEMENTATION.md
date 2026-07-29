# Phase E：Generic DDP conformance

## 实现结果

Phase E 新增 `moegambit.adapters.generic_ddp`，作为不依赖 Megatron 的完整参考适配器。
它把普通 PyTorch DDP 投影到 Phase B 定义的四个窄协议，并通过 Phase C 的
`RecoveryExecutor` 执行冻结恢复计划。

## 关键实现

### RebindableModel 与 DDP 重建

DDP 的 reducer 绑定在创建 wrapper 时使用的 ProcessGroup 上，不能通过修改元数据安全地
换组。`RebindableModel` 为训练循环提供稳定句柄；恢复时适配器销毁旧通信组、按固定顺序
重建 WORLD 和 data-parallel group，然后用底层 module 构造新的 DDP wrapper/reducer，
最后原子替换稳定句柄中的当前对象。

如果调用者交给适配器的是不可替换的 live DDP wrapper，适配器会在销毁通信组之前拒绝
恢复。live DDP 可以由 `RebindableModel` 持有并使用安全的默认 rebuilder；replacement
进程从 bare module 启动时必须提供 `module_rebuilder`。非 `RebindableModel` 所有权模型则
必须同时提供 `module_rebuilder/module_setter`，否则不会声明完整重建能力。

默认 rebuilder 强制 `init_sync=False`，避免 logical rank 0 正好是 replacement 时，其旧
基线在显式 peer restore 前覆盖 survivor。自定义 communication hook、device mesh 或延迟
all-reduce 配置无法自动复原时，会在 quiesce 和通信组销毁前失败关闭。

### 参数与 optimizer peer restore

参数、model buffers、optimizer slots 和 param-group options 使用稳定的名称生成 identity。
恢复计划为每个 identity 指定 `rank://N` peer 来源，恢复阶段按 identity 排序执行
broadcast。计划摘要包含来源 rank、版本、shape/dtype/device metadata，因此不同 rank
无法静默采用不同恢复来源，shape 或 dtype 不一致也会在数据 collective 前被拒绝。
catalog agreement 和 scalar 传输使用普通 tensor collectives，不依赖 PyTorch object
collective 或 NumPy；序列化 optimizer option 设有 1 MiB 上限。

AdamW 在第一次 step 时才创建 `exp_avg`、`exp_avg_sq` 和 step 等状态。replacement worker
在 peer restore 前根据冻结计划中的 parameter name、state key、shape、dtype 和 scalar type
物化空槽，不执行伪 optimizer step，然后再接收 survivor 数据。学习率、weight decay、
betas 等 param-group options 也会恢复，避免 replacement 使用初始化时的旧超参数。

DDP buffers 在 forward 后可能因各 rank 本地 batch 再次分叉。适配器在 committed optimizer
step 后同步 rank 0 的权威 buffer 并保存 committed 快照；下一次 forward 中发生故障时，
peer restore 使用该快照，而不是可能已部分更新的当前 buffer。

### 生命周期与安全边界

训练示例显式调用 iteration boundary、optimizer step 前后、distributed error 和 iteration
commit。适配器不会对任意 RuntimeError 猜测恢复；只有 typed distributed error 或调用者
提供的明确分类器才能进入恢复。拓扑 preflight 在 quiesce 之前完成；如果 optimizer step
已经开始但尚未提交，peer fast path 会拒绝执行，以免传播部分更新状态。

## 实机验证

验证环境：macOS arm64、Python 3.9、PyTorch 2.8.0、CPU/Gloo、无 NumPy，
两进程，不导入 Megatron。

执行命令：

```bash
python examples/generic_ddp/fault_replacement.py \
  --backend gloo --steps 6 --fail-step 2 --fail-rank 1

python examples/generic_ddp/fault_replacement.py \
  --backend gloo --steps 6 --fail-step 2 --fail-rank 0
```

验证过程：

1. logical rank 0/1 完成两个 optimizer step；
2. rank 1 和 rank 0 分别使用 fail-stop 方式退出，未执行分布式清理；
3. controller 启动新的进程承接相同 logical rank；
4. survivor 与 replacement 在新 rendezvous 重建 WORLD 和 DDP reducer；
5. replacement 从 survivor 恢复参数、committed buffers、AdamW slots 和
   optimizer param-group options；
6. 两个 rank 继续训练到 step 6；
7. 最终 model/optimizer/完整状态摘要一致：`9a78172febbe8ea3...`。

最终程序输出：

```text
PASS: logical rank 0 was replaced, model/buffer/optimizer state restored,
and rebuilt DDP continued; digest=9a78172febbe8ea3

PASS: logical rank 1 was replaced, model/buffer/optimizer state restored,
and rebuilt DDP continued; digest=9a78172febbe8ea3
```

普通的 disabled-runtime 训练循环也在同一 PyTorch 环境完成 20 个 step，证明生命周期钩子
关闭时保持惰性，不改变普通训练行为。

## 能力与未验证项

当前 Generic DDP 支持：单 DP 轴、完整 group rebuild、replacement logical rank、参数、
committed buffers、普通 optimizer peer restore、param-group options 和 AdamW lazy slot
materialization。它不声明 optimizer memory replication、MoE state classification、
选择性 group rebuild 或 ZeRO 支持；scheduler 内部计数、RNG 和 dataloader cursor 仍需由
调用者通过 checkpoint/replacement loader 或训练接入层管理。

`examples/generic_ddp/fault_replacement.py --backend nccl` 已提供 GPU/NCCL 路径；当前机器
没有两张可用 GPU，因此该路径尚未实机执行。在 NCCL 实测完成前，Generic DDP adapter
仍保持 `experimental`，不能据 CPU/Gloo 结果提升为生产 Supported。
