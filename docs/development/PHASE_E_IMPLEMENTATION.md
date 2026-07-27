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
恢复。只有提供 `RebindableModel`，或者同时提供 `module_rebuilder/module_setter`，才声明
`full_group_rebuild=True`。

### 参数与 optimizer peer restore

参数和 optimizer 状态使用稳定的 parameter name/state key 生成 identity。恢复计划为每个
identity 指定 `rank://N` peer 来源，恢复阶段按 identity 排序执行 broadcast。计划摘要包含
来源 rank、版本和 shape/dtype metadata，因此不同 rank 无法静默采用不同恢复来源。

AdamW 在第一次 step 时才创建 `exp_avg`、`exp_avg_sq` 和 step 等状态。replacement worker
在 peer restore 前根据冻结计划中的 parameter name、state key、shape、dtype 和 scalar type
物化空槽，不执行伪 optimizer step，然后再接收 survivor 数据。

### 生命周期与安全边界

训练示例显式调用 iteration boundary、optimizer step 前后、distributed error 和 iteration
commit。适配器不会对任意 RuntimeError 猜测恢复；只有 typed distributed error 或调用者
提供的明确分类器才能进入恢复。

## 实机验证

验证环境：macOS arm64、Python 3.9、PyTorch 2.8.0、CPU/Gloo，两进程，不导入 Megatron。

执行命令：

```bash
python examples/generic_ddp/fault_replacement.py \
  --backend gloo --steps 6 --fail-step 2
```

验证过程：

1. logical rank 0/1 完成两个 optimizer step；
2. rank 1 使用 fail-stop 方式退出，未执行分布式清理；
3. controller 启动新的进程承接 logical rank 1；
4. survivor 与 replacement 在新 rendezvous 重建 WORLD 和 DDP reducer；
5. replacement 从 rank 0 恢复模型参数和 AdamW 状态；
6. 两个 rank 继续训练到 step 6；
7. 最终参数摘要一致：`7e04523ab57766ea...`。

最终程序输出：

```text
PASS: logical rank 1 was replaced, peer state restored, and DDP continued;
digest=7e04523ab57766ea
```

普通的 disabled-runtime 训练循环也在同一 PyTorch 环境完成 20 个 step，证明生命周期钩子
关闭时保持惰性，不改变普通训练行为。

## 能力与未验证项

当前 Generic DDP 支持：单 DP 轴、完整 group rebuild、replacement logical rank、参数和
普通 optimizer peer restore、AdamW lazy slot materialization。它不声明 optimizer memory
replication、MoE state classification、选择性 group rebuild 或 ZeRO 支持。

`examples/generic_ddp/fault_replacement.py --backend nccl` 已提供 GPU/NCCL 路径；当前机器
没有两张可用 GPU，因此该路径尚未实机执行。在 NCCL 实测完成前，Generic DDP adapter
仍保持 `experimental`，不能据 CPU/Gloo 结果提升为生产 Supported。
