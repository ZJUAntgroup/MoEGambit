<div align="center">

<div style="margin: 20px 0;">
  <a href="docs/assets/moegambit-chess.png">
    <img src="docs/assets/moegambit-chess.png"
         width="420"
         alt="MoEGambit 国际象棋棋盘">
  </a>
</div>

# MoEGambit

### 面向分布式混合专家训练的选择性状态修复

为 Megatron-LM 与 DeepSpeed 提供框架无关的热替换、版本感知状态恢复和事务式故障恢复。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Megatron Core](https://img.shields.io/badge/Megatron_Core-0.15.3-76B900?style=flat-square)](Megatron-LM)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.19.3-00539C?style=flat-square)](DeepSpeed)

[![English](https://img.shields.io/badge/🇺🇸_English-1a1a2e?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/🇨🇳_简体中文-1a1a2e?style=for-the-badge)](README-zh.md)

[项目概览](#项目概览) · [快速开始](#快速开始) ·
[系统架构](#系统架构) · [运行示例](#多机运行示例) ·
[恢复契约](#恢复契约) · [实验结果](#实验结果)

<a href="docs/assets/moegambit-runtime-architecture.png">
  <img src="docs/assets/moegambit-runtime-architecture.png"
       alt="MoEGambit 恢复架构"
       width="100%">
</a>

<sub>恢复架构。点击查看原始分辨率大图。</sub>

</div>

## 项目概览

MoEGambit 是面向分布式MoE与dense训练的选择性恢复runtime。发生fail-stop rank故障后，
MoEGambit 能够保持分布式训练作业存活，激活常驻替补 worker，按确定性顺序
重建通信组，并从当前最安全的状态源恢复训练状态。

系统的核心是混合恢复：

- 从健康 peer 拉取处于当前已提交版本的非专家复制状态；
- 当不存在在线专家副本时，从 checkpoint 一起恢复专家权重与对应优化器状态；
- 在 adapter 与版本校验支持时，使用在线专家副本和已确认的主机内存优化器副本；
- 来源与版本检查决定执行 peer 修复或完整 checkpoint relaunch。

MoEGambit 将恢复策略和编排逻辑与训练框架解耦。Megatron-LM 与 DeepSpeed
内部只保留生命周期 hook；对应 adapter 将框架对象转换到同一套恢复契约。

### EDP distribution · 状态分布示意图

![EDP=1 与 EDP=2 的状态布局：唯一专家分片与重复专家组。](docs/assets/benchmarks/edp_distribution.png)

[查看原始矢量 PDF](docs/assets/benchmarks/edp_distribution.pdf)

EDP=1 时，专家分片没有在线副本，但非专家状态仍有复制；EDP=2 时，
另一个专家数据并行组可以提供当前专家状态。主要实验的 64-GPU 布局使用 PP=8、EP=8、ETP=1，因此 EDP=1。

## 核心特性

- **一套恢复 runtime：** `src/moegambit/` 中统一实现 controller、watcher
  协议、策略、进程组编排、可观测性和 CLI。
- **两套框架 adapter：** Megatron-LM 与 DeepSpeed 使用相同的 adapter
  边界以及 launcher/watcher 入口。
- **常驻热备：** survivor 保留原有 Python 与 CUDA 进程，由 spare 接管
  故障进程的 logical rank。
- **版本感知恢复：** 仅在 model、optimizer、scheduler、RNG、data cursor
  和 checkpoint 记录版本兼容时执行恢复。
- **阶段感知事务：** forward、backward、optimizer 和 checkpoint 发布阶段
  的故障均有明确的 replay 或 fallback 语义。
- **质量风险准入（可选）：** 公共协调器使用 `R <= 1` 质量风险规则，支持审计与
  执行模式；执行需要有效的整场训练风险证据。引擎兼容路径仍使用原有策略，详见
  [审计工具说明](docs/ARTIFACT_AUDIT.md)。
- **提交完整性：** 部分 optimizer 更新需恢复 model 和 optimizer 状态后再 replay。

## 快速开始

### 安装

要求 Python 3.10 或更高版本。安装 MoEGambit 之前，请先安装与集群匹配的
CUDA 版 PyTorch。

```bash
# 进入本地检出或解压后的源码目录。
cd /path/to/MoEGambit

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 首先安装与集群 CUDA 版本匹配的 PyTorch wheel。
python -m pip install torch --index-url <PYTORCH_CUDA_WHEEL_INDEX>

# 安装 MoEGambit 及开发检查依赖。
python -m pip install -e '.[dev]'
```

根据训练任务安装所需框架。下面列出的是仓库内 Megatron 与 DeepSpeed
Qwen3-MoE 示例的必要依赖：

```bash
# Megatron-LM
python -m pip install \
  'numpy<2.0.0' 'packaging>=24.2' \
  pybind11 Cython sentencepiece tiktoken
python -m pip install 'transformer-engine[pytorch]'
python -m pip install -e ./Megatron-LM

# DeepSpeed
python -m pip install -r deepspeed_requirements.txt
python -m pip install --upgrade 'transformers>=5.0.0,<6'
python -m pip install -e ./DeepSpeed
```

`deepspeed_requirements.txt` 会安装 `accelerate`、`einops`、`hjson`、
`msgpack`、`ninja`、`numpy`、`packaging`、`psutil`、`py-cpuinfo`、
`pydantic`、`tqdm` 和 `transformers`。上面的显式升级不能省略：
`deepspeed_qwen3_moe_pretrain.py` 与 DeepSpeed 启动前检查均要求
`transformers>=5.0.0`。如果环境中仍是 `transformers==4.45.0`，
分布式作业会在启动前被拒绝。

仓库中的脚本会显式优先加载项目内置的框架源码：

```bash
export PYTHONPATH="$PWD/src:$PWD/DeepSpeed:$PWD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

### 验证安装

```bash
python - <<'PY'
import torch
import moegambit
import moegambit.adapters.deepspeed
from moegambit.runtime.discovery import discover_adapters

assert torch.cuda.is_available()
assert torch.distributed.is_available()
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("moegambit:", moegambit.__file__)
print("adapters:", sorted(discover_adapters()))
PY

moegambit-doctor
python -m pytest tests -q
```

安装后提供以下命令：

| 命令 | 用途 |
| --- | --- |
| `moegambit-launch` | 构建或启动由 adapter 选择的训练命令 |
| `moegambit-watch` | 通用 watcher client 入口 |
| `moegambit-watcher` | 带认证的框架无关 coordinator |
| `moegambit-doctor` | 检查本地 runtime 环境 |
| `moegambit-elastic-launcher` | Megatron/DeepSpeed 兼容 launcher |
| `moegambit-elastic-watcher` | Megatron/DeepSpeed 兼容 watcher |

## 系统架构

```text
framework hook
      │
      ▼
EngineAdapter（Megatron / DeepSpeed / Generic DDP）
      │
      ├── 框架生命周期与状态转换
      ▼
moegambit runtime + policy + control plane
      │
      ├── 故障分类与恢复计划
      ├── spare 分配与拓扑 manifest
      ├── 确定性进程组重建
      ├── peer / replica / checkpoint 状态源选择
      └── commit、rollback、fallback 与可观测性
```

依赖方向必须保持为：

```text
framework hook -> framework adapter -> moegambit interfaces/runtime/core
```

框架无关恢复逻辑集中于 `src/moegambit/`，框架 adapter 负责翻译引擎状态和生命周期事件。
Megatron 使用统一的公开集成入口：

```python
from moegambit.adapters.megatron.hooks import megatron_hooks
```

Megatron 通过该 singleton 发布框架事件和对象。恢复配置、故障分类、
rollback/replay 策略、进程组重建状态和诊断逻辑均位于
`src/moegambit/adapters/megatron/`。

完整设计请参阅
[统一恢复架构](docs/design/UNIFIED_RECOVERY_ARCHITECTURE.md)。

### 仓库结构

```text
.
├── src/moegambit/
│   ├── core/                  # 不依赖框架的契约与决策
│   ├── runtime/               # 编排、热备、watcher client 与协议
│   ├── interfaces/            # EngineAdapter 协议
│   ├── adapters/
│   │   ├── megatron/          # Megatron 状态与拓扑集成
│   │   ├── deepspeed/         # DeepSpeed engine 与 ZeRO 集成
│   │   └── generic_ddp/       # 框架无关的参考 adapter
│   ├── control/               # 带认证的控制面与冻结恢复计划
│   ├── distributed/           # 拓扑与 c10d 兼容层
│   └── replication/           # 优化器主机内存复制
├── Megatron-LM/               # Megatron Core 0.15.3 + 最小 hook
├── DeepSpeed/                 # DeepSpeed 0.19.3 + 最小 hook
├── examples/
│   ├── megatron/run_hot_spare.sh
│   ├── deepspeed/run_hot_spare.sh
│   └── generic_ddp/
├── elastic_launcher.py        # adapter-aware 兼容 launcher
├── elastic_watcher.py         # adapter-aware 兼容 watcher
├── test_hotspare_replace.sh
└── test_deepspeed_hotspare_replace.sh
```

### Adapter 支持情况

| 能力 | Megatron-LM | DeepSpeed | Generic DDP |
| --- | :---: | :---: | :---: |
| 通用 launcher/watcher 分发 | ✓ | ✓ | 参考实现 |
| 常驻 rank 热替换 | ✓ | ✓ | 示例实现 |
| 确定性通信组重建 | TP/PP/EP/DP | engine groups | DDP |
| Peer 模型状态恢复 | ✓ | ✓ | adapter 契约 |
| 主机端优化器复制 | distributed optimizer | ZeRO-2 | adapter 契约 |
| Checkpoint fallback | ✓ | ✓ | ✓ |
| 阶段感知事务契约 | ✓ | ✓ | ✓ |

## 恢复契约

对于 fail-stop 事件 `<r, t, c>`——故障 logical rank `r`、当前 iteration
`t` 和最后 checkpoint `c`——恢复协议按以下步骤执行：

1. 冻结恢复计划并增加 recovery epoch。
2. 将故障 rank 标记为 `RECOVERING`，阻止 optimizer commit，并丢弃
   in-flight iteration。
3. 分配物理 spare，同时保持故障进程原有的 logical rank。
4. 使用唯一的标准拓扑 manifest 重建全部进程组。
5. 按组件和已提交版本选择兼容的状态源。
6. 先恢复权重，再在 update barrier 保护下挂载 optimizer state。
7. 仅当需要 replay 时回退 data cursor。
8. 执行一个完整的恢复后 iteration。
9. 提交 epoch，并执行状态转换
   `RECOVERING → REPAIRED → BARRIER → HEALTHY`。

公共协调器新增可选质量风险策略：只有经过校准的整场训练质量越界风险上界满足
`R <= 1` 时，才允许候选混合恢复。默认目标固定为 `eta_final=0.005`、
`eta_peak=0.01`、`alpha_run=0.05`，不以 checkpoint 年龄或专家密度作为准入阈值。
证据绑定模型、遥测版本、策略版本、当前故障、状态来源和完整恢复历史。证据缺失、
过期或超出验证范围时回退 checkpoint；完整 checkpoint 也不可用时中止。

`--policy quality-risk` 默认审计模式。执行模式使用协调器管理的可信风险提供器和
外部审核的校准证据，训练好的目标负载预测器由使用者提供。`MoeHybridPolicy` 和
引擎兼容路径沿用各自原有策略。策略选择、校准要求、状态比对、逐 rank 完成核验与
`moegambit-audit` 使用方法见 [审计工具说明](docs/ARTIFACT_AUDIT.md)。

质量特征可在训练期间通过 `AsyncQualityOffloader` 后台上传到独立 watcher 的
CPU 内存。开启 `--quality-cpu-features` 后，协调器使用故障前已确认的路由、
敏感度、参数及 optimizer 漂移摘要和整场运行历史；训练 rank 进程退出后仍可读取。
CUDA 特征先固定快照再异步复制，未传完、缺失、冲突或版本不匹配时回退 checkpoint。
训练节点失效场景需要将 watcher 部署在其他主机。接入方式、故障边界和 CPU 验证
脚本见 [CPU 特征保留说明](docs/QUALITY_CPU_OFFLOAD.md)。

### 不同故障阶段的语义

最后一次已提交的 optimizer version 与当前正在执行的 iteration 是两个不同概念：

| 故障阶段 | 必须执行的恢复 |
| --- | --- |
| forward、backward、optimizer-before | 丢弃梯度，回退数据，并从最后已提交 step replay |
| optimizer-during | 从上一个已提交 replica 恢复 model 和 optimizer 后 replay |
| optimizer-after，但 replica 未提交 | checkpoint relaunch；禁止只回退 bookkeeping |
| step 已提交 | 从已提交 optimizer version 恢复 |
| checkpoint commit record 发布前或发布中 | 忽略未完成 checkpoint |
| checkpoint commit record 发布后 | 新 checkpoint 可用于 restart |

主机内存 peer replica 确认同一版本后，optimizer step 成为 safe point。
optimizer-during 恢复先还原该已提交版本的 model 和 optimizer；恢复检查失败时
选择 checkpoint relaunch，或在无完整来源时中止。

## 多机运行示例

仓库提供的验证拓扑使用九台同构节点：

- 节点 `0-7`：每台八张 active GPU；
- 节点 `8`：八个常驻 replacement worker；
- 64 个 logical rank；
- step `10` 保存 checkpoint，step `17` 注入故障。

每台节点运行相同脚本，仅修改 `NODE_RANK`。`MASTER_ADDR` 和
`ELASTIC_WATCHER_ADDR` 必须能被所有节点访问。

### Megatron-LM

```bash
# Active 节点：依次使用 NODE_RANK=0、1 ... 7 运行。
NODE_RANK=0 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
bash examples/megatron/run_hot_spare.sh

# Spare 节点。
NODE_RANK=8 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
bash examples/megatron/run_hot_spare.sh
```

已验证的 Megatron 拓扑为 PP=8、EP=8、TP=1。常用覆盖项：

```bash
export FAULT_INJECT_STEP=17
export FAULT_INJECT_NODE=0
export FAULT_INJECT_LOCAL_RANK=1
export TRAIN_ITERS=100
export SAVE_INTERVAL=10
export DATA_PATH=/shared/moegambit/data/train_text_document
export CKPT_DIR=/shared/moegambit/checkpoints/megatron
```

### DeepSpeed

```bash
# Active 节点：依次使用 NODE_RANK=0、1 ... 7 运行。
NODE_RANK=0 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
TEST_MODE=hot_swap \
bash examples/deepspeed/run_hot_spare.sh

# Spare 节点。
NODE_RANK=8 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
TEST_MODE=hot_swap \
bash examples/deepspeed/run_hot_spare.sh
```

DeepSpeed 验证模式：

| `TEST_MODE` | 拓扑 | 行为 |
| --- | --- | --- |
| `hot_swap` | PP=8、EP=8、ZeRO-1 | 在节点 8 替换一个故障 rank |
| `zero2` | PP=1、EP=8、ZeRO-2 | 通过 D2H/TCP 复制 optimizer shard |
| `combined` | PP=1、EP=8、ZeRO-2 | rank 替换与 optimizer 复制同时启用 |
| `all` | 顺序执行 | 先运行 `hot_swap`，再运行 `zero2` |

DeepSpeed 的现有 ZeRO-2 配置使用数据／专家并行；`PipelineEngine` 排除 ZeRO-2/3。

### Dense 模型

Megatron-LM 和 DeepSpeed 均可使用 dense 模型。必须在所有训练节点和 watcher/备用节点设置
`MOEGAMBIT_MODEL_KIND=dense`，且使用不含专家层的模型。训练端会校验这个声明；
误将带专家的模型声明为 dense 会直接报错。下面两个独立 dense 示例使用合成数据；
上面的 Qwen3-MoE 验证脚本仍是 MoE 专用的。

两个示例默认使用两个单 GPU 训练节点和一个单 GPU 备用节点。先按上文安装依赖，
在所有节点设置相同的共享 `RUN_ROOT` 及可互通的地址。先启动备用节点
（`NODE_RANK=2`），再启动训练节点 0、1。下面的地址仅用于文档示例，运行前须
替换为实际可达地址；不需要私有数据集、tokenizer 或模型权重。

```bash
export RUN_ROOT=/shared/runs/dense-example
export MASTER_ADDR=192.0.2.10
export ELASTIC_WATCHER_ADDR=192.0.2.12

# 备用节点：
NODE_RANK=2 bash examples/megatron/run_dense.sh
# 两个训练节点分别执行：
NODE_RANK=0 bash examples/megatron/run_dense.sh
NODE_RANK=1 bash examples/megatron/run_dense.sh
```

DeepSpeed 示例使用另一个 `RUN_ROOT`，每次运行设置不同的 `RUN_ID`：

```bash
export RUN_ROOT=/shared/runs/dense-deepspeed-example
export RUN_ID=dense-demo-001
export MASTER_ADDR=192.0.2.10
export ELASTIC_WATCHER_ADDR=192.0.2.12

# 备用节点：
NODE_RANK=2 bash examples/deepspeed/run_dense.sh
# 两个训练节点分别执行：
NODE_RANK=0 bash examples/deepspeed/run_dense.sh
NODE_RANK=1 bash examples/deepspeed/run_dense.sh
```

脚本写入
`completed.json`，检查恢复后的最终 step、`mode=rank_in_process_peer` 与零专家
陈旧度。两个脚本均可设置 `DRY_RUN=1`，只打印命令而不占用 GPU。默认在第 4 步
checkpoint 之后、第 5 步已提交边界注入故障。这些小型合成任务仅检验恢复路径，
不用于测量恢复耗时或训练质量。

Megatron-LM 使用常规 `pretrain_gpt.py` dense 配置：不传 `--num-experts`，
设置 `EP_SIZE=1`，仍传 `--moe-moegambit-enable` 开启恢复 hook，并确保同一
TP/PP/CP 分片有至少两个 DP 副本。替换 rank 从 checkpoint 构造基础状态，再从当前
step 的 DP 同伴接收全部 dense 参数、持久化 buffer 和优化器状态。没有有效同伴时
拒绝热修复；只有存在有效 checkpoint 时才能执行 checkpoint relaunch。dense 路径
不使用专家 sidecar。

DeepSpeed 使用相同的 `MOEGAMBIT_HOT_SWAP=1`、`MOEGAMBIT_ZERO2=1`
（ZeRO-2 时）和 checkpoint 设置。模型参数、buffer 和 RNG 从同一步 DP 同伴恢复，
ZeRO-1/2 优化器 shard 从主机内存副本恢复。无需启用 packed-expert checkpoint。
环境变量 `MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY` 仍使用
`rank_in_process_hybrid` 作为传输协议名；恢复结果的 `mode` 为
`rank_in_process_peer`，`expert_staleness=0`。PP 与 ZeRO-2 的组合仍不受支持。

dense 路径已在目标集群通过多机 GPU 故障恢复测试；自动化测试覆盖 dense 选择、
状态源契约和错误配置。定量延迟与质量比较使用[实验结果](#实验结果)列出的MoE配置。

### Dry run

只生成命令，不启动分布式 worker：

```bash
DRY_RUN=1 NODE_RANK=0 \
MASTER_ADDR=192.0.2.1 ELASTIC_WATCHER_ADDR=192.0.2.9 \
bash examples/megatron/run_hot_spare.sh

DRY_RUN=1 TEST_MODE=all NODE_RANK=8 \
MASTER_ADDR=192.0.2.1 ELASTIC_WATCHER_ADDR=192.0.2.9 \
bash examples/deepspeed/run_hot_spare.sh
```

多网卡主机需要显式设置：

```bash
export MOEGAMBIT_HOT_SPARE_ADVERTISE_ADDR=<this-node-routable-ip>
export MOEGAMBIT_REPLICA_ADVERTISE_ADDR=<this-node-routable-ip>
```

### Generic DDP

Generic DDP 示例用于说明恢复契约并不依赖任何一个内置框架：

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/generic_ddp/train_loop.py \
  --steps 8 \
  --checkpoint-dir /tmp/moegambit-generic-ddp
```

故障替换流程请参阅
[examples/generic_ddp/README.md](examples/generic_ddp/README.md)。

### MoC-System 对比复现

[`examples/moc_system`](examples/moc_system/README.md) 提供独立的物理 PEC
机制复现，包括五步训练的保存/恢复计时，以及默认 660 次实际更新的端到端
实验。后者包含真实训练、worker 与通信组重建、状态恢复和 replay。
使用已有 Megatron baseline、micro batch size 1 和内置 expert bias，
关闭辅助损失。日志、汇总和图表保存在 `/personal`，成功验证后清理本实验
的临时 checkpoint。完整运行条件和测量边界见该目录说明。

```bash
MASTER_ADDR=127.0.0.1 bash examples/moc_system/run_moc_e2e.sh --plan-only
```

该实验将固定 K 的 PEC 检查点与恢复机制独立移植到 Megatron。实验说明区分物理部分保存、
原作者的 ZeRO-2 实现和旧 checkpoint 状态 overlay，并列出GPU要求及各段计时范围。

## 实验结果

主要实验在64张NVIDIA H20上训练
Qwen3-30B-A3B，使用 Megatron 内置 expert bias，关闭辅助负载均衡损失，
micro batch size 为 1，运行时恢复 checkpoint 间隔为 200 step。
完整 Hybrid 恢复旧专家的**权重与对应 optimizer state**，从健康 peer
获取当前非专家复制状态和 router 状态。

| 结果 | 实测结果 | 对照与测量范围 |
| --- | --- | --- |
| 选择性恢复与两阶段加载 | 延迟降低 **20.6%**；DeepSeek-V2-Lite 配置降低 **55.0%** | 替换 rank 就绪时间，相对 rank 局部 **FullLoad**，不含 replay |
| 保留已提交训练进度 | checkpoint gap 为 100 iteration 时，恢复加速 **35.6×** | 对照整作业 **Restart**，包含恢复及 replay 到相同已提交进度 |
| 64/128 专家完整状态质量 | 500-step 窗口的最大采样峰值退化 **0.393%**，低于 1% 容忍度 | 固定验证集，相对整作业 Restart；覆盖早中晚期、负载和 rank 数 |
| 单次及重复故障后的完成质量 | 最大最终退化 **0.00306%**，采样峰值 **0.00454%** | 四条选定轨迹完成到第 10,000 step，共用种子/训练前缀 |
| 架构比较 | GQA MoE 最大采样峰值 **0.0893%**；DeepSeek 风格 MLA MoE 为 **0.0230%** | 六条 Hybrid 分支，100-step 配对窗口，对照 Restart |
| R2 策略审计 | 准入率 **66%**；危险候选拦截率 **95%**；实际整场轨迹越界 **3/200** | 作者报告的独立、冻结策略审计；单侧 95% 整场风险上界 **3.83%** |
| 下游任务质量 | 任务等权均值 **45.06% / 44.67% / 45.32%** | Restart / MoC PEC / MoEGambit；八项零样本任务，保留各任务 acc/acc_norm 指标 |
| 无故障完整控制路径开销 | 20 次重复的平均 step 耗时变化 **−0.003%**，观测范围 **−0.09%～+0.07%** | 包含设备完成确认和 all-rank fence；范围不是置信区间 |
| Web模型更新流程 | **1.13–1.53×**加速，分配GPU小时减少 **11.3%–34.8%** | 六组64卡冷启动Hybrid／Restart配对；Hybrid恢复前启用R2准入、检查点模拟状态丢失 |

### 实验图表

以下图表展示训练稳定性、恢复质量、模型配置与状态恢复性能。恢复架构位于本页顶部；
每项结果均提供矢量图与测量说明。

#### 十次故障下的训练 loss

![Restart、MoC PEC 复现与完整状态 MoEGambit 的训练 loss。](docs/assets/benchmarks/train_loss.png)

[查看原始矢量 PDF](docs/assets/benchmarks/train_loss.pdf)

Restart / MoC PEC / MoEGambit 最后 200 step 的平均训练 loss 为 **2.7919 / 2.8254 / 2.7910**。训练 loss 用于稳定性诊断；质量边界使用固定验证集 loss。

#### 训练阶段与故障 rank 数的质量结果

![相对整作业 Restart 的十二组 500-step 续训和三组完成到第 10000 step 的结果。](docs/assets/benchmarks/quality_checkpoint_study.png)

[查看原始矢量 PDF](docs/assets/benchmarks/quality_checkpoint_study.pdf)

左图展示十二组恢复窗口；右图展示三组筛选出的长程续训结果。对照为整作业 Restart，最终/峰值损失容忍度为 **0.5% / 1%**。

#### 64/128 专家的完整 Hybrid 质量

![恢复旧专家权重及 optimizer state 的完整 Hybrid：训练阶段、负载和 rank 数比较。](docs/assets/benchmarks/quality_full_state_500.png)

[查看原始矢量 PDF](docs/assets/benchmarks/quality_full_state_500.pdf)

受影响专家同时恢复旧权重与 optimizer state。500-step 窗口最大采样峰值为 **0.393%**，低于 1% 峰值容忍度；窗口终点与整场训练终点需区分。

#### 完成质量与重复故障

![四条完整 Hybrid 恢复轨迹完成到第 10000 step 的质量及归一化结果。](docs/assets/benchmarks/quality_full_state_terminal.png)

[查看原始矢量 PDF](docs/assets/benchmarks/quality_full_state_terminal.pdf)

四条选定轨迹的最大最终退化为 **0.00306%**，采样峰值为 **0.00454%**。这些轨迹共享种子/前缀，与独立 R2 审计分开。

#### GPU 规模与并行布局

![不含 replay 的恢复结果：64/128 GPU 与四种 64-GPU 布局中的 MoEGambit 和两种 MoC PEC 复现模式。](docs/assets/benchmarks/recovery_scaling.png)

[查看原始矢量 PDF](docs/assets/benchmarks/recovery_scaling.pdf)

各布局共用 **rank 局部 FullLoad** 分母。A/B 的 EDP=1，专家来自 checkpoint；C/D 的 EDP=2，可使用在线专家副本。这些加载耗时比与包含 replay、对照整作业 Restart 的 **35.6×** 不同。

#### GQA 与 DeepSeek 风格 MLA MoE

![小模型完整状态恢复：五个评测偏移点，分别对照 NoFault 与整作业 Restart。](docs/assets/benchmarks/quality_architecture_transfer.png)

[查看原始矢量 PDF](docs/assets/benchmarks/quality_architecture_transfer.pdf)

100-step 窗口中，相对 Restart 的最大采样峰值为 GQA MoE **0.0893%**、MLA MoE **0.0230%**。两组配置还改变了 shared expert、路由与层布局。

### 八项下游任务

![八项零样本任务得分及相对 Restart 的百分点变化。](docs/assets/benchmarks/downstream_accuracy.png)

[查看矢量 PDF](docs/assets/benchmarks/downstream_accuracy.pdf)

Restart / MoC PEC / MoEGambit 的任务等权均值为
**45.06% / 44.67% / 45.32%**；PEC 保留 128 个专家中的 16 个。
图中保留各任务的 `acc`/`acc_norm` 指标，同时展示改善与退化。

### 补充 Web 实验 · 模型更新流程与 FiQA 质量

#### 冷启动模型更新流程

![六组64卡案例中Restart与R2准入后的完整Hybrid从请求到首次HTTP响应的耗时。](docs/assets/benchmarks/web_update_pipeline.png)

[矢量 PDF](docs/assets/benchmarks/web_update_pipeline.pdf) · [SVG](docs/assets/benchmarks/web_update_pipeline.svg)

专家年龄 **50／100／150 步**，交叉 **1／4 个受影响 rank**，共六组。
两臂均完成100次新更新，再进行验证、模型加载及首次真实HTTP重排响应；
Hybrid耗时 **892–916秒**，Restart为 **1,017–1,367秒**。
更新流程加速 **1.13–1.53×**，分配的64卡GPU小时降低 **11.3%–34.8%**。
窗口终点相对验证loss变化为 **−0.0628%至+0.0355%**，每个案例一组成对计时。

#### FiQA 重排序质量

![FiQA MRR@10与nDCG@10相对Restart的百分点差异及95%成对查询区间。](docs/assets/benchmarks/fiqa_reranking_quality.png)

[矢量 PDF](docs/assets/benchmarks/fiqa_reranking_quality.pdf) · [SVG](docs/assets/benchmarks/fiqa_reranking_quality.svg)

两组案例使用 **500个固定FiQA查询** 和BM25 top-20候选，分别评测恢复时点及
继续100次更新后。图中保留原汇总的 **95%成对查询bootstrap区间**（2,000次重采样）。
全部四个窗口终点区间跨0；专家年龄150步／4rank的初始差异为正。
候选查询覆盖率为56.6%，平均候选召回36.8%。区间量化查询抽样不确定性；等价检验还需预先规定任务差异容限。

两张图分别展示模型更新完成时间和检索质量。
实验恢复完整旧专家权重及optimizer state，采用 **冷启动、检查点模拟状态丢失、Hybrid恢复前启用R2准入**；
HTTP probe在loopback上执行真实推理。作者确认启用R2的重跑保留了这些数值结果；结果包记录了协议更正及原始导出。更新流程收益与35.6×热恢复加速、独立R2审计分别解释。
[数据、协议与重绘命令](examples/benchmarks/README.md#web-update-and-fiqa-experiments-2026-10-20)。

### R2 审计 · 报告计数

独立冻结策略审计中，实际整场质量越界 **3/200**，单侧 95% 上界为
**3.83%**。获准候选中有 **3/132** 越界，上界为 **5.77%**。
整场风险与准入条件风险的分母不同，汇总复核方法与来源见结果包说明。

### 脚本与复核入口

[**实验图表与结果复核说明**](examples/benchmarks/README.md) 为每项结果
列出输入、脚本、对照及测量范围，包含配对损失一致性检查、风险上界精确
计算，以及恢复、质量、重复故障和架构比较图的重绘脚本：

```bash
python -m pip install matplotlib numpy
export BENCHMARK_RESULTS_DIR=/personal/moegambit/benchmarks
mkdir -p "$BENCHMARK_RESULTS_DIR"
nohup bash examples/benchmarks/run_benchmarks.sh \
  >> "$BENCHMARK_RESULTS_DIR/reproduce.log" 2>&1 &
```

该CPU流程核对已有结果导出，并将图表、报告及逐阶段日志写入 `$BENCHMARK_RESULTS_DIR`。
GPU实验使用 [MoC保存／恢复与端到端脚本](examples/moc_system/README.md)、
[Megatron热替换示例](#多机运行示例) 和
[DeepSpeed MoE/dense示例](examples/deepspeed/run_hot_spare.sh)。
[状态、完成及风险审计工具](docs/ARTIFACT_AUDIT.md)核查新运行记录。结果包包含汇总与重绘脚本；
完整checkpoint拼接训练和预测器拟合使用另行提供的负载专用启动器及预测器包。

## 环境与配置

### 环境要求

- Linux x86_64；
- Python 3.10+；
- NVIDIA GPU，以及兼容的 CUDA 版 PyTorch；
- active 与 spare 节点均可使用 NCCL；
- Megatron 示例需要 Transformer Engine；
- Qwen3-MoE DeepSpeed workload 需要 `transformers>=5.0.0,<6`；
- 所有节点共享数据集和 checkpoint，或使用完全相同的挂载路径；
- 主机内存足以容纳 optimizer replica 和预取 expert state。

必要依赖集合：

| 组件 | 必要依赖 |
| --- | --- |
| MoEGambit runtime | Python `>=3.10`；带 `torch.distributed` 的 PyTorch；GPU 作业需要 NCCL |
| Megatron-LM 0.15.3 | `torch>=2.6.0`、`numpy<2.0.0`、`packaging>=24.2`、Transformer Engine；编译可选 dataset helper 时需要 `pybind11` 和 C++17 编译器 |
| 仓库内 Megatron workload | `sentencepiece`、`tiktoken`、Hugging Face 兼容 tokenizer 目录以及 CUDA/NCCL |
| DeepSpeed 0.19.3 | `torch>=2.0.0`、`einops`、`hjson`、`msgpack`、`ninja`、`numpy`、`packaging>=20.0`、`psutil`、`py-cpuinfo`、`pydantic>=2.0.0`、`tqdm` |
| 仓库内 DeepSpeed Qwen3-MoE workload | `transformers>=5.0.0,<6`、`accelerate`、CUDA/NCCL，以及本地 `./DeepSpeed` 安装 |
| 开发检查 | `pytest>=7.0` |

### 已验证软件栈

你提供的环境快照对应以下集群软件栈：

| 软件包/runtime | 已验证版本 |
| --- | --- |
| PyTorch | `2.6.0+cu126` |
| CUDA runtime | `12.6.77` |
| NCCL | `2.21.5` |
| NumPy | `1.26.4` |
| Transformer Engine | `2.4.0.dev0+3b411e79` |
| Triton | `3.2.0` |
| Accelerate | `1.10.1` |
| Einops | `0.8.1` |
| HJSON / msgpack | `3.1.0` / `1.1.0` |
| Ninja / pybind11 / Cython | `1.11.1.4` / `2.11.1` / `3.0.12` |
| Packaging / psutil / py-cpuinfo | `24.2` / `7.0.0` / `9.0.0` |
| Pydantic / tqdm | `2.10.3` / `4.67.1` |
| sentencepiece / tiktoken | `0.2.1` / `0.7.0` |

该快照记录框架升级前的环境，包括 `transformers==4.45.0` 和 `deepspeed==0.16.2`。
运行现有示例时，将Transformers升级到 `>=5.0.0,<6`，再安装内置 `./DeepSpeed`，
使 `deepspeed.__version__` 为 `0.19.3`。

环境快照中的 `flash_attn`、`grouped_gemm`、`megablocks`、`torchvision`
和 `torchaudio` 不是当前恢复脚本的强制依赖。仅当所选模型或 kernel 路径
需要它们时再安装。

项目不会固定一套适用于所有集群的 CUDA PyTorch wheel。复现上述环境时使用
CUDA 12.6 对应构建；在其他集群上则应安装彼此兼容的 PyTorch、CUDA 与 NCCL。

完成所有 editable install 后，验证最终生效的环境：

```bash
python - <<'PY'
from packaging.version import Version
import deepspeed
import numpy
import torch
import transformers

assert Version(torch.__version__.split("+", 1)[0]) >= Version("2.6.0")
assert Version(transformers.__version__) >= Version("5.0.0")
assert Version(transformers.__version__) < Version("6")
assert Version(deepspeed.__version__) >= Version("0.19.3")
assert Version(numpy.__version__) < Version("2.0.0")
assert torch.cuda.is_available()
assert torch.distributed.is_available()
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"deepspeed={deepspeed.__version__} "
    f"transformers={transformers.__version__} numpy={numpy.__version__}"
)
PY
```

### 示例路径

将以下示例路径设为自己的数据集、tokenizer与输出目录。
请在所有训练节点和备用节点设置一致的共享挂载路径：

```text
dataset:
  /shared/moegambit/data/train_text_document

Megatron checkpoint:
  /shared/moegambit/checkpoints/megatron

Megatron logs:
  /shared/moegambit/logs/megatron

DeepSpeed run root:
  /shared/moegambit/runs/deepspeed
```

可通过 `DATA_PATH`、`TOKENIZER_DIR`、`MODEL_CONFIG`、`CKPT_DIR`、
`TRAIN_LOG_DIR` 或 `RUN_ROOT` 覆盖。Megatron mmap 数据集同时需要
`${DATA_PATH}.idx` 和 `${DATA_PATH}.bin`。

### 网络端口

| 用途 | 默认值 |
| --- | ---: |
| Megatron rendezvous | `20117` |
| Megatron watcher | `20200` |
| DeepSpeed rendezvous | `20121` |
| DeepSpeed hot-spare coordinator | `MASTER_PORT + 100` |
| optimizer replica | 通常从 `20300` 开始 |

多机任务不得广播 `127.0.0.1`。

## Runtime CLI

根目录兼容入口会根据 adapter 分发。不指定 `--adapter` 时，保持历史
Megatron 行为：

```bash
python elastic_launcher.py --adapter megatron \
  --nproc-per-node 8 --nnodes 8 --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" --master-port 20117 \
  -- python Megatron-LM/pretrain_gpt.py ...

python elastic_watcher.py --adapter megatron \
  --port 20200 --training-nnodes 8 --nproc-per-node 8 \
  --master-addr "${MASTER_ADDR}" --master-port 20117
```

DeepSpeed 的 launcher 运行在 active 节点，watcher 运行 coordinator 和常驻
spare agent：

```bash
# Active nodes 0-7
python elastic_launcher.py --adapter deepspeed \
  --coordinator-host "${SPARE_ADDR}" --coordinator-port 20221 \
  --run-id ds-run-001 --training-nodes 8 --spare-node 8 \
  --physical-node "${NODE_RANK}" --local-world-size 8 \
  --base-master-port 20121 --rank-hot-swap \
  -- python -m deepspeed.launcher.runner ...

# Spare node 8
python elastic_watcher.py --adapter deepspeed \
  --coordinator-host "${SPARE_ADDR}" --coordinator-port 20221 \
  --listen-host 0.0.0.0 --run-id ds-run-001 \
  --training-nodes 8 --spare-node 8 --physical-node 8 \
  --local-world-size 8 --base-master-port 20121 --rank-hot-swap \
  -- python -m deepspeed.launcher.runner ...
```

## 验证与成功标准

成功的 rank 替换必须满足：

- 在配置的 step 写入 fault marker；
- survivor Python 进程没有重启；
- replacement 保持故障进程原有的 logical rank；
- 所有进程组使用同一 manifest 重建；
- 按恢复契约选择的 committed version 恢复训练；
- 第一个完整的恢复后 iteration 成功提交；
- 最终 `global_step` 等于 `TRAIN_ITERS`。

启用 ZeRO-2 时，DeepSpeed 还会在所选 run-state 目录下写入
`completed.json` 并验证 optimizer replica version。

开发检查：

```bash
bash -n \
  examples/megatron/run_hot_spare.sh \
  examples/megatron/run_dense.sh \
  examples/deepspeed/run_hot_spare.sh \
  examples/deepspeed/run_dense.sh \
  test_hotspare_replace.sh \
  test_deepspeed_hotspare_replace.sh

python -m py_compile examples/deepspeed/dense_workload.py
python -m compileall -q src
python -m pytest tests -q
git diff --check
```

CPU/GPU 冒烟测试、checkpoint 语义和 wheel 部署说明见
[验证与部署](docs/VALIDATION.md)。每个目标集群使用对应GPU恢复流程完成验证。

## 常见问题

<details>
<summary><strong>无法访问 watcher</strong></summary>

```bash
nc -vz <watcher-ip> 20200
nc -vz <rank-0-ip> 20117
```

检查路由和防火墙策略；跨节点通信不能使用 loopback 地址。
</details>

<details>
<summary><strong>NCCL 在恢复处理故障前退出</strong></summary>

Megatron 验证脚本使用：

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_ENABLE_MONITORING=0
```

变更 PyTorch 或 NCCL 后，需要重新验证这些恢复路径配置。
</details>

<details>
<summary><strong>进程组重建时恢复卡住</strong></summary>

- 确认每个 survivor 都到达相同的 safe point；
- 确认所有节点使用完全相同的代码和环境；
- 检查进程组创建顺序和 timeout 日志；
- 设置 `NCCL_DEBUG=INFO` 和 `NCCL_DEBUG_SUBSYS=INIT,NET,ENV`；
- 确认 replacement 广播的是可路由 IPv4 地址。
</details>

<details>
<summary><strong>主机内存不足</strong></summary>

```bash
export MOEGAMBIT_ZERO2_BUFFER_SLOTS=1
export MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB=64
export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE=0
```
</details>

## 安全说明

控制面只应运行在可信训练网络中。请使用每个作业独立的 token，通过防火墙限制
watcher 和 replica 端口，并且不要将 watcher 直接暴露到公网。具体配置请参阅
`moegambit.config.SecurityConfig`。

## 许可证

MoEGambit 自有 runtime 代码使用根目录的
[Apache License 2.0](LICENSE)。内置的 Megatron-LM 与 DeepSpeed 源码保留
各自上游许可证和 notice。重新分发前，请检查 [LEGAL.md](LEGAL.md)、
[Megatron-LM/LICENSE](Megatron-LM/LICENSE) 和
[DeepSpeed/LICENSE](DeepSpeed/LICENSE)。
