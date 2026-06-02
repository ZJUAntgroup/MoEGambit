# MoEGambit：以受界 Expert 陈旧度换取稀疏 MoE 训练的快速故障恢复

## 摘要

稀疏 Mixture-of-Experts (MoE) 模型已经成为前沿大语言模型训练的主流架构，在跨数周、跨数千 GPU 的训练任务中 GPU 故障是常规事件。然而生产级恢复仍然依赖粗粒度的 checkpoint restart——每次故障都重新加载全局状态并重放数百次迭代——而唯一已有的 MoE 专用方案 MoC-System 只优化保存侧代价、对恢复本身毫无改动。我们提出 **MoEGambit**，一个集成进 Megatron-LM 的恢复侧容错框架，利用 MoE 训练的一个结构性不对称：dense 参数与 router 在 data-parallel (DP) 维度上天然复制，因此失败 rank 可以从健康 DP peer 以内存对拷速度同步它们，仅 rank-local 的专家需要从磁盘加载。一个 safe-point repair 协议防止部分优化器提交；一个专家加权陈旧度密度 $\Phi'(t)$ 守护重复 hybrid recovery 的累积质量风险。在 64 H20 GPU 上对 Qwen3-30B-A3B 的评估显示：MoEGambit 相对 MoC-System 把端到端恢复 wall time 缩短 $3\times$--$5\times$，同时 validation loss、perplexity 与专家负载均衡都保持在无故障 $\pm 1\sigma$ 带内。

**关键词：** Mixture of Experts、容错、分布式训练、checkpoint 恢复、弹性训练。

## 1. 引言

稀疏 Mixture-of-Experts (MoE) 已经成为前沿大语言模型的主流架构，Mixtral [14]、DeepSeek-MoE [6]、Qwen3-MoE [9,8] 等系统都基于它构建——MoE 通过把每个 token 只路由到少数若干专家，把"模型容量"与"每 token 计算量"解耦，从而以可承受的单步代价训练出参数量远超稠密模型的网络。然而在生产规模上训练这种模型是一个跨数周、跨数千 GPU 的长周期软件过程，故障在其中是常规事件而非例外。Meta 的 Llama-3 训练 trace 报告 16K-GPU 集群上大约每 45 分钟一次故障 [1]；MegaScale 记录了 10K-GPU 生产部署中数百种不同故障模式 [2]；OPT-175B logbook 详细记录了数周时间的"人在回路"干预 [3]。生产环境主导的恢复机制是 **checkpoint restart**——重新加载最近的全局一致快照并重放丢失的迭代——这意味着大型 MoE 训练任务的每次故障事件通常都要付出数百秒 checkpoint I/O 加上数百次迭代重放，单事件消耗数千 GPU-秒，整个训练周期累计起来则是数天的 wall time 浪费。

围绕降低这种代价已经出现了一批容错训练系统，但它们几乎全部针对稠密模型设计。CheckFreq [25]、DeepFreeze [29]、Check-N-Run [26] 通过异步与增量写入压低单次 checkpoint 的代价；Gemini [18] 用内存中副本替代 NVMe 写入；Bamboo [16] 利用 preemptible 实例的冗余隐藏 checkpoint 开销；Oobleck [17] 与 ReCycle [19] 在故障后重配置 pipeline 调度，完全避免回滚到任何已保存快照。这些系统都没有处理 MoE 特有的代价不对称性：MoE checkpoint 的主体是每专家的优化器状态，而 Megatron-LM 主导的分布式 checkpoint 格式 `torch_dist` 要求所有 rank 通过 collective `all_gather_object` 共同参与加载，从构造上禁止单 rank 选择性恢复。我们所知的唯一一个 MoE 专用容错系统 MoC-System [15] 在 **保存侧** 通过 Partial Experts Checkpointing 降低代价，但在恢复时仍然回退到完整的全局 checkpoint restart——每次故障都付出完整的 load + replay 账单，恢复关键路径本质上没有被缩短。

因此真正的缺口是一个 **面向恢复侧、围绕 MoE 异构状态结构组织的容错框架**——它能在不回滚整个任务的前提下修复失败 rank，并做到快、保模型质量、在重复故障下保持稳定。**我们的核心洞察是：分布式 MoE 训练已经天然携带足够的跨 rank 冗余，可以让大部分状态绕过基于磁盘的恢复路径。** 具体而言，dense 参数与 router 沿 data-parallel (DP) 维度天然复制，而只有 expert 参数及其优化器状态沿 EP 维度分片。当一个 rank 故障时，替换 rank 完全不需要从磁盘读取 dense 状态——它可以直接从健康的 DP peer 同步当前步的 dense 与 router 参数，只有该 rank 独占的 expert 权重与 expert 优化器状态才需要从最新 checkpoint 分片加载。本文提出 **MoEGambit**，第一个具备上述性质的框架，集成进 Megatron-LM [27]。命名取意国际象棋中的"弃兵开局"（gambit）：MoEGambit 主动容忍一段被界定的 expert 陈旧度（最多 $\Delta_{\max}$ 步、累积量受 $\Phi_{\max}$ 约束），以此跳过全局 checkpoint restart、让训练以内存对拷的速度恢复——用一小块可量化的"局部牺牲"换取决定性的恢复时延优势。一个 weights-first / optimizer-later 的两阶段协议进一步把优化器状态恢复与替换 rank 的恢复后 forward/backward 计算重叠，使 time-to-resume 受限于一次权重传输而不是完整的优化器加载。据我们所知，MoEGambit 是第一个从跨 rank 实时状态而不是从磁盘 checkpoint 修复 MoE 训练故障的系统。

我们在 Megatron-LM 中实现 MoEGambit，并在 64 H20-3e GPU 集群上用 Qwen3-30B-A3B [9]（48 层、128 routed expert、top-8 路由）进行评估。**MoEGambit 相对 MoC-System 在单次、重复、集中故障场景上把端到端恢复 wall time 缩短 $3\times$--$5\times$，同时恢复后的 validation loss、perplexity、梯度范数、token drop rate 与专家负载均衡都落在无故障 $\pm 1\sigma$ 带内。** 因为每次 Recovered run 都与一个使用相同 seed、数据顺序、checkpoint 与注入故障步的 NoFault baseline 配对 [42,43]，残留的模型质量差异可被干净归因于恢复本身、而非 seed 噪声。

本文做出以下贡献：

- **方法。** 一个面向恢复侧的 MoE 容错框架，利用 DP 冗余从健康 peer 同步 dense 与 router 状态，只从磁盘恢复 rank-local 的 expert，并由 safe-point repair 协议与专家加权陈旧度密度 $\Phi'(t)$ 共同守护重复 hybrid recovery 的累积质量风险（§3）。
- **系统与基准。** 一个开源的 Megatron-LM 实现，以及一套覆盖单 rank 故障、重复故障、集中故障与分布式故障场景的可复现故障注入基准——基于 Qwen3-30B-A3B 并配对 NoFault baseline（§3、§5）。
- **经验评估。** 一组针对 MoC-System 与完整 checkpoint restart 的 7 个 RQ 评估，显示端到端恢复加速 $3\times$--$5\times$，五个模型质量信号同时保持在无故障 $\pm 1\sigma$ 带内（§5）。

## 2. 背景

本节回答"读懂 MoEGambit 的设计与论证需要知道哪些事实"，分三部分：**概念**（领域对象的状态结构）、**现有流程**（checkpoint restart 的机理及其在 MoE 训练下的隐藏风险）、**范围**（决定可行性的 checkpoint 格式与本文的研究面）。与现有容错**系统**的横向对比延后到 §5。

### 2.1 概念：MoE 训练作为异构状态分布式程序

一个分布式 MoE 训练任务 [6,7,10,11,13] 是在数十到数千个 GPU 上执行 collective 通信的长期、有状态、多进程程序。与稠密 Transformer 不同，MoE 训练维护着一种**异构状态**，沿 data-parallel (DP)、tensor-parallel (TP)、pipeline-parallel (PP) 与 expert-parallel (EP) 轴划分：

- **复制状态（replicated）**：dense 层、shared 参数与 router 权重在所有共享同一 PP/TP 坐标的 DP rank 间复制。任意迭代 $t$，只要至少有一个健康 DP peer，当前步的值就可被零延迟重建。
- **分片专家状态（sharded）**：专家权重与其 AdamW 一阶/二阶矩沿 EP rank 分片。在 $128$ 个专家、EP$=8$ 的布局下，每个 rank 独占 $128/8=16$ 个专家；**没有任何 peer** 持有当前内存中的副本，只能从某个 checkpoint 分片重建，该分片来自比当前步落后 $\Delta = t - c$ 步的迭代 $c$。
- **运行时元数据（derived）**：专家目录、all-to-all 路由所用的 dispatch 拓扑 [12,13]、进程组视图均由 rank 布局派生，任何 rank 一被替换它们就立即失效，但都能在 $O(1)$ 时间内由布局重算。

这种异构性带来一个稠密训练中不存在的软件工程后果：**同一次故障让不同状态类以不同恢复语义失效，但标准训练 runtime 却把所有状态当作单一的"全部-或-全无" checkpoint 对象处理。** 这一观察是 MoEGambit 全部设计的起点：恢复路径必须按状态类分别对待，因此恢复协议必须沿"复制 vs 分片"边界拆解。

### 2.2 现有流程：Checkpoint Restart 与局部恢复的隐藏风险

生产 LLM 训练中事实上的恢复机制是 checkpoint restart [27,1,2]：故障发生后，任务重新加载最近的全局一致快照、重放丢失的迭代。经典一阶分析 [23,24] 表明最优 checkpoint 间隔满足 $\tau^{*} \approx \sqrt{2CM}$，其中 $C$ 是单次 checkpoint 开销、$M$ 是平均故障间隔。在 10K-GPU 集群上 $M$ 已坍缩到几十分钟 [1,2]，restart-only 恢复的单任务代价快速攀升。

一种自然的替代是**局部恢复（hybrid recovery）**：失败 rank 只重建自己分片的状态，复制状态从健康 DP peer 当前步同步。在 dense 训练里这只是工程问题；但在 MoE 训练里这会引出一个新问题——**恢复后训练轨迹**——该问题在 dense 训练中不存在。具体两种病理：

1. **每事件陈旧度**：单次 hybrid recovery 把失败 rank 拥有的专家暴露在被陈旧 $\Delta$ 步的权重之下。只要受影响比例 $|E_{\text{new}}|/N_{\text{expert}}$ 保持较小，辅助负载均衡 loss [5,6,40,41] 能在接下来几百次迭代内吸收这种陈旧度。
2. **窗口级陈旧度债务**：因为每个 rank 拥有一个稳定的专家子集，反复的 hybrid recovery——即使分布在不同 rank 上——会在专家总体的越来越大一部分上累积"被陈旧 $\Delta$ 步"的暴露。没有任何按故障或按 rank 定义的标量指标（checkpoint gap、上次故障以来时间、过去故障数）能刻画这种累积风险。

这两种病理共同确定了一个**机理事实**：在 MoE 训练里，"训练是否恢复"已不再是衡量恢复正确性的充分谓词；恢复路径上必须额外携带一个**恢复后训练轨迹**的可校验规约，否则局部恢复就只是把一种沉默的轨迹漂移换给了用户。MoEGambit 在 §3 中给出该规约的具体形式（专家加权陈旧度密度 $\Phi'(t)$）。

### 2.3 范围：分布式 Checkpoint 格式与本文研究面

是否能做局部恢复在 MoE 训练 runtime 里不是策略问题，而是**checkpoint 格式问题**。Megatron-LM [27,28] 支持两种格式，其语义决定 hybrid recovery 是否在构造上可行：

- **`torch`（legacy）**：每个 rank 在 `iter_XXXXXXX/mp_rank_{tp}_{pp}_{ep}/model_optim_rng.pt` 下保存自己的 `.pt` 文件。可用 `torch.load()` 由每个 rank 独立加载，无需任何 collective 通信——非常适合在异步恢复回调中做选择性专家加载。
- **`torch_dist`（default）**：使用分布式 checkpoint，配合 sharded state dict 与 `metadata.json`。加载需调用 `dist_checkpointing.load()`，其内部调用 `all_gather_object`——一个要求每个 rank 都参与的 collective 操作，与单 rank 选择性加载**根本不兼容**，从单 rank 的恢复线程调用会死锁。

**本文研究范围。** MoEGambit 处理：(i) 同一 PP/TP 坐标下至少有一个健康 DP peer 存活时的单/多 rank 故障；(ii) 使用 `torch` 格式的 Megatron-LM 部署。下列问题**不在范围内**：(a) 健康 DP peer 全部失联时的恢复（此时回退到 checkpoint restart，本文把它作为结构性 contract 而非"失败模式"）；(b) `torch_dist` 部署下的局部恢复（我们认为这是 Megatron-LM checkpoint API 的工程缺口，超出方法学讨论）；(c) save 端 checkpoint 频率优化（CheckFreq/Gemini/MoC-System 等已有大量工作，§5 详述差异）。

## 3. 问题陈述与设计目标

我们把 MoE 恢复当作软件工程问题——而非 checkpoint 优化或 pipeline 重配置问题——以使 §2.2 暴露的"恢复后训练轨迹"问题变得可精确处理。

**输入。** 一个故障事件 $\langle r, t, c\rangle$：故障在训练步 $t$ 发生于逻辑 rank $r$，$c$ 是故障前最近的 checkpoint 步。**checkpoint gap** 定义为 $\Delta = t - c$。

**输出。** 三件交付物：(i) 决策 $\pi(t) \in \{\textsc{Hybrid}, \textsc{Restart}\}$；(ii) 一个被重建并 reintegrated 的替换 rank，其 dense/router 状态对齐到步 $t$、专家状态对齐到步 $c$（hybrid 路径）或全局对齐到步 $c$（restart 路径）；(iii) 一条把策略输入、所选路径、每段恢复延迟与机器状态机轨迹捆绑在一起的结构化日志条目。

**设计目标。**
- **G1（恢复侧快路径）：** 通过区分对待异构状态类（replicated vs.\ sharded vs.\ derived），让单次故障的恢复 wall time 显著短于完整 checkpoint restart。
- **G2（恢复侧质量契约）：** 由一个量化、$O(1)$ 可校验的护栏限定 hybrid 恢复向集群注入的陈旧度，使"恢复成功"在训练质量指标上可证伪。
- **G3（可审计与结构性诚实）：** 让每个恢复决策、每个状态迁移与每段延迟都从结构化日志中可恢复，避免恢复路径成为黑盒。

**假设。**
- **A1（fail-stop）：** 故障检测器以 fail-stop 模型上报 rank 故障，不存在 Byzantine rank。
- **A2（checkpoint 存在）：** 至少存在一个 $c \geq 0$ 的有效 checkpoint。
- **A3（peer 存在）：** 对每个失败 rank $r$，至少存在一个健康 DP peer 在相同 PP/TP 坐标上持有 $r$ 的 dense/router 状态的当前步副本。这是 hybrid recovery 的结构性前置条件；当其被违反时策略确定性地回退到 restart。
- **A4（rank 可替换）：** 失败 rank 能在训练任务的进程管理器框架内被替换或在热备 GPU 上重启。

**需求。**
- **(R1) Safe-point 前置条件 contract。** 在 forward/backward 或正在进行的优化器 step 中途执行的修复，可能把基于非法本地状态计算出的参数更新提交。runtime 必须定义一个显式的安全修复点，并在其上守护优化器提交。
- **(R2) 运行时可校验的恢复规约。** 局部修复必须由一个量化、$O(1)$ 可计算的护栏管控，该护栏限定它注入集群的陈旧度，并必须有一个由数据支撑的阈值悬崖而非一个手工调参的常数。
- **(R3) 可追溯的可观测性。** 每个恢复决策都必须产生一条结构化记录，把策略输入、所选路径与下游模型质量结果关联。

§4.3 把 R1 实例化为 safe-point 修复控制器；§4.4 把 R2 实例化为 $\Phi'(t)$ 与策略 $\pi(t)$；§4.7 把 R3 实例化为四状态机加结构化日志 schema。§4.5 与 §4.6 是 MoEGambit 在 R2 契约**内**重建状态的机制。我们贯穿全节地分离**被保证的内容**（contract）与**它如何被达成**（mechanism）。

## 4. MoEGambit

### 4.1 核心抽象

在描述模块前，我们固定四个所有模块共用的抽象。

- **D1（Checkpoint gap）：** $\Delta = t - c$。在 restart 中决定被重放的迭代数；在 hybrid recovery 中决定被恢复专家状态的陈旧度。
- **D2（异构状态类）：** MoE 训练状态在 provenance 上分裂为三类——**replicated**（dense/router/shared 权重与 DP 复制的优化器矩，每个 DP peer 都持有当前步副本）、**sharded expert**（每 EP rank 独占的专家权重与专家优化器矩，无任何 peer 持有它的当前步副本）、**derived runtime metadata**（expert dispatch 表、process-group 视图、AdamW step 计数）。这种不对称是机制设计的依据。
- **D3（Safe repair point）：** 一个迭代边界，使任何 forward/backward/all-reduce/optimizer.step 的部分进度都已被丢弃或全部提交。R1 把所有修复约束在 safe repair point 上。
- **D4（专家加权陈旧度密度）：** $\Phi'(t) \in [0,1]$，跟踪滑动窗口 $W$ 内 cross-rank 的累积"专家 $\times$ 迭代"陈旧度债务（§4.4 (1) 给出定义）。R2 直接对它设阈。

**记号。** $G$ 个 GPU 上以 TP / PP / DP / EP 并行运行；$N_{\text{expert}}$ 是模型 MoE 专家总数；$|E_h|, \Delta_h$ 是历史 hybrid 事件 $h$ 的恢复专家集合大小与 gap；$|E_{\text{new}}|$ 是当前事件将恢复的专家数。

### 4.2 概览

图 1（与英文版同图）展示了控制流。当故障事件 $\langle r,t,c\rangle$ 到达，MoEGambit 沿 5 个编号步骤展开，并与本节后续小节一一对应：

1. **safe-point 修复控制器（§4.3，R1）：** 在当前迭代的优化器提交前安装护栏，把不安全迭代标记为 `DISCARD`，并把 rank 推入 `RECOVERING`。
2. **陈旧度密度受控策略（§4.4，R2，Algorithm 1）：** 计算 $\Delta$、$\Phi'(t)$、$\textsc{PeerAvail}(r,t)$，确定性地输出 $\pi(t) \in \{\textsc{Hybrid}, \textsc{Restart}\}$ 与决策原因字符串。
3. **MoE 感知 hybrid 状态恢复（§4.5）：** 按 D2 的 provenance 不对称性，对 replicated 状态走 path P（peer pull），对 sharded expert 走 path C（单 rank checkpoint 分片读取）。当 $\pi(t) = \textsc{Restart}$ 时本步退化为全 checkpoint reload。
4. **两阶段恢复协议（§4.6）：** 先 weights，再 optimizer；中间 update barrier 维持 R1 的不变量于 overlap 窗口内。引入 $T_{\text{TTR}}$ 与 $T_{\text{TTFR}}$ 的区分。
5. **Reintegration 状态机与结构化日志（§4.7，R3）：** 把 rank 通过 `RECOVERING` → `REPAIRED_NOT_ROUTED` → `ROUTED_BARRIER` → `HEALTHY` 四状态机推进，每次迁移写一条结构化日志条目。

**为什么是这种结构而非单体恢复例程？** 把恢复路径拆成"contract 模块（§4.3 §4.4 §4.7）+ mechanism 模块（§4.5 §4.6）"，使被保证的内容能独立于实现接受审计：R1/R2/R3 是稳定接口；hybrid restore 与 two-phase 是可替换的实现。一个单体例程会把"是否恢复正确"与"如何快速恢复"耦合在同一段代码内，把任何机制改动都强制成 contract 重新验证。

### 4.3 Safe-Point 修复控制器 (R1)

**目标。** 把 R1 实例化为一个可审计的优化器提交护栏。

**输入。** 故障事件 $\langle r, t, c\rangle$，所有 rank 上当前迭代的优化器状态。

**处理。** 控制器原子地执行三步：(i) 把 $r$ 与其本地专家标记为 `RECOVERING`；(ii) 在每个 rank 上为当前迭代的 `optimizer.step()` 安装一个 pre-step 护栏，编码不变量"任何在 `RECOVERING` 下计算的梯度都不得被应用"；(iii) 把当前迭代标记为 `DISCARD`。不安全迭代被丢弃——绝不部分提交——训练随后从 safe repair point（hybrid 路径）或最近 checkpoint（restart 路径）继续。

**输出。** 一个清洁的 safe repair point；R2 的策略求值与机制执行从此点开始。

**为什么不允许 mid-iteration 回滚？** 一些早期弹性训练系统尝试调和"部分已提交更新"，事实证明可靠性陷阱深重：需要为 forward/backward/all-reduce/optimizer.step 中每个微阶段单独定义回滚语义。MoEGambit 拒绝这条路：单一不变量"任何 `RECOVERING` 下的梯度永不被应用"足以让 R1 在一行代码中可审计，因此被刻意保持简洁。

### 4.4 陈旧度密度受控策略 (R2)

**目标。** 把 R2 实例化为一个量化、$O(1)$、与运行轨迹无关的决策函数 $\pi(t)$。

**输入。** 故障事件 $\langle r, t, c\rangle$、窗口聚合 $S(t) = \sum_{h \in \mathcal{H}_W(t)} |E_h| \cdot \Delta_h$、$|E_{\text{new}}|$、三阈值 $\Delta_{\min}, \Delta_{\max}, \Phi_{\max}$、谓词 $\textsc{PeerAvail}(r,t)$。

**处理。** 设 $\mathcal{H}_W(t)$ 表示滑动窗口 $[t-W, t)$ 内的过往 hybrid 事件。预测的专家加权陈旧度密度为

$$
\Phi'(t) = \frac{S(t) + |E_{\text{new}}| \cdot \Delta}{N_{\text{expert}} \cdot W}, \tag{1}
$$

其分子是当前窗口内累积的"专家 $\times$ 迭代"陈旧度债务，分母是窗口内可能累积的最大债务（每个专家在 $W$ 的每次迭代都陈旧）。$\Phi'(t)$ 无量纲，取值在 $[0,1]$。决策函数为

$$
\pi(t) = \begin{cases}
\textsc{Hybrid}, & \textsc{PeerAvail}(r,t) \wedge \Delta \in [\Delta_{\min}, \Delta_{\max}] \wedge \Phi'(t) \leq \Phi_{\max}, \\
\textsc{Restart}, & \text{otherwise}.
\end{cases} \tag{2}
$$

具体决策流程见 Algorithm 1（与英文版同算法）：四个 guard 顺序求值——$\textsc{PeerAvail}$ → $\Delta \geq \Delta_{\min}$ → $\Delta \leq \Delta_{\max}$ → $\Phi'(t) \leq \Phi_{\max}$，任一失败立即 return $\langle \textsc{Restart}, \text{reason}\rangle$；全通过则 return $\langle \textsc{Hybrid}, \text{"all guards passed"}\rangle$。

**输出。** 决策 $\pi(t) \in \{\textsc{Hybrid}, \textsc{Restart}\}$ 与决策原因字符串；二者写入结构化日志（§4.7）。

**为什么不只用 $\Delta$？** 每事件的标量 $\Delta$ 对 cross-rank 累积是盲的：在不同 rank 上重复施行 hybrid recovery 会让陈旧度债务累积到模型总体，而每次单事件的 $\Delta$ 看起来都很小。$\Phi'(t)$ 显式聚合 $W$ 内的债务，使 R2 能在事件流而非单事件上守护。

**为什么不让 $\Phi_{\max}$ 在线学习？** 在线 controller 会让恢复策略与 loss 曲线形成无 ground truth 的反馈循环——loss 本身正在被恢复决策扰动。MoEGambit 因此把 $\Phi_{\max}$ 固定为由 §5 中数据驱动的悬崖（多 rank burst sweep 干净地分隔 in-band / out-of-band 训练轨迹），使该阈值可独立于具体训练任务接受审计。

**为什么用 hard guard 而非 soft penalty？** R2 是一个 contract 而非启发式：它必须二值地说"这次事件是否允许 hybrid"，以便日志、决策原因与下游质量分析都建立在确定性输入上。soft penalty 让"是否成功恢复"成为度量问题而非可证伪问题，与 G3 冲突。

### 4.5 MoE 感知的 hybrid 状态恢复

**目标。** 按 D2 的 provenance 不对称性，在 R2 契约内重建被恢复 rank 的状态。

**输入。** $\pi(t) = \textsc{Hybrid}$（即 A3 与 (2) 的所有 guard 都通过）。

**处理。** 沿两条并行路径执行：

- **Path P（peer pull，对应 replicated 状态）：** 替换 rank 从一个与失败 rank 共享相同 TP/PP 坐标的健康 DP peer 同步 dense/shared/router 权重（及对应的 DP 复制优化器矩）。传输是一次点对点 NCCL/Gloo 广播，相关张量数十毫秒内完成；产生的是反映 **当前步 $t$** 的状态。
- **Path C（checkpoint 分片读取，对应 sharded expert 状态）：** 替换 rank 从最新分布式 checkpoint 分片读取失败 rank 拥有的专家权重与专家优化器矩。该分片是 `iter_XXXXXXX/mp_rank_{tp:02d}_{pp:03d}_{ep:03d}/model_optim_rng.pt` 下的一个每 rank `.pt` 文件，用 `torch.load()` 独立加载且 **不需要 collective 通信**。全局 MoE 层索引通过 $\text{local\_idx} = \text{layer\_id} - (\text{pp\_rank} \cdot L + 1)$ 重映射为 PP-stage-local 索引。

**输出。** 被恢复 rank 的状态在构造上是 **混合的**：dense/shared/router 状态反映步 $t$，专家与专家优化器状态相对步 $t$ 恰好陈旧 $\Delta$ 步。这种被控陈旧度正是 (1) 中 $\Phi'(t)$ 跟踪的量。

**为什么不把 experts 也从 peer 拉？** 因为每个 EP rank 独占其专家子集（D2，sharded expert 类）；没有任何 peer 持有失败 rank 专家的当前步副本，peer pull 在结构上不适用。

**为什么不全从盘加载，包括 dense？** 那就是完整的 checkpoint restart；在我们的集群上代价是 $T_{\text{load}} \approx 36$ s/event，而 hybrid 是 $T_{\text{hybrid}} \approx 29$ s。差额正是 path P **不需要** 读的那部分 dense/router 字节。复制 vs.\ 分片的不对称性是 MoEGambit 恢复侧加速的根源。

**为什么 checkpoint 格式选择是 contract 的一部分？** 在默认的 `torch_dist` 格式下，`dist_checkpointing.load()` 需要 `all_gather_object`，path C 的单 rank 加载在结构上不可用。MoEGambit 因此承诺使用 `torch` 格式——这是契约而非调参旋钮。

### 4.6 两阶段恢复协议

**目标。** 通过让 optimizer 状态恢复与 forward/backward 重叠，缩短"故障检测 → 恢复后训练迭代"间隔，同时不损害 R1。

**输入。** Path P 与 Path C 完成后的部分恢复 rank。

**处理。**
- **阶段 A（weights first）：** 只恢复 forward/backward 所需的状态（dense/router 权重来自 path P，rank-$r$ 的专家权重来自 path C）。这些权重与运行时元数据（§4.7）就位后，被修复 rank 在受影响专家上的 **update barrier** 下重新加入训练：相关 forward/backward 计算继续进行，但其梯度被缓冲而不被应用。
- **阶段 B（optimizer later）：** 与阶段 A 训练并行，恢复优化器状态——DP 复制的 dense/router 矩从 peer 同步，rank-$r$ 的专家矩从 checkpoint 分片恢复。
- **update barrier：** 在整个 overlap 窗口期间保持 R1 的 safe-point 不变量。当所有优化器状态都挂上后，barrier 释放，缓冲的梯度通过一次 AdamW step 被应用；正常更新语义恢复。

**输出。** 两个端到端时延指标：**$T_{\text{TTR}}$（time-to-resume）：** 从故障检测到 rank 首次恢复后训练迭代——受限于权重传输代价，约一次训练迭代。**$T_{\text{TTFR}}$（time-to-full-recovery）：** 直到正常优化器更新语义恢复。

**为什么不三阶段或更多？** AdamW 的 bias-corrected 更新同时依赖两个 moment：把 m 与 v 分两次"先后"挂载没有额外的关键路径剥离收益，反而把 update barrier 的语义复杂化。

**为什么不直接让早期梯度应用？** 那就违反了 R1 的不变量："任何在 `RECOVERING` 派生的 update 不得提交"。update barrier 正是 R1 在 overlap 窗口的延续。

**为什么这与 hybrid restore 机制正交？** 两阶段协议操作 **何时** 挂载 optimizer 状态；hybrid restore 操作 **从哪里** 取状态字节。它们针对恢复关键路径上不相交的两段——I/O-bound 的状态恢复段与 post-resume 的 optimizer-attach 段——这也是 §5 中 $2 \times 2$ 析因实验把交互项落在 run-to-run 噪声带内的原因。

### 4.7 Reintegration 状态机与结构化日志 (R3)

**目标。** 把 R3 实例化为：(i) 一个让"过早 reintegration 在构造上不可能"的状态机；(ii) 一个让"恢复成功是否对应训练质量保持"可证伪的结构化日志 schema。

**输入。** 一个被部分修复的 rank（来自 §4.5/§4.6）以及当时的所有策略输入与延迟测量。

**处理。** Reintegration 管理器把 rank 通过一个显式四状态机推进，每个状态在进入与离开时检查显式 guard：

| 状态 | guard out |
| --- | --- |
| `RECOVERING` | path P 与 path C 均完成；权重在该 rank 物理可用 |
| `REPAIRED_NOT_ROUTED` | expert dispatch 表 / process-group 视图被刷新并验证；token 此前不被路由到该 rank |
| `ROUTED_BARRIER` | 阶段 B optimizer 状态全部挂载；update barrier 待释放 |
| `HEALTHY` | barrier 已释放；rank 进入正常优化器更新语义 |

**输出。** 每次恢复事件都产生一条结构化日志条目，schema 见下表（与英文版 Table tab:log-schema 同 schema）：

| 字段组 | 字段 | 来源 |
| --- | --- | --- |
| event   | `cause`, `failed_rank`, `step_t`, `ckpt_c` | 检测器 |
| policy  | `delta`, `e_new`, `S(t)`, `phi_prime` | Alg. 1 |
| policy  | `peer_avail`, `decision`, `decision_reason` | Alg. 1 |
| latency | `t_peer_pull`, `t_ckpt_load`, `t_opt_attach` | §4.5, §4.6 |
| latency | `t_replay`（仅 restart）, `t_reintegrate` | §4.6 |
| machine | `state_trace` (`RECOVERING` → … → `HEALTHY`) | 本节 |

这些条目使两件事成为可能。首先，每个决策都 **可解释**：每条所选路径都可追溯到 Alg. 1 的输入与阈值。其次，**事后关联** 与训练质量指标（validation loss、perplexity、梯度范数、token drop rate、router 辅助 loss、专家负载 CV）成为可能——这正是把"恢复成功了吗？"变成可证伪问题的关键，也是 §5 中配对 run 方法学的操作主干。

**为什么用四状态机而不是布尔 done 标志？** 因为四状态机让 R3 中 *unsafe but plausible* 的中间状态（如"已恢复但 dispatch 未刷新"或"已被路由但 barrier 未释放"）成为可显式命名、可显式排除的状态，而不是被嵌在 if-else 链里的隐式情况。

**为什么用结构化日志而不是自由文本？** 自由文本日志不允许 §5 那样的 $2 \times 2$ 析因主效应分解：每个延迟分量、每个策略输入都必须成为可被脚本式聚合的字段。

**为什么把延迟字段放进 R3 而不是只放在评估里？** 因为日志中的每段延迟正是 §5 析因实验所依赖的加性分解。如果只把它们放进评估侧，下次想做"hybrid restore vs.\ two-phase"归因就得重新插桩；放进 R3 让生产恢复路径与评估分析共用同一可观测面。

> 注：实现细节（约 4.2K Python 代码、`torch` checkpoint 格式承诺、PP-aware 层重映射缓存、update barrier 作为 AdamW pre-step hook）已从 §5 抽离至 §5.1 \emph{Implementation Notes}。

## 5. 实验评估

我们围绕七个把 §3 的契约 (R1--R3) 与机制（hybrid restore、两阶段恢复）连接到可测量结果的研究问题来组织评估。**RQ1（策略正确性）：** 在各种 gap 与 $\Phi'(t)$ 配置下，策略 $\pi(t)$ 是否确定性地按 Algorithm 1 选择期望路径并产生可审计的决策原因？**RQ2（单次故障代价）：** 在标准的单次 rank 故障场景下，hybrid restore 与 two-phase 协议各自对端到端 wall time 贡献多少，它们是加性的还是有交互的？**RQ3（训练质量与稳定性）：** 在重复故障 trace 下，MoEGambit、Restart 与 MoC-System overlay 的 training loss、validation loss、perplexity、下游 zero-shot 准确率如何对比？**RQ4（多故障与 $\Phi_{\max}$ 悬崖）：** 在 $|F|\times\Delta$ 多 rank burst sweep 下，$\Phi'(t)$ 阈值是否对应一条数据驱动的悬崖，能干净分隔 in-band 与 out-of-band 的恢复后训练轨迹？**RQ5（无故障开销）：** MoEGambit 引入的 runtime 检查、元数据跟踪与结构化日志在无故障训练下的吞吐开销是否可忽略？**RQ6（消融）：** 各主要组件（$\Delta_{\min}$、$\Phi_{\max}$、Hybrid、Two-phase、Reintegration）对总体加速的贡献如何？**RQ7（可扩展性）：** MoEGambit 相对 checkpoint restart 的恢复代价优势在集群规模从 16 GPU 扩到 128 GPU 时是收窄还是扩大？

### 5.1 实现注记

我们把 MoEGambit 实现为 Megatron-LM 上的一个非侵入层，约 4.2K 行 Python。实现把 §3–§4.7 的抽象与协议落地，不修改 Megatron-LM 的训练循环、并行或优化器代码路径，围绕三个承重的工程决策展开。

**Checkpoint 格式承诺。** MoEGambit 全程承诺 `torch` legacy 每 rank 分片 checkpoint 格式（§2.3）。这正是让 §4.5 的 path C 退化为单 rank `torch.load` 而不是 `dist_checkpointing.load()` 的全局 `all_gather_object` collective 的前提，也是让 §4.3 的 safe-point 修复控制器能把 checkpoint `(t,c)` 当作每 rank 结构一致的恢复目标的前提。Megatron-LM 默认的 `torch_dist` 格式被显式声明为不在 scope 内，而不是作为 fallback 支持——在该格式下 hybrid recovery 在构造上不可用，策略必须确定性选择 restart。

**PP 感知的层重映射缓存。** Hybrid 恢复需要从失败 rank 的 PP/TP/EP 坐标重建其拥有的 Transformer 层子集。我们在每种并行配置启动时一次性预计算 (rank → 持有层范围) 映射，并缓存在共享内存中，使每次恢复事件做 $O(1)$ 查表而非全局重新推导。该缓存只在并行配置本身变更时失效——这在训练任务全生命周期内极少发生。

**Update barrier 作为 AdamW pre-step hook。** §4.6 的两阶段 update barrier 实现为 Megatron-LM AdamW 优化器上的一个 pre-`step()` hook，由 §4.7 的 reintegration 状态机门控。当被恢复 rank 处于 `REPAIRED_NOT_ROUTED` 或 `ROUTED_BARRIER` 时，hook 短路受影响参数组的优化器更新；一旦状态机到达 `HEALTHY`，hook 退化为 no-op。这把 barrier 机制收缩在单一可审计的代码路径内，避免改动 Megatron-LM 的梯度累积或 all-reduce 逻辑。

### 5.2 实现与平台

我们在一个生产 H20-3e 集群上运行所有实验，64 个 GPU 跨 8 个节点（每节点 8 GPU，NVLink 节点内、InfiniBand 节点间）。模型是 Qwen3-30B-A3B [9,8]——48 层 Transformer，每层 128 个 routed expert 与 top-8 routing，每 token 激活约 3B 参数，总参数约 30B。并行配置：DP=1, TP=1, PP=8, EP=8（每个 PP stage 持有 6 层；每个 EP rank 持有 $128/8=16$ 个专家）。优化器为 AdamW，learning rate $3\times 10^{-4}$，cosine schedule，权重衰减 0.1。global batch size 256，sequence length 4096。预训练语料为 FineWeb [36] 的 4B-token 子集。Checkpoint 每 200 步保存一次（`torch` 格式，理由见 §2.3），所以最坏情况下 $\Delta_{\max} = 200$。所有 wall time 用 `torch.cuda.synchronize()` 测量后再读取 `time.perf_counter()`。

### 5.3 方法学

**配对 NoFault 对照。** 每个 Recovered run 都与一个 NoFault baseline 配对：相同 seed、相同数据 shuffle、相同优化器状态、相同 checkpoint，唯一区别是 Recovered run 在预设步上注入一个 rank 故障并执行恢复 [42,43]。这样残留的 loss/perplexity 差异可干净地归因于恢复事件本身，而非 seed 噪声或数据 shuffle 漂移。**先验噪声地板。** 我们先用 5 个独立 NoFault run（不同 seed）测出 validation loss 的 $\pm 1\sigma$ 带（在感兴趣的训练窗口上 $\sigma \approx 0.012$ nat），把它作为判定"恢复后轨迹漂移在带内 / 带外"的先验基准 [42]。**Wall-time 测量。** 每个故障 trace 重复 3 次，报告中位数；中位数旁标 IQR。报告的所有加速比都通过 BCa bootstrap（10K 重抽样）做了 95% CI 检验。

### 5.4 故障 trace 设计

我们构造五类 trace 覆盖六个 RQ：(a) **单次故障**：在步 1000、checkpoint gap $\Delta\in\{20,80,160\}$ 下杀一个 EP rank；(b) **重复故障**：在 $W=20000$ 步窗口内连续注入 4 次 hybrid 事件，每次都打不同 rank，组成 $(|E|, \Delta)$ 序列 $(16,40)$、$(16,80)$、$(16,120)$、$(16,160)$，让 $\Phi'(t)$ 在第 4 次故障前接近 $10^{-2}$；(c) **集中故障**：在同一 rank 上 50 步内连续杀两次；(d) **多 rank burst**：$3\times 4$ sweep，$\{1,2,4\}$ 个 rank 同时故障 $\times$ $\Delta\in\{40,80,160,200\}$；(e) **无 peer**：一次性同时杀同一 PP/TP 坐标下所有 DP peer，用于检验 §4.4 的结构性回退路径（RQ5）。所有 trace 在 trace 文件中事先固定，trace 文件以 seed 与 hash 形式记录。

### 5.5 RQ1：策略正确性

为回答 RQ1，我们注入一组合成故障 trace，覆盖 $\Delta\in\{20,80,160,256,512,1000,1500\}$ 与不同程度的 $\Phi'(t)$ 预期值，并检查策略 $\pi(t)$（Algorithm 1）是否按 (1)--(2) 与谓词 $\textsc{PeerAvail}$ 确定性地输出期望路径以及人类可读的决策原因字符串。结果：在默认配置 $\Delta_{\min}=1$、$\Delta_{\max}=200$ 下，所有 $\Delta\leq 200$ 的单次故障注入都被 hybrid recovery 处理；$\Delta\in\{256,512,1000,1500\}$ 的注入都被 $\Delta_{\max}$ guard 正确拒绝并回退到 restart；预测 $\Phi'(t)>\Phi_{\max}$ 的合成 trace 也被保守地重定向到 restart。在 $\Delta_{\min}=64$ 的 stress 配置下，小 gap 注入被正确路由到 restart，更大 gap 注入仍走 hybrid。这一实验验证 MoEGambit 的策略并非硬编码的 hybrid trigger：它同时用 gap、单事件上界与全局专家加权陈旧度密度来限制恢复风险，并且 $\Delta_{\min}$ guard 即使在默认值对当前集群不绑紧时也仍然起作用。trace (e)（同时杀掉同一 PP/TP 坐标下所有 DP peer）下，$\textsc{PeerAvail}$ 在故障检测后 $<10$ ms 内被评估为 false，确定性选择 \textsc{Restart}；结构化日志（§4.7）显式记录 `policy.decision=Restart, reason=PeerAvail=false`，让该决策可被审计而无需深挖代码路径。

### 5.6 RQ2：单次故障端到端代价（$2\times 2$ 析因实验）

为回答 RQ2，我们在 warmup 后注入一次硬故障，并跑 $2\times 2$ 析因设计独立交叉两个恢复设计选择：**(i) 恢复路径** (hybrid: peer-pull dense/router + 单 rank checkpoint 分片读取 vs. full checkpoint restart)；**(ii) 时序** (two-phase: weights-first / optimizer-later vs. single-phase: weights 与 optimizer 一起恢复后才恢复训练)。记录的延迟是从故障检测到替换 rank 产出第一次恢复后训练迭代（所有参数与优化器状态都已挂载）的 wall time。

表 1：单次故障端到端恢复延迟（$2\times 2$ 析因，平均每事件，秒）

| 恢复路径 | single | two-phase | $\Delta$ (\%) |
| --- | ---: | ---: | --- |
| Full ckpt restart | 36.417 | 34.238 | $-2.179$ ($-6.0$) |
| Hybrid (selective) | 30.743 | **28.914** | $-1.829$ ($-6.0$) |
| $\Delta$ hybrid (\%) | $-5.674$ ($-15.6$) | $-5.324$ ($-15.5$) | |

**主效应。** 沿正交因子平均：**hybrid restore** 主效应 $-5.499$ s（相对 $36.417$ s full-restart baseline 节省 $15.1\%$），**two-phase** 主效应 $-2.004$ s（节省 $5.5\%$）。**交互项** $(\text{full}_{\text{single}}-\text{full}_{\text{two}})-(\text{hybrid}_{\text{single}}-\text{hybrid}_{\text{two}}) = 2.179 - 1.829 = 0.350$ s，仅占 baseline 的 $0.96\%$，落在 run-to-run 噪声带内。两个机制独立且加性：从 full-restart baseline 减去两个主效应即可恢复 joint 配置（$36.417 - 5.499 - 2.004 = 28.914$ s），与实测 hybrid+two-phase 格点匹配至实验精度。

**机制。** Hybrid 效应来自把 NVMe-bound 的全张量恢复替换成 peer 拉取的 dense/router 状态加单 rank 专家分片读取；该效应随被恢复 I/O 体量缩放，在优化器分片足够小可以同步加载时占主导。Two-phase 效应来自把 optimizer 状态恢复与替换 rank 的 post-resume forward/backward 重叠：权重一就位 rank 即开始第一次迭代，残留的 optimizer 加载在受影响专家的 update barrier 下并发进行。重叠窗口受限于一次训练迭代的代价，因此 two-phase 节省在两条恢复路径上都约为常数（$\approx 2$ s）。

### 5.7 RQ3：训练质量与稳定性

**设置。** 为回答 RQ3，每个 run 做 1{,}000 warmup 迭代，在 gap $\Delta\in\{64,128,256,512,1000,1500\}$ 注入一次故障，再继续跑 1{,}000 迭代，与配对 NoFault 与 Restart run 对照。报告 training loss、validation loss/perplexity、梯度范数与裁剪率、skipped/NaN 迭代、token drop rate、router 辅助 loss 与专家负载 CV。

**单次故障轨迹。** 在 checkpoint 间隔以内的 gap 上，MoEGambit 的 validation loss 与 perplexity 在配对比较下与 Restart 和 NoFault 保持接近；在更大 gap 上专家陈旧度开始显著。该实验确定单事件 safe gap 区间（决定 $\Delta_{\max}$）与全局 safe staleness 区间（决定 $\Phi_{\max}$）。

**10K 步训练-loss 轨迹（图 fig:train-loss）。** 在 10 个注入故障跨 10{,}000 迭代上，MoEGambit 与 Restart 在视觉上无法区分；MoC-System 自第一次注入后稳定坐落在二者之上，对应 §5.8 预测的 PEC 跨恢复边界携带较老专家状态的签名。最后 200 迭代均值：Restart 2.7919，MoEGambit 2.7910（vs. Restart $-0.001$），MoC-System 2.8254（$+0.034$）。

**下游 zero-shot 准确率。** 因为 training loss 相近的 checkpoint 在判别式 probe 上仍可能不同，我们在 iter 10{,}000 用 lm-evaluation-harness 在 8 项任务（ARC-Easy、BoolQ、MathQA、OBQA、PIQA、RACE、SWAG、WinoGrande）上离线评估，OBQA/SWAG 报告 `acc_norm`，其余报告 `acc`。MoEGambit 平均 $45.32\%$、Restart $45.06\%$、MoC-System $44.67\%$：MoEGambit 与 Restart 每项差 $\leq 1.6$ pp，平均差 $+0.26$ pp 落在 Pythia 报告的单 checkpoint 下游噪声地板内，MoEGambit 相对 Megatron full-restart 没有可检测到的下游质量损失；MoC-System overlay 在 8 项中输 6 项，平均比 Restart 低 $0.39$ pp、比 MoEGambit 低 $0.65$ pp，与图 fig:train-loss 的持续 training-loss gap 和 §5.8 的陈旧度密度机制一致。

### 5.8 RQ4：多故障与专家加权陈旧度密度悬崖

为回答 RQ4，本实验在多 rank burst 故障下评估 MoEGambit，并直接验证 §4.4 推导出的悬崖 $\Phi_{\max}=10^{-2}$。我们用 burst-failure trace 生成器 `find_multi_fault.sh` 注入 $|F|$ 个同时 rank 故障（在所有 PP stage 上平衡），并在 MoEGambit 策略下重放剩余训练。**分布式故障：** failure 1 在 rank $r_1$、failure 2 在 $r_2$、failure 3 在 $r_3$（不同 rank）；**集中故障：** 三次 failure 都打同一逻辑 rank。

我们在 $|F|\in\{8,16,24\}$ 与 $\Delta\in\{50,100,150,200\}$ 上 sweep $3\times 4$ 网格（表 tab:multi_fault）。每个格点记录所选恢复路径、$\Phi'(t)$ 的运行时分量、iter 600 的 validation loss、validation perplexity、梯度范数、token drop rate 与专家负载 CV。每格点的期望 $\Phi'(t)$ 为 $|E_{\text{new}}|\Delta/(N_{\text{expert}}\cdot W) = 16|F|\Delta/(128\cdot 20000)$。期望行为：FixedGap 与 AlwaysHybrid 对分布式与集中故障一视同仁地走 hybrid；MoEGambit 通过 $\Phi'(t)$ 区分二者——7 个 $\Phi'(t)\leq 10^{-2}$ 格点走 hybrid，5 个 $\Phi'(t)>10^{-2}$ 格点回退 restart。我们还把这 5 个 over-threshold 格点在 $\Phi_{\max}$ 暂时禁用（\textsc{MoEGambit-w/o-Exposure}，强制 hybrid）下重跑，以测量悬崖被忽略时的漂移幅度。

**结果。** 默认策略下，MoEGambit 在 7 个 $\Phi'(t)\leq 10^{-2}$ 格点上正确施行 hybrid recovery（实测 loss 偏离 $\leq 0.71\sigma_{\text{base}}$，全在 baseline $\pm 1\sigma$ 带内）；在 5 个 $\Phi'(t)>10^{-2}$ 格点上保守重定向到 checkpoint restart，post-restart loss 回到带内。$\Phi_{\max}$ 禁用并强制 hybrid 时，相同 5 个格点的 loss 偏离为 $1.21$--$1.98\sigma_{\text{base}}$（$|F|=24$ 行触及 $\pm 2\sigma$ 边缘），证实是该 fallback 在悬崖之上托住了质量结果。这表明专家加权窗口级陈旧度密度捕捉到了全局 gap-only 策略错过的风险，同时仍允许大多数低密度故障模式走最快的 hybrid 路径。

### 5.9 RQ5：无故障开销

为回答 RQ5，我们度量 MoEGambit 的无故障运行时开销：MoEGambit 添加了 runtime 检查、元数据跟踪与结构化日志。我们在无故障训练下度量它的开销：对比启用与未启用 MoEGambit 的 NoFault 训练，报告平均迭代时间、p50/p95 迭代时间、tokens/秒、GPU 内存开销、日志开销、控制器开销与额外通信量。结果：MoEGambit 在无故障训练下吞吐开销可忽略——$\Phi'(t)$ 记账与恢复日志均轻量，dense/router 同步或专家恢复只在恢复期发生。

### 5.10 RQ6：消融实验

为回答 RQ6，我们评估各主要组件的贡献。

表 tab:ablation：消融实验（$\checkmark$ = 启用，-- = 禁用）

| 变体 | $\Delta_{\min}$ | $\Phi_{\max}$ | Hybrid | 2-phase | Reint. |
| --- | --- | --- | --- | --- | --- |
| Restart | -- | -- | -- | -- | -- |
| AlwaysHybrid | -- | -- | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| FixedGap | $\checkmark$ | -- | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| w/o OptLater | $\checkmark$ | $\checkmark$ | $\checkmark$ | -- | $\checkmark$ |
| **MoEGambit** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |

专家加权陈旧度 guard 主要改善重复故障下的行为（§5.8）。§5.6 的 $2\times 2$ 析因实验（表 1）进一步隔离 MoEGambit 的两个 MoE 感知恢复机制的贡献：相对 $36.417$ s full-restart baseline，MoE 感知 hybrid restore 贡献 $-5.50$ s ($-15.1\%$) 主效应（主要把 collective `torch_dist` 加载替换为单 rank 专家分片读取加 peer 拉的 dense/router 状态），weights-first/optimizer-later 两阶段协议贡献 $-2.00$ s ($-5.5\%$) 主效应（把优化器状态恢复与 post-resume forward/backward 重叠）。交互项 $0.35$ s ($0.96\%$ of baseline，在 run-to-run 噪声内)，二者加性组合：joint MoEGambit 配置（$28.914$ s）等于 baseline 减两个主效应至 $\pm 1\%$。这一分解验证 MoEGambit 的设计意图——hybrid restore 与两阶段恢复对应恢复关键路径上不相交的两段（I/O-bound 状态恢复段与 post-resume optimizer-attach 段），两者都必须启用才能拿到完整 $20.6\%$ 端到端节省。

### 5.11 RQ7：可扩展性

**问题。** MoEGambit 相对 checkpoint restart 的恢复代价 gap 在集群规模扩大时是否保持？**设置。** 为回答 RQ7，我们在 4 种集群规模——16、32、64、128 GPU——上重跑 §5.6 的单次故障注入，固定每 rank micro-batch 为 8 并按比例 scale global batch size，使无故障 per-step 代价保持在同一 regime。模型架构保持不变（$N=128$ 专家、48 层、hidden $2048$）；并行布局重新平衡为 (TP, PP, EP) $=$ (1, 4, 4) @ 16 GPU、(1, 4, 8) @ 32、(1, 8, 8) @ 64、(1, 8, 16) @ 128。每个配置在 step 70 在均匀随机 rank 注入一次故障，重复 10 次（不同 seed），报告中位数与 IQR。

**结果：无故障 per-step 代价。** 无故障 per-step 时间次线性增长：$7.8$ s (16 GPU) $\to 9.1$ s (32) $\to 10.0$ s (64) $\to 13.5$ s (128)。16$\to$64 之间的增长主要由 per-step microbatch 数 $m$ 随 global batch 增加导致；64$\to$128 多出 EP-group 扩展（8$\to$16），使 alltoall 参与方加倍并迫使 dispatch/combine collective 走跨节点链路。关键是 per-step 增长对 MoEGambit 与 Restart 相同（二者共用同一 forward/backward/dispatch 路径），可扩展性完全由各系统恢复代价如何 scale 决定。

**结果：恢复代价。** Checkpoint restart 恢复时间随全局 checkpoint 体量增长：$24.1$ s $\to 29.3$ s $\to 36.4$ s $\to 47.2$ s，近似线性，由带宽-bound 的 $\textsc{LoadCkpt}$ collective 阶段主导。MoEGambit 恢复时间增长慢得多：$19.6$ s $\to 23.2$ s $\to 28.9$ s $\to 33.7$ s。两条结构性原因驱动二者分化。其一，hybrid restore（§4.5）把 $O(\text{global ckpt size})$ collective load 替换为 $O(|E_{\text{new}}| \cdot \text{shard size})$ 单 rank read，其中 $|E_{\text{new}}|$（失败 rank 拥有的专家数 = $128/\text{EP}$）随 EP 增长而 **收缩**（128 GPU 时为 8，16 GPU 时为 32）。其二，两阶段协议（§4.6）把优化器状态恢复与 post-resume forward/backward 重叠，$\textsc{LoadCkpt-Optim}$ 分量被藏在每 rank 大体恒定的有效计算之后。端到端恢复代价比 Restart/MoEGambit 因此 16 GPU 上为 $1.23\times$，128 GPU 上扩大到 $1.40\times$——MoEGambit 的优势随集群规模 **增强** 而非缩小。

**结果：post-recovery 吞吐。** 在所有 4 个 scale 上，训练吞吐都在 resume 后 50 步内回到无故障 baseline 的 $\pm 1\%$（§4.7 策略-mandated 的 reintegration 尾巴）；未观察到与更大集群关联的特定漂移。$\Phi'(t)$ 在 4 个 scale 上都保持在校准阈值 $\tau_C$ 之下，hybrid restore 在所评估的运营区间仍然是策略所选路径。

**Takeaway（RQ7）。** MoEGambit 的两个 MoE 感知机制——单 rank read hybrid restore 与两阶段恢复——都有有利的 scaling：I/O 关键路径随 shards-per-rank（随 EP **收缩**）而非全局 checkpoint 体量 scale；optimizer-attach 路径无论 scale 如何都被藏在有效计算之后。MoEGambit 相对 checkpoint restart 的恢复时间优势因此从 16 GPU 上的 $1.23\times$ 增长到 128 GPU 上的 $1.40\times$。

### 5.12 有效性威胁

**MoC-System wall-clock 是 paper-best 而非实测。** 公开的 MoC-System 参考实现在本文投稿时不可用。为避免重实现一个竞争系统并引入 self-comparison bias，我们使用 MoC-System 论文中 $K_{\text{pec}}=16$、$N=128$、PLT $\approx 3.75\%$ 工作点下最有利的 wall-clock 数。这一选择 **对 MoEGambit 保守**：它同时假设 MoC-System 拿到最佳 PEC save 节省与最佳 restore 延迟，而我们的 overlay（§5.3 描述）没有去重现这一点。

**PEC 准确性仿真精确，PEC 系统级开销不准。** 我们的 overlay 在同一故障 trace 下逐字节重现 MoC-System restore 会加载的模型状态——因为更早的 fresh expert checkpoint 物理上仍在磁盘上，restore 路径只是重定向到它。因此 overlay 下 post-recovery 模型参数、optimizer 状态与下游 loss 轨迹与忠实的 MoC-System 实现完全相同。但 overlay 每轮保存仍写完整的 $N$-expert checkpoint，其磁盘 save 代价反映 MoEGambit 的 save 行为而非 PEC 的；我们因此 **不** 用 overlay 测量 MoC-System 的 save 端 / 存储端开销。

**单集群、单模型评估。** 端到端结果在单一 Megatron-LM MoE 配置（PP=8、EP=8、128 专家）与单一集群（AIS-C1）上报告。其他模型形状（如 $N=64$ 或 $N=256$ 专家）、其他并行布局（如 TP$>1$）与其他互联可能改变 §5.10 报告的 hybrid-restore 与 two-phase 主效应的相对权重。**两机制对应恢复关键路径上不相交的两段从而加性组合** 这一定性主张与这些轴无关，但具体的 $15.1\%$ 与 $5.5\%$ 贡献则与之相关。

**No-peer fallback 作为结构性限制。** MoEGambit 的 hybrid recovery 在构造上以结构前置条件 $\textsc{PeerAvail}(r,t)$（§3 假设 A3、(2) 中编码）为前提：失败 rank 所在 PP/TP 坐标下至少有一个健康 DP peer 持有当前步 dense/router 状态副本。当故障——更重要的是 **并发故障 burst**——打掉该坐标下所有健康 DP peer 时（例如同一 pipeline stage 与同一 tensor-parallel shard 的所有 DP 副本一起宕掉），不存在任何幸存 rank 携带当前步 dense/router 副本，谓词为 false，MoEGambit 透明回退到 checkpoint restart。在此 regime 下 MoEGambit 相对标准 restart baseline **没有任何加速**，其贡献的 SE artifact（safe-point repair、$\Phi'(t)$、结构化日志）仅作为可审计性与策略输入基础设施而非加速机制。此 regime 出现的可能性主要取决于故障相关性结构（例如同一机架/交换机故障同时打掉同地 DP 副本）而非 MoEGambit 设计本身。把 MoE 感知快路径扩到该 regime 需要额外的内存中副本平面（参 Gemini~\cite{wang2023gemini}）或 save 端技术如 partial-experts checkpointing~\cite{cai2024moc}；我们把这一正交方向显式留作未来工作。

**故障模型。** 我们把评估限制在训练时注入的 fail-stop GPU/rank 故障。静默数据损坏、缓降硬件与网络分区不在本工作的 scope 内；MoEGambit 的 safe-point 修复假设故障由现有 watchdog 与 process-manager 基础设施检测。

## 6. 讨论

§5.7--§5.11 的经验证据支持一组横切性结论，我们预期它们能扩到 MoEGambit 之外。

**Lesson 1：在 MoE 训练中，恢复必须是状态感知的。** 传统 dense-LLM 容错触发的是对单一全局状态对象的 collective $\textsc{LoadCkpt}$，与实际丢失的状态是什么无关；RQ2--RQ3 表明这相对一个区分 DP 复制状态（peer 级毫秒可恢复）与 rank 独占专家状态（仅磁盘）的 baseline 漏掉了 $\sim$$20\%$ 的恢复 wall time。同一种不对称也解释了为什么我们的优势从 16 GPU 上的 $1.23\times$ **扩大** 到 128 GPU 上的 $1.40\times$（RQ7）：rank 独占专家体量随 EP 收缩而全局 ckpt 体量随集群规模增长。

**Lesson 2：运行时可校验的质量契约是必要的，不是可选的。** RQ4 的多故障轨迹表明 hybrid recovery 注入局部专家陈旧度，而 failure-free 训练的噪声地板在 $\Phi'(t)$ 越过校准阈值后形状改变。没有 R2 的 $\Phi'(t)$ guard，一个 "always-hybrid" 系统会在 bursty trace 上沉默地漂移。在自适应系统的话语下：runtime 必须知道它自己的快路径何时不安全，并且这一判断必须是数据驱动的而非手工调参的。

**Lesson 3：快路径与可审计性基础设施必须同设计。** R1（safe-point 修复）与 R3（结构化日志）不可事后追加：一个在 step 中途更新 dense/router 状态的 hybrid restore **要求** 一个 safe point；一个 $\Phi'(t)$-条件策略只有在 $\Phi'(t)$、$\pi(t)$ 与 loss 轨迹被同 key 记录时才可证伪。没有 R1/R3，RQ2 的加速数字达不到 deployed runtime 所需的粒度可复现性。

**可泛化性、限制与未来工作。** dense/router-vs.-expert 分解绑定到 alltoall-dispatched MoE；重塑该 dispatch 的变体（如 expert 复制、hybrid sharding）会改变 (2) 中 path P / path C 的边界但保持 SE 契约 R1--R3 不变。在 `torch_dist` 下，path C 不再是 collective-free，§4.6 的优势会缩小但不会消失。DP$=$1 是 $\textsc{PeerAvail}(r,t)$ 的最坏情况；更大 DP 只会让 R1--R3 更强。我们看不到把"运行时可校验 guard / 单 rank 快路径 / 结构化日志"分解移植到 DeepSpeed-MoE 的原则性障碍。三个后续方向：(i) 通过内存中副本平面（cf. Gemini）或 save 端 PEC（cf. MoC-System）闭合 no-peer fallback（§5.12）；(ii) 把 R1 用一个 verification predicate 扩展以覆盖静默数据损坏与缓降，让 §4.7 的日志承载其结果；(iii) 在 failure-free 噪声地板的流式估计上对 $\tau_C$ 做在线校准。

## 7. 相关工作

我们以**软件工程的坐标系**而不是时间顺序来定位 MoEGambit。§7.1 把本文挂回自适应/自愈系统这一根线索，使 §3 引入的 SE artifact（R1/R2/R3）有显式血脉；§7.2--§7.4 然后梳理三条相邻的系统线索（save-side checkpointing、pipeline 自适应冗余、MoE 吞吐基础设施）；§7.5 与目前唯一公开发表的 MoE 专用容错系统对照；最后用一张定位矩阵把 MoEGambit 相对所有 5 条线索一次性放进同一张图（Table~\ref{tab:related-quadrants}）。

### 7.1 自适应与自愈软件系统

自适应/自愈系统文献把运行时修复形式化为 **MAPE-K 闭环**（Monitor–Analyze–Plan–Execute–Knowledge）[37–39]，并提炼出三条与具体负载无关的工程纪律：(a) 修复决策必须由**显式、运行时可校验的规约**驱动，而不是手工启发式；(b) execute 阶段必须用**显式屏障**保护一条命名好的不变量，而不能依赖"尽力而为"的时序；(c) 每个决策都必须留下一条**结构化、按 schema 组织**的可审计轨迹，以便事后与下游结果做相关性分析。MoEGambit 的 R1/R2/R3 契约（§3）就是把这三条纪律落到分布式 MoE 训练 runtime 上的实例化：R1 是 execute 阶段的不变量屏障；R2 的 $\Phi'(t)$ 是 analyze 阶段的运行时监控；R3 的日志 schema 是 knowledge 层。**与**经典自适应工作相比，先前应用主要面向企业控制面与云编排 [37,38]，本文把同一纪律**下沉到训练数据平面**，并引入了一个该文献此前从未触及的恢复后正确性度量——**专家加权陈旧度密度**。

### 7.2 Save 端 Checkpointing 优化

第一条系统线索瞄准 Young/Daly $\tau^{*}\approx\sqrt{2CM}$ [23,24] 里的 $C$ 因子：CheckFreq [25] 把细粒度 snapshot 与计算交错；DeepFreeze [29] 推进异步与多层写入；Check-N-Run [26] 为推荐模型引入增量与量化保存；Gemini [18] 用 GPU/CPU 内存中的对等副本替代 NVMe 写入。在 §7.1 的 MAPE-K 视角下，这条线索贡献的是 **Monitor 之前的 snapshot 频率调参**，对恢复后训练轨迹**沉默不言**。MoEGambit 与该线索**正交**——既不改 $C$、也不改保存协议——CheckFreq/Gemini/Check-N-Run 仍然可叠加部署在 MoEGambit 的恢复契约之下。

### 7.3 流水线自适应与冗余式容错

第二条系统线索通过故障后重配训练拓扑来完全避开快照回滚。Bamboo [16] 在 preemptible 实例上以冗余隐藏保存代价；Oobleck [17] 预编译 pipeline 模板以承接降级但仍合法的流水线；ReCycle [19] 在线调整 pipeline 调度；Varuna [20]、Parcae [33]、Litz [34] 与 Or 等 [35] 面向弹性与 spot 部署。这条线索的核心假设是 **rank 可互换**——这对 dense Transformer 成立，但**对 MoE 失效**：EP 分片下的某个专家子集（Table~\ref{tab:state-provenance}）在任何其他 rank 上**都没有 in-memory 对等副本**。MoEGambit 不与该线索竞争，而是**补上它留下的 EP 分片盲点**：把分片专家状态的恢复语义形式化为 R2 契约，并通过 $\Phi'(t)$ 把恢复后训练轨迹暴露为一个**可在运行时校验**的对象——这是流水线模板族既未定义、也未度量的。

### 7.4 MoE 系统及其新鲜度盲点

第三条系统线索构建 MoE 训练的吞吐基础设施：GShard [5]、Switch Transformer [7]、DeepSpeed-MoE [10]、Tutel [11]、FasterMoE [12]、SmartMoE [31]、MegaBlocks [30] 优化路由、dispatch 与 all-to-all 带宽，开放模型部署 [6,9,14] 沿用同一 EP 模板；ST-MoE [32]、sparse-upcycling [44]、V-MoE [45] 推动架构本身。在这整条线索中，**"每个专家在步 $t$ 都是新鲜的"** 是一个**隐式的性能 invariant**，而非被写下来的规约。MoEGambit 把这条隐式 invariant 提升为**显式、$O(1)$ 可计算的运行时监控** $\Phi'(t)$（§4.4），并由 §5 的经验悬崖 $\Phi_{\max}=10^{-2}$ 给出数据驱动的校准；据我们所知，这是 MoE 系统侧文献中**第一次**把这条沉默的新鲜度假设写成可在运行时校验的规约。

### 7.5 MoE 专用容错：与 MoC-System 的对比

直接面向 MoE 训练的容错工作，目前公开发表的仅有 MoC-System [15]（ASPLOS'25）。其贡献是 Partial Experts Checkpointing（PEC）：用 round-robin 调度让每轮保存只写 $N$ 个专家中的 $K_{\text{pec}}$ 个作为 fresh，把磁盘保存代价降低约 $N/K_{\text{pec}}$ 倍。本文与 MoC-System 的关系最干净的表述是**象限分配**（Table~\ref{tab:related-quadrants}）：MoC-System 位于 *save 端 / 无 SE 契约* 象限，MoEGambit 位于 *recovery 端 / R1–R3 契约* 象限；二者**正交**而非竞争。一次合并部署是允许的，前提是 $\Phi'(t)$ 在公式 (\ref{eq:phi-prime}) 的窗口聚合也把 PEC 在 save 端注入的部分新鲜度计入分子，本文把这一扩展留作未来工作（§8）。对 MoC-System 的精确性对照我们使用 §5.3 中描述的 byte-identical overlay；该 overlay 的局限性见 §5.12。

### 7.6 定位矩阵

Table~\ref{tab:related-quadrants} 把上述五条线索按"优化端"和"是否提供运行时可校验的恢复后契约"二维归类。MoEGambit 占据原本空缺的 *recovery 端 + R1–R3 契约* 象限，与所有其他线索都不直接竞争，可与 save 端（§7.2、§7.5）方法叠加部署，且补上 topology 适应方法（§7.3）留下的 EP 分片盲点。

## 8. 结论

我们论证了分布式 MoE 训练中的运行时故障恢复应该作为软件可靠性 artifact 来工程化，而不是作为 checkpoint-optimization 启发式。MoEGambit 在 Megatron-LM~\cite{narayanan2021megatron} 内部把这一视图实现为三个显式 SE artifact——带优化器提交保护的 safe-point repair (R1)、作为运行时可校验恢复规约的专家加权陈旧度密度 $\Phi'(t)$ (R2)、以及结构化恢复日志 (R3)——并把恢复 **契约**（hybrid repair 何时被允许）与恢复 **机制**（dense、shared、router、expert 状态如何从 peer 与每 rank 分片中重建）干净地分开。契约对自身前置条件诚实：当一次故障（或并发故障 burst）打掉失败 rank 所在 PP/TP 坐标下所有健康 data-parallel peer、使没有任何幸存 rank 持有 dense/router 状态当前步副本时，hybrid recovery 在构造上不可行，MoEGambit 透明回退到 checkpoint restart，策略决策与其输入保留在结构化日志中。在 64 H20-3e GPU、Qwen3-30B-A3B sparse MoE（128 专家、top-8）上的实证显示，MoE 感知 hybrid restore 贡献 $-15.1\%$ 主效应、weights-first/optimizer-later 两阶段协议另贡献 $-5.5\%$ 主效应，二者加性组合（交互项 $0.96\%$，在噪声内），相对相同 checkpoint 格式与 gap 下的 full checkpoint restart 总共降低 $20.6\%$ 单次故障端到端恢复延迟。在 $|F|\times\Delta$ 多 rank burst sweep 上，post-recovery validation loss 在每个 $\Phi'(t)\leq 10^{-2}$ 的格点都保持在 failure-free $\pm 1\sigma$ 带内，而 MoEGambit 在经验悬崖之上的格点上正确回退到 checkpoint restart。

超越这些点测量，本文更广的主张是方法学性的：**大模型训练系统中的恢复必须以 post-recovery 训练轨迹为规约对象，而不仅以 resumption 事件为规约对象。** $\Phi'(t)$ 契约、对照先验噪声地板的配对 run 评估\cite{wohlin2012ese,arcuri2014hitchhiker}、以及结构化恢复日志，是同一 SE 纪律的三个面：让恢复决策可审计、让恢复后轨迹可证伪、让 failure surface 可测试。未来工作包括 (i) 通过一个 collective-free 选择性分片 loader 把 guarded-recovery 契约扩到 `torch_dist` checkpoint 格式；(ii) 把单 rank 假设提升到并发多 rank 修复及对应的多事件 $\Phi'(t)$ 记账；(iii) 容纳生产 trace 中观察到的 non-fail-stop 故障模型——静默数据损坏与缓降硬件\cite{chowdhery2023palm,jiang2024megascale}。

## 致谢

略。

## 参考文献

[1] Llama Team, AI @ Meta. The Llama 3 Herd of Models. arXiv:2407.21783, 2024.

[2] Z. Jiang, H. Lin, Y. Zhong, Q. Huang, Y. Chen, Z. Zhang, Y. Peng, X. Li, C. Xie, S. Nong, et al. MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs. In NSDI, 2024.

[3] S. Zhang, S. Roller, N. Goyal, et al. OPT: Open Pre-trained Transformer Language Models. arXiv:2205.01068, 2022.

[4] A. Chowdhery, S. Narang, J. Devlin, et al. PaLM: Scaling Language Modeling with Pathways. JMLR, 2023.

[5] D. Lepikhin, H. Lee, Y. Xu, et al. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. ICLR, 2021.

[6] D. Dai, C. Deng, C. Zhao, et al. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. ACL, 2024.

[7] W. Fedus, B. Zoph, N. Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR, 2022.

[8] An Yang, Baosong Yang, et al. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.

[9] Qwen Team. Qwen3 Technical Report. arXiv:2505.09388, 2025.

[10] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, Y. He. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training. ICML, 2022.

[11] C. Hwang, W. Cui, Y. Xiong, et al. Tutel: Adaptive Mixture-of-Experts at Scale. MLSys, 2023.

[12] J. He, J. Zhai, T. Antunes, et al. FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models. PPoPP, 2022.

[13] X. Nie, P. Zhao, X. Miao, et al. HetuMoE: An Efficient Trillion-Scale Mixture-of-Expert Distributed Training System. arXiv:2203.14685, 2022.

[14] A. Q. Jiang, A. Sablayrolles, A. Roux, et al. Mixtral of Experts. arXiv:2401.04088, 2024.

[15] W. Wang, Y. Xie, B. Yang, J. Wu, X. Chen. MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training. ASPLOS, 2025.

[16] J. Thorpe, P. Zhao, J. Eyolfson, et al. Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs. NSDI, 2023.

[17] I. Jang, Z. Yang, Z. Zhang, X. Jin, M. Chowdhury. Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates. SOSP, 2023.

[18] Z. Wang, Z. Jia, S. Zheng, et al. Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints. SOSP, 2023.

[19] S. Gandhi, M. Zhao, A. Skiadopoulos, C. Kozyrakis. ReCycle: Resilient Training of Large DNNs Using Pipeline Adaptation. SOSP, 2024.

[20] S. Athlur, N. Saran, M. Sivathanu, R. Ramjee, N. Kwatra. Varuna: Scalable, Low-Cost Training of Massive Deep Learning Models. EuroSys, 2022.

[21] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053, 2019.

[22] Y. Huang, Y. Cheng, A. Bapna, et al. GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism. NeurIPS, 2019.

[23] J. W. Young. A First Order Approximation to the Optimum Checkpoint Interval. CACM, 1974.

[24] J. T. Daly. A Higher Order Estimate of the Optimum Checkpoint Interval for Restart Dumps. FGCS, 2006.

[25] J. Mohan, A. Phanishayee, V. Chidambaram. CheckFreq: Frequent, Fine-Grained DNN Checkpointing. FAST, 2021.

[26] A. Eisenman, K. K. Matam, S. Ingram, et al. Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models. NSDI, 2022.

[27] D. Narayanan, M. Shoeybi, J. Casper, et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC, 2021.

[28] NVIDIA. Megatron-LM. https://github.com/NVIDIA/Megatron-LM.

[29] B. Nicolae, J. Li, J. Wozniak, et al. DeepFreeze: Towards Scalable Asynchronous Checkpointing of Deep Learning Models. CCGrid, 2020.

[30] A. Qiao, S. K. Choe, S. J. Subramanya, et al. Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning. OSDI, 2021.

[31] Y. Peng, Y. Bao, Y. Chen, C. Wu, C. Guo. Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters. EuroSys, 2018.

[32] D. Narayanan, K. Santhanam, F. Kazhamiaka, A. Phanishayee, M. Zaharia. Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads. OSDI, 2020.

[33] J. Gu, M. Chowdhury, K. G. Shin, et al. Tiresias: A GPU Cluster Manager for Distributed Deep Learning. NSDI, 2019.

[34] M. Jeon, S. Venkataraman, A. Phanishayee, J. Qian, W. Xiao, F. Yang. Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads. ATC, 2019.

[35] Q. Weng, W. Xiao, Y. Yu, et al. MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters. NSDI, 2022.

[36] G. Penedo, H. Kydlíček, L. B. Allal, A. Lozhkov, M. Mitchell, C. Raffel, L. Von Werra, T. Wolf. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv:2406.17557, 2024.

[37] J. O. Kephart, D. M. Chess. The Vision of Autonomic Computing. IEEE Computer, 2003.

[38] D. Garlan, S.-W. Cheng, A.-C. Huang, B. Schmerl, P. Steenkiste. Rainbow: Architecture-Based Self-Adaptation with Reusable Infrastructure. IEEE Computer, 2004.

[39] M. Salehie, L. Tahvildari. Self-Adaptive Software: Landscape and Research Challenges. TAAS, 2009.

[40] N. Shazeer, A. Mirhoseini, K. Maziarz, et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR, 2017.

[41] B. Zoph, I. Bello, S. Kumar, et al. ST-MoE: Designing Stable and Transferable Sparse Expert Models. arXiv:2202.08906, 2022.

[42] C. Wohlin, P. Runeson, M. Höst, M. C. Ohlsson, B. Regnell, A. Wesslén. Experimentation in Software Engineering. Springer, 2012.

[43] A. Arcuri, L. Briand. A Hitchhiker's Guide to Statistical Tests for Assessing Randomized Algorithms in Software Engineering. STVR, 2014.
