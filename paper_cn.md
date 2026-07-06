# MoEGambit：面向 Mixture-of-Experts 训练的契约驱动混合恢复

## 摘要

大规模 Mixture-of-Experts（MoE）训练通常要在数千张 GPU 上持续数月，rank 故障是这类作业的常态而非偶发事件。现有恢复机制大多从全局一致 checkpoint 重启，将可复制的非专家状态和 rank 独占的专家状态一并视为一个整体。这样做会丢弃 checkpoint 之后已经完成的 GPU 工作：若要从 checkpoint 步 $c$ 回到故障步 $t$，作业必须重新加载旧状态并重放 $t-c$ 次迭代，即使许多当前状态仍然保存在健康 peer 上。稠密模型的 peer recovery 可以利用完整状态副本避免部分重放，但在专家数据并行（EDP）为 1 的 MoE 训练中，这一假设被打破：只有非专家状态可以从 peer 恢复，专家状态则是 rank 独占的。

本文提出 **MoEGambit**，一个面向恢复阶段的 MoE 框架：当所有状态都有副本时，它退化为稠密 peer recovery；当 MoE 状态呈现非专家可复制、专家不可复制的异构结构时，它进一步支持混合恢复。若运行时契约允许混合修复，MoEGambit 从健康的稠密数据并行（dense-DP）peer 拉取当前非专家状态，只从 checkpoint 加载失败 rank 的专家分片，从而消除重放，并把 checkpoint I/O 限制在没有存活 peer 的状态上。该契约由三部分组成：安全修复点控制器、用于界定部分修复适用范围的专家加权陈旧度密度 $\Phi'(t)$，以及带结构化日志的带守卫的重新接入状态机。weights-first/optimizer-later 两阶段协议进一步把优化器状态恢复与恢复后的计算重叠起来。

在 Qwen3-30B-A3B 和 DeepSeek-V2-Lite 上，MoEGambit 将原始恢复延迟降低 $20.6\%$--$55.0\%$，并取得 $36.9\times$ 的含重放端到端加速。在契约允许的单次、重复和突发故障场景中，配对评估未观察到验证损失、perplexity 或下游零样本准确率的恢复后退化。

**关键词：** 大语言模型、混合专家、容错、分布式训练、故障恢复、checkpoint、运行时监控

## 1 引言

大语言模型训练是一个跨数月、横跨数千张 GPU 的长期软件过程，故障因此是常规运行条件，而不是罕见异常。Llama 3 报告在 54 天、16,384-GPU 的预训练期间发生 419 次意外中断，平均约每三小时一次 [1]；MegaScale 记录了 10K-GPU 部署中的生产故障和落后节点 [2]；OPT-175B 的训练日志也记录了长时间人工干预 [3]。标准恢复机制 **checkpoint restart** 会重新加载最新的全局一致快照，并重放所有丢失迭代，而不区分哪些状态真正失效。这种全有或全无的回滚浪费了 checkpoint 之后已经投入的 GPU 时间；在当前故障率下，即使单次停机并不夸张，整段训练中累积的 GPU 时间损失也会相当可观。

这种重放成本越来越多地出现在稀疏 Mixture-of-Experts（MoE）训练中，例如 Mixtral [4]、DeepSeek-V2/V3 [5,6]、Qwen3-MoE [7]、Ling（百灵）[8]、Ring-flash-linear-2.0 [9] 和 Kimi K2 [10] 等模型。稠密训练提供了一个直接的优化机会：如果失败 rank 的完整状态有存活副本，替换 rank 可以从健康的数据并行 peer 重建状态。MoE 训练改变了这一恢复假设。在 $\text{EDP}=1$ 时，只有**非专家**状态（注意力、嵌入、router）以这种方式复制；专家参数和优化器矩在专家分片后是 rank 独占的。Restart 忽略了这种状态差异：它回滚整个作业，重新加载所有 checkpoint 状态，并重放每一次丢失迭代。

已有容错训练系统分别改进了恢复路径中的若干环节：降低 checkpoint 成本 [11,12,13,14,15,16]，在故障后调整拓扑 [17,18,19]，或从 peer 恢复稠密副本 [20]。但这些系统并没有处理 MoE 在恢复阶段的状态不对称性。Checkpoint/副本系统保留 checkpoint 一致的重启语义；拓扑适应系统依赖 rank 可互换；稠密 peer recovery 假设失败 rank 的完整状态存在存活 peer。对于 $\text{EDP}=1$，这一假设在专家状态上不成立：非专家状态可以从 peer 恢复，但专家状态没有存活 peer，必须来自 checkpoint。当 $\text{EDP}>1$ 时，专家副本存在，稠密式 peer recovery 也可以修复专家状态；本文关注的是先前 peer recovery 方法无法覆盖的 $\text{EDP}=1$ 操作点。这个边界很重要，因为近期 MoE 模型和生产系统倾向于使用更多专家和更宽的专家并行，以扩展模型容量并降低通信成本。在 Megatron Core 的 MoE 并行映射下，固定 GPU 预算内增加 EP 或专家张量并行（ETP）会使 $\text{EDP}=W/(\text{PP}\times\text{EP}\times\text{ETP})$ 趋近 1 [6,7,21,22]。我们所知唯一的 MoE 专用容错系统 MoC-System [23] 通过 Partial Experts Checkpointing 优化保存侧成本，但仍从 checkpoint 状态恢复。由此留下一个恢复侧空白：现有系统尚未把 MoE 可复制状态的 peer recovery 与 rank 独占专家的 checkpoint 修复结合起来，同时约束部分修复带来的质量风险。

**关键观察。** Peer recovery 在稠密训练中已经成熟，但 MoE 的异构状态使 $\text{EDP}=1$ 下的恢复只具有**部分** peer 可恢复性。非专家参数（注意力、嵌入、router）沿 dense-DP 维度复制，可以像稠密 peer recovery 一样从健康 dense-DP peer 拉取；专家参数和优化器状态则沿 EP 分片，在 $\text{EDP}=1$ 时没有存活 peer。因此，当某个 rank 失败时，替换 rank 可以从健康 dense-DP peer 拉取当前步的非专家状态，只从最新 checkpoint 加载 rank 本地专家分片——这既消除了重放，也把磁盘 I/O 限制在 rank 独占的专家体量上。当 $\text{EDP}>1$ 时，MoEGambit 沿用同一状态分类边界，但可从存活专家 peer 恢复专家状态，从而覆盖先前 peer recovery 系统所假设的更简单情形。

混合恢复带来的问题不止 I/O 成本。它用当前非专家状态搭配 checkpoint 中较旧的专家状态恢复训练，因此被恢复专家会落后模型其余部分 $\Delta=t-c$ 次迭代；若多次恢复，这种陈旧度还会在专家群体中累积。若缺少运行时可检查的安全条件，系统运维者只能在速度和模型质量之间做未经审计的权衡。借鉴自适应系统中“可监控、可审计的运行时适应”观点 [24,25,26]，MoE 恢复不应只保证进程存活，还应围绕恢复后的训练轨迹给出规范。

本文提出 **MoEGambit**，一个面向稀疏 MoE 训练恢复阶段的框架。MoEGambit 只有在专家陈旧度被界定并记录时才允许混合恢复。它引入一个显式恢复契约，包括三个构件：防止部分优化器提交的安全修复点控制器（R1），带经验校准质量阈值的专家加权陈旧度密度 $\Phi'(t)$（R2），以及带结构化日志的带守卫的重新接入状态机（R3）。在此基础上，MoEGambit 提供两种恢复机制：一条 MoE 感知的混合修复路径，从健康 dense-DP peer 拉取当前非专家参数、仅从 checkpoint 加载失败 rank 的专家分片；以及一个 weights-first/optimizer-later 两阶段恢复协议，将优化器恢复与恢复后的计算重叠。

我们在 Qwen3-30B-A3B [7]（128 个专家，64 张 H20 GPU）和 DeepSeek-V2-Lite [5]（64 个专家）上按照软件工程（SE）实验规范评估 MoEGambit。每个恢复运行都与 NoFault 基线配对，二者共享相同 seed、数据顺序、checkpoint 和注入故障步 [27,28]，从而把残余质量差异归因于恢复机制本身，而不是 seed 噪声。MoEGambit 将原始恢复延迟降低 **$20.6\%$--$55.0\%$**，并在混合路径下避免重放，实现 **$36.9\times$** 的含重放端到端加速。在契约允许的单次、重复和突发故障场景下，配对测试未显示验证损失或 perplexity 的统计可检测退化；在 50 次单故障热力图中，44/50 次运行落在 NoFault $\pm2\sigma_{\text{base}}$ 区间内，下游零样本准确率与 Restart 无法区分，并保持在 NoFault 噪声尺度内。

本文做出三个贡献：

- **运行时恢复契约。** 三个软件工程构件——安全修复点不变量（R1）、带经验校准质量阈值的专家加权陈旧度密度 $\Phi'(t)$（R2），以及带结构化审计日志的带守卫的重新接入状态机（R3）——把**何时**允许混合修复与**如何**重构状态解耦（§3）。
- **混合修复和两阶段恢复。** 一种面向恢复阶段的机制：在步 $t$ 从健康 peer 拉取 dense-DP 复制的非专家状态，只从步 $c$ 的 per-rank checkpoint 分片加载 EP 分片专家状态；同时提出 weights-first/optimizer-later 协议，将优化器恢复与恢复后训练重叠（§3.5-§3.6）。
- **配对故障注入评估。** 在 Qwen3-30B-A3B 和 DeepSeek-V2-Lite 上构建遵循 SE 实验指南 [27,28] 的可复现配对 NoFault 基线，展示 $20.6\%$--$55.0\%$ 的恢复延迟降低、$36.9\times$ 的端到端加速，以及六个 RQ 中未观察到可检测质量退化（§4）。

表 1 将 MoEGambit 与最接近的容错训练系统族进行了定位。

**表 1：与代表性容错训练系统和系统族的定位对比。**

| 代表性系统/族 | 恢复侧 | MoE感知 | 无重放 | 运行时质量守卫 | 结构化审计轨迹 |
| --- | --- | --- | --- | --- | --- |
| Checkpoint/副本系统 [11,12,13,14,16] | 否 | 部分 | 否 | 否 | 有限 |
| 拓扑适应 [17,18,19] | 是 | 否 | 是 | 否 | 有限 |
| FlashRecovery（稠密对等恢复）[20] | 是 | 否 | 是 | 否 | 有限 |
| MoC-System [23] | 否 | 是 | 否 | 否 | 否 |
| **MoEGambit** | **是** | **是** | **是** | **是** | **是** |

## 2 背景与动机

我们首先定义使部分恢复成为可能的 MoE 状态类别，然后说明 checkpoint restart 为什么没有利用这一结构。

### 2.1 概念：MoE 训练作为异构状态分布式程序

分布式 MoE 训练任务 [29,30,31,32,33,22] 是在数十到数千张 GPU 上执行集合通信的长期、有状态、多进程程序。现代 MoE 框架通常对稠密/注意力层和 MoE 层采用不同并行映射：稠密层使用张量并行（TP）、上下文并行（CP）、流水线并行（PP）和数据并行（DP），MoE 层则使用专家并行（EP）、专家张量并行（ETP）和专家数据并行（EDP）。与稠密 Transformer 不同，MoE 训练维护的是**异构状态**，其中有三类状态与恢复直接相关：

- **复制状态。** 非专家层（注意力、嵌入、层归一化）和 router 权重，会在具有相同 PP/dense-TP 坐标的 dense-DP rank 间复制。因此在任意迭代 $t$，至少有一个健康 peer 持有当前步状态。
- **分片专家状态。** 专家权重及其 AdamW 一阶/二阶矩沿 EP 和 ETP 组切分。在 $128$ 个专家、EP$=8$、ETP$=1$ 且无专家复制的布局中，每个 rank 独占 $128/8=16$ 个专家；没有 peer 持有这些专家的当前内存副本。
- **运行时元数据。** 专家目录、all-to-all 路由所需的 dispatch 拓扑 [34,35]，以及进程组视图都由 rank 布局派生；一旦 rank 被替换，这些元数据立即失效。

**专家数据并行（EDP）。** 按照 Megatron Core 当前的 MoE 术语 [22]，我们区分稠密张量并行（TP）和专家张量并行（ETP）。稠密层由 TP、CP、PP 和 dense-DP 组织，而 MoE 层由 PP、EP、ETP 和 EDP 组织。由于我们的实验未使用上下文并行，专家布局满足

$$W=\text{PP}\times\text{EP}\times\text{ETP}\times\text{EDP}, \quad \text{EDP}=\frac{W}{\text{PP}\times\text{EP}\times\text{ETP}}.$$

这不同于早期“EP 是 DP 子维度”的描述；在旧表述中，TP 有时会被放入 EDP 分母。按当前映射，稠密 TP 影响注意力层分片，但并不直接决定专家复制；ETP 才是 MoE 层的张量并行度。

当 $\text{EDP}=1$ 时，每个专家分片只存在于一个存活 rank 上；当 $\text{EDP}>1$ 时，失败专家分片可能还有存活 peer 副本，类似非专家参数的 peer 副本。$\text{EDP}=1$ 是恢复最受约束的操作点，因为每个专家分片都是 rank 独占的。在本文评估的 Qwen3-30B-A3B 布局中，$W=64$，PP$=8$，EP$=8$，ETP$=1$，因此 $\text{EDP}=64/(8\cdot8\cdot1)=1$；虽然模型共有 128 个专家，每个 EP rank 在其流水线阶段内只拥有其中 16 个。大型专家 MoE 系统 [6,7,21] 中出现这一操作点往往是资源驱动的：大量专家已经把内存分散到各 rank，而额外复制专家会放大专家权重和优化器内存，却不会增加激活计算量，因为每个 token 只激活 $K\!\ll\!E$ 个专家。这正是 MoE 恢复不同于稠密恢复的关键场景：失败 rank 的专家状态没有存活 peer，只能从 checkpoint 恢复。

图 1 说明了这种区分。

*图 1：8 GPU 下的专家分布（dense TP=1，ETP=1，PP=2）。EDP>1 时，专家有存活 peer；EDP=1 时，每个专家分片是唯一的，必须从 checkpoint 恢复。非专家参数（绿色）在两种布局中都被 dense-DP 复制。*

由此产生的软件工程后果是：**同一次故障会以不同语义使不同状态类别失效**；标准 checkpoint restart 没有利用这一边界。

### 2.2 现有流程：Checkpoint Restart 与部分恢复的隐藏风险

**动机示例。** 在我们的 64-GPU Qwen3-30B-A3B 作业中，若在步 $t\!=\!200$ 检测到故障，而最新 checkpoint 位于 $c\!=\!100$，标准 Megatron-LM 会重新加载 56 GB 分布式 checkpoint 并重放 100 次迭代，总停机约 1067 秒。然而，失败 rank 只拥有 128 个专家中的 16 个；非专家层以及另外 112 个专家仍保存在健康 rank 上。因此，MoE 感知恢复路径可以从 peer 拉取 3.2 GB 非专家状态，只加载失败 rank 的专家分片，并在 29 秒内从步 $t$ 恢复，无需重放。当然，这一机会也带来质量风险：恢复专家相对当前模型陈旧了 $\Delta=t-c$ 次迭代。

生产级 LLM 训练通常采用 **checkpoint restart** [36,1,2]：故障发生后，作业重新加载步 $c$ 的最近全局一致快照，并重放 $t-c$ 次丢失迭代。经典回滚恢复和检查点间隔分析 [37,38,39] 表明，最优检查点间隔满足 $\tau^{*}\!\approx\!\sqrt{2CM}$，其中 $C$ 是单次 checkpoint 成本，$M$ 是平均故障间隔；在 10K-GPU 集群上，$M$ 会缩短到几十分钟 [1,2]，重复故障因此会在 checkpoint I/O 和重放上消耗大量聚合 GPU 时间。

Restart 把整个训练作业当成一个全有或全无的状态对象。表 2 展示了这种做法为何浪费：单个 rank 故障只会使该 rank 的 EP 分片专家状态失效，而同一 PP/dense-TP 坐标上的其他 rank 仍持有步 $t$ 的 dense-DP 复制非专家状态。一个自然替代是**混合恢复**：在当前步从健康 dense-DP peer 同步非专家状态，只从最新 checkpoint 分片加载步 $c$ 的 rank 本地专家。不过，混合恢复也引入 restart 没有的副作用：恢复 rank 的专家状态相对于模型其余部分落后 $\Delta = t-c$ 次迭代。

**表 2：EDP=1 恢复情况下步 $t$ 的 MoE 训练状态来源。**

| 状态 | 布局 | 来源 |
| --- | --- | --- |
| 非专家（attn/embed/router） | Dense-DP | peer（步 $t$） |
| 专家权重 | EP/ETP | Checkpoint 分片（$c\!<\!t$） |
| 专家优化器 | EP/ETP | Checkpoint 分片（$c\!<\!t$） |
| 运行时元数据 | Rank 派生 | 重计算 |

这种陈旧度并不是二元错误：恢复专家仍能产生有效的前向和反向信号。但它可能扰动 MoE 辅助损失 [40,30,41,42] 所驱动的负载均衡动态。风险有两个维度：失败 rank 所拥有专家的**单次事件**间隔 $\Delta$，以及重复混合恢复在不同专家分片上累积形成的**窗口级**债务。因此，一个允许部分修复的恢复系统不能只提供快速机制，还需要给累积陈旧度设置运行时可检查边界。

**问题陈述。** MoEGambit 从集群 watchdog 接收 fail-stop 事件 $\langle r,t,c\rangle$，其中 $r$ 是失败逻辑 rank，$c$ 是最新 checkpoint，$t$ 是安全点处理后应执行的第一个步。如果故障在迭代中途被检测到，R1 会丢弃该进行中迭代，并把 $t$ 推进到下一个有效步；因此 $\Delta=t-c$ 正是 checkpoint restart 需要重放的迭代数。MoEGambit 必须在 **RESTART**（从 checkpoint 恢复所有状态并重放 $t-c$ 次迭代）和 **HYBRID**（从 dense-DP peer 恢复当前非专家状态，从失败 rank 的 checkpoint 分片恢复专家状态）之间做选择。无论选择哪条路径，恢复后的第一次 `optimizer.step()` 都必须是合法 AdamW 更新：梯度不能应用到未初始化或部分恢复的参数上。

**需求。** 我们把恢复路径形式化为包含三个软件工程需求的契约。**R1** 定义安全修复点并守护优化器提交，避免中途故障破坏模型状态。**R2** 用一个量化的 $O(1)$ 守卫给出混合恢复准入条件，同时捕获单次事件陈旧度和累积陈旧度。**R3** 记录每个决策、守卫输入、延迟分段和状态机转换，以支持事后审计。MoEGambit 假设故障为 fail-stop rank 故障，存在至少一个可用 checkpoint，失败 rank 的非专家状态有健康 dense-DP peer，且存在替换/热备 GPU；如果 peer 条件不满足，策略选择 **RESTART**。

## 3 方法

### 3.1 概览

MoEGambit 是一个运行时恢复层：它拦截 rank 故障事件，并针对每一类状态选择当前最新的可用来源来重建替换 rank，而不回滚作业的其他部分。图 2 总结了运行时架构。设计上，MoEGambit 将**需要保证什么**（恢复契约，R1-R3）与**如何修复状态**（混合恢复机制和两阶段协议）分开，使契约可以独立于具体实现接受审计。

框架在每个故障事件上执行五个组件（图 2）：

1. **故障检测。** 外部 watchdog 发出故障事件 $\langle r,t,c\rangle$，标识失败 rank、当前训练步和最新 checkpoint 步。
2. **安全修复点控制器**（§3.2，R1）。安装优化器提交守卫，确保没有 rank 会应用恢复期间计算出的梯度。
3. **陈旧度密度守卫策略**（§3.3，R2）。评估 $O(1)$ 守卫 $\Phi'(t)$，并返回确定性决策 $\pi(t)\!\in\!\{\text{HYBRID}, \text{RESTART}\}$。若预测陈旧度密度超过经验质量阈值，MoEGambit 回退到 checkpoint restart。
4. **MoE 感知混合恢复与两阶段协议**（§3.5-§3.6）。在步 $t$ 从健康 dense-DP peer 拉取非专家状态（路径 P），从步 $c$ 的 checkpoint 分片加载 rank 本地专家（路径 C），并把优化器状态恢复与恢复后计算重叠。
5. **重新接入状态机与结构化日志**（§3.7，R3）。将替换 rank 推进到 HEALTHY，并输出一条结构化记录，把策略输入、延迟和状态机转换关联起来。

### 3.2 安全修复点控制器（R1）

R1 定义一个唯一的修复程序点，并维持如下不变量：优化器提交不能使用 RECOVERING 状态下计算出的梯度。收到故障事件 $\langle r,t,c\rangle$ 后，控制器原子地将 $r$ 及其专家标记为 RECOVERING，在每个 rank 的当前迭代 `optimizer.step()` 前安装 pre-AdamW 提交守卫，并把当前迭代标记为 DISCARD。任何处于 DISCARD 状态且到达 `optimizer.step()` 的 rank 都会跳过本次更新；尚未完成的梯度被丢弃。随后训练通过混合修复（§3.5）或 restart 继续，控制器记录安全点状态。

### 3.3 陈旧度密度守卫策略（R2）

R2 定义一个 $O(1)$ 守卫 $\Phi'(t)$ 和一个确定性决策 $\pi(t)\!\in\!\{\text{HYBRID}, \text{RESTART}\}$。该决策使用故障事件 $\langle r,t,c\rangle$、窗口陈旧债务 $S(t)=\sum_{h\in\mathcal{H}_{W_{\mathrm{exp}}}(t)}|E_h|\Delta_h$、谓词 $\text{PeerAvail}(r,t)$，以及在 §4 中校准的阈值 $(\Delta_{\min},\Delta_{\max},\Phi_{\max})$。其中，$W_{\mathrm{exp}}$ 是暴露窗口长度，$\mathcal{H}_{W_{\mathrm{exp}}}(t)$ 是 $[t\!-\!W_{\mathrm{exp}},t)$ 内先前混合事件集合，$E_h$ 是事件 $h$ 恢复的专家集合，$\Delta_h$ 是其 checkpoint 间隔，$N_{\text{expert}}$ 是暴露域中的路由专家数；因此 $S(t)$ 的单位是专家-迭代。如果当前事件走混合路径，则预测密度为

$$\Phi'(t) = \frac{S(t) + |E_{\text{new}}| \cdot \Delta}{N_{\text{expert}} \cdot W_{\mathrm{exp}}}, \tag{1}$$

其中 $N_{\text{expert}}\!\cdot\!W_{\mathrm{exp}}$ 表示“所有专家暴露一个完整窗口”所对应的债务；因此，$\Phi'(t)=0.1$ 表示预测暴露量相当于该完整窗口预算的 10%。该值是暴露密度，不是概率；如果窗口内反复暴露，它可以超过 1。运行时只需维护一个标量 $S(t)$：混合事件 $h$ 进入窗口时加上 $|E_h|\Delta_h$，离开窗口（超过 $W_{\mathrm{exp}}$ 次迭代）时减去同一项，因此每次迭代只需 $O(1)$ 记账。若多个 rank 同时失败，每个失败 rank 都会先把自己的 $|E_h|\Delta_h$ 项加入 $S(t)$，然后再评估下一个 rank 的策略。

**直觉。** 这个守卫是一个可监控的暴露预算，而不是全局收敛保证。一次混合事件会让 $|E_h|$ 个专家使用相对模型其余部分落后 $\Delta_h$ 次迭代的权重；重复事件会让更多专家、在更长时间内处于这种暴露状态。用 $|E_h|\Delta_h$ 给每个事件计费，正好同时刻画“影响多少专家”和“落后多久”两个因素；再用 $N_{\text{expert}}\!\cdot\!W_{\mathrm{exp}}$ 归一化，使 $\Phi'(t)$ 可以跨模型规模和窗口长度比较。阈值 $\Phi_{\max}\!=\!10^{-1}$（§4）相对于观察到的无故障噪声水平进行校准。

决策为

$$\pi(t) = \begin{cases}
\text{HYBRID}, & \text{if } \text{PeerAvail}(r,t) \wedge \Delta_{\min} \leq \Delta \leq \Delta_{\max} \wedge \Phi'(t) \leq \Phi_{\max}, \\
\text{RESTART}, & \text{otherwise}.
\end{cases} \tag{2}$$

算法 1 给出同一逻辑，并在每次回退时记录第一个失败的条件。三个阈值作用不同。$\Delta_{\min}$ 是成本收益下界，用来避免在重放间隔太小、无法抵消协调成本时使用混合修复；$\Delta_{\max}$ 是单次事件中任一恢复专家分片允许的最大陈旧度；$\Phi_{\max}$ 则是累积专家暴露的窗口级上界。每个阈值都对应可测量来源：$\Delta_{\min}$ 对应 $T_{\text{load}}\!:\!T_{\text{hybrid}}\!:\!T_{\text{iter}}$ 比率，$\Delta_{\max}$ 对应 Young/Daly [38,39] 限定的检查点间隔，$\Phi_{\max}$ 对应禁用守卫后的质量扫描，$\text{PeerAvail}$ 对应存活进程组。日志条目包含 $\langle\pi(t),\text{reason}\rangle$ 和 $(\Delta,|E_{\text{new}}|,S(t),\Phi'(t))$（§3.7）；紧凑的原因标签分别表示 peer 不可用、两个间隔条件失败、暴露条件失败，以及混合路径被允许。

**算法 1：** 带守卫的恢复决策 $\pi(t)$。

输入：事件 $\langle r,t,c\rangle$；债务 $S(t)$；常数 $N_{\text{expert}}, W_{\mathrm{exp}}$
输入：守卫 $(\Delta_{\min},\Delta_{\max},\Phi_{\max})$；PeerAvail
输出：恢复决策和首个失败原因

1. $\Delta \gets t-c$；$e \gets |E_{\text{new}}(r)|$
2. $\rho \gets (S(t)+e\Delta)/(N_{\text{expert}}W_{\mathrm{exp}})$
3. 如果 $\neg\,\text{PeerAvail}(r,t)$：返回 (RESTART, NoPeer)
4. 否则如果 $\Delta < \Delta_{\min}$：返回 (RESTART, SmallGap)
5. 否则如果 $\Delta > \Delta_{\max}$：返回 (RESTART, LargeGap)
6. 否则如果 $\rho > \Phi_{\max}$：返回 (RESTART, HighDebt)
7. 否则：返回 (HYBRID, Admit)

### 3.4 守卫的理论依据

恢复契约给出了 MoEGambit 选择快速路径时必须满足的可监控安全条件。对于每个恢复事件，若选择混合恢复，则三个条件都必须成立：

$$\pi(t)=\text{HYBRID} \Rightarrow \text{PeerAvail}(r,t) \wedge \Delta_{\min}\leq\Delta\leq\Delta_{\max} \wedge \Phi'(t)\leq\Phi_{\max}, \quad \forall t.$$

R3 会记录每个条件成立的证据，因此该契约既可以在线检查，也可以离线审计。$\Phi_{\max}$ 是经验阈值，但 $\Phi'(t)$ 的形式来自一阶暴露边界。

**命题 1（专家暴露边界）。** 设 $\theta_e(s)$ 表示步 $s$ 之后专家 $e$ 的状态。假设在恢复窗口内、非专家状态固定时，被监控的 MoE 损失分量 $\ell_{\text{aux}}$ 关于每个专家状态是按坐标 $L_{\text{aux}}$-Lipschitz 的，并且正常训练步在同一范数下对任一专家的改变量至多为 $G$。若事件 $h$ 从 checkpoint 步 $c_h$ 恢复专家集合 $E_h$，并在 $t_h$ 恢复训练，则相对于使用当前状态恢复，其陈旧专家扰动满足

$$\delta\ell_h \leq L_{\text{aux}}\sum_{e\in E_h}\|\theta_e(t_h)-\theta_e(c_h)\| \leq L_{\text{aux}}G\,|E_h|\,(t_h-c_h).$$

因此，把当前候选事件加入已有窗口债务后，

$$\sum_{h\in\mathcal{H}_{W_{\mathrm{exp}}}(t)\cup\{\text{new}\}}\delta\ell_h \leq L_{\text{aux}}G\,(S(t)+|E_{\text{new}}|\Delta)=L_{\text{aux}}G\,N_{\text{expert}}W_{\mathrm{exp}}\Phi'(t).$$

由此可见，$\Phi'(t)$ 是一个归一化暴露代理；它诱导的一阶扰动边界随 $\Phi'(t)$ 线性变化，但它本身不是独立的收敛定理。

### 3.5 MoE 感知混合状态恢复

当 $\pi(t)=\text{HYBRID}$ 时，替换 rank 会按状态类别从最新可用来源重建状态（表 2），因此 I/O 与 rank 独占专家体量成比例，而不是与完整分布式 checkpoint 成比例。两条路径并行执行。**路径 P（peer 拉取，复制状态）** 通过 NCCL/Gloo 上的一次 P2P 广播，从同一 PP/dense-TP 坐标的健康 peer 同步当前步非专家参数（以及 dense-DP 复制的优化器状态）；传输通常在数十毫秒内完成。**路径 C（分片读取，分片专家状态）** 从步 $c$ 的 per-rank 分片加载 rank-$r$ 的专家，不需要集合通信。恢复后的 rank 是有意混合的：非专家状态来自步 $t$，专家及其优化器状态则陈旧 $\Delta$，即式 (1) 中由 $\Phi'(t)$ 约束的量。

### 3.6 两阶段恢复协议

两阶段协议通过把路径 C 中优化器状态的读取与恢复后的前向/反向计算重叠，缩短恢复时间，同时禁止在未初始化的专家槽上提交更新。**阶段 A（权重优先）** 恢复非专家状态和专家权重，然后在 **更新屏障** 下重新接入训练：前向/反向可以贡献到全局损失，但受影响专家的梯度会被缓冲。**阶段 B（优化器延后）** 并行恢复非专家和专家矩；恢复完成后释放屏障，并通过一次 AdamW 步应用缓冲梯度。该屏障复用 §3.2 的 pre-`step()` 钩子，因此 R1 在整个过程中成立。日志记录 $T_{\text{TTR}}$ 和 $T_{\text{TTFR}}$；二者差值对应相对于单阶段 restart 节省的延迟。由于该协议控制的是优化器状态**何时**挂载，而不是字节**从哪里**读取，§4.4 的 $2{\times}2$ 析因实验可以独立于混合恢复对其进行测试。

### 3.7 重新接入状态机与结构化日志（R3）

R3 要求每个恢复决策及其后续模型质量结果都可以追溯。我们通过两个机制实现 R3：一个强制执行重新接入协议的守卫状态机，以及一个记录每个恢复事件输入与输出的结构化日志。

**守卫状态机。** 每个替换 rank 按四状态线性协议推进：

$$\text{RECOVERING} \xrightarrow{g_1} \text{REPAIRED} \xrightarrow{g_2} \text{BARRIER} \xrightarrow{g_3} \text{HEALTHY},$$

其中每个状态转换都由守卫谓词控制：

- $g_1$：状态恢复完成——所有状态类别（非专家经路径 P，专家经路径 C）均已加载，并通过来源校验和验证。
- $g_2$：dispatch 拓扑重新推导——专家目录和 all-to-all 路由表已重算，以反映替换 rank 在 EP 布局中的位置。
- $g_3$：优化器状态已挂载且更新屏障已释放——两阶段协议的阶段 B（§3.6）已完成，缓冲梯度已应用。

由于每个转换都必须满足对应守卫，状态机可以拒绝过早接入，例如避免把 token 路由到优化器状态尚未恢复的专家。

**结构化恢复日志。** 每个恢复事件都会生成一条记录，包含策略输入（$\Delta$、$|E_{\text{new}}|$、$S(t)$、$\Phi'(t)$）、决策和原因字符串、各阶段延迟以及状态转换时间戳。该记录把每条恢复路径关联回式 (2) 中的守卫条件，并支持在 §4 的配对运行方法下，将恢复行为与训练质量指标做事后关联 [27,28]。$\Phi'(t)$ 记账和日志输出的每迭代开销为 $O(1)$，实验中可忽略（§4.7）。

### 3.8 实现

MoEGambit 以 Megatron-LM（commit `core_r0.9.0`）的 Python 扩展形式实现，覆盖安全点控制、策略评估、混合恢复、两阶段优化器恢复和重新接入。它挂接优化器步、checkpoint 加载以及广播/集合通信路径，同时保留 Megatron 的标准 checkpoint restart 路径；不使用 `--enable-moegambit` 时，训练遵循标准路径。我们将在发表后开源实现和评估脚本。

## 4 评估

本节评估 MoEGambit 是否能在不损害恢复后训练轨迹的前提下降低恢复成本。我们围绕六个研究问题展开：

- **RQ1：** MoEGambit 能降低多少恢复成本？混合恢复和两阶段恢复是否近似加性组合？
- **RQ2：** 契约允许的混合恢复是否能避免相对于无故障训练的可检测质量退化？
- **RQ3：** $\Phi'(t)$ 是否在重复和突发故障下识别质量退化？
- **RQ4：** 监控钩子在无故障训练期间增加了什么开销？
- **RQ5：** 恢复优势是否能跨 GPU 规模和并行布局保持？
- **RQ6：** 该机制是否泛化到不同的 MoE 模型配置？

### 4.1 实验设置

**集群和并行。** 除非另有说明，实验运行在 64 张 H20-3e GPU（8 节点 × 8 GPU）上，默认配置为 dense TP=1，PP=8，EP=8，ETP=1，EDP=1（按 Megatron Core 定义）。可扩展性和并行敏感性实验采用 RQ5 中说明的覆盖配置。

**模型和分词器。** 我们使用 Qwen3-30B-A3B [7]（30B 总参数 / 3B 激活参数，48 层，128 个专家，top-8 路由）和 Qwen2Tokenizer（151,936-token 字节级 BPE）。默认训练超参数见表 3。

**表 3：默认训练配置。**

| 参数 | 值 | 参数 | 值 |
| --- | --- | --- | --- |
| 层数 | 48 | 专家数 | 128 |
| 隐藏大小 | 2048 | MoE top-$k$ | 8 |
| 注意力头 | 32 | MoE FFN 隐藏 | 768 |
| Query 组 | 4 | 负载均衡 | aux. loss |
| 序列长度 | 4096 | Aux. 系数 | $1 \times 10^{-3}$ |
| 全局 batch | 64 | 优化器 | AdamW |
| 精度 | BF16 | LR/min LR | $10^{-4}/10^{-5}$ |
| LR 调度 | cosine | 权重衰减 | 0.1 |
| 梯度裁剪 | 1.0 | | |

**数据。** 我们使用 FineWeb [43] 的 4B-token 子集，并处理为 Megatron 索引格式。验证损失和 perplexity 在保留验证集上每 1,000 次迭代报告一次。所有恢复实验均采用**配对运行**：基线和 MoEGambit 共享相同 seed、数据顺序、checkpoint 和注入故障步。

**基线计时和噪声尺度。** 稳态 $T_{\text{iter}}\!\approx\!10.31$ s；完整 checkpoint 加载 $T_{\text{load}}\!\approx\!35.95$ s；混合恢复 $T_{\text{hybrid}}\!\approx\!28.91$ s；`save-interval`=200（Young/Daly 最优 [38,39]）。10 次 NoFault 运行在 iter-600 评估损失上得到 $\mu_{\text{base}}=4.8543$、$\sigma_{\text{base}}=0.024$。我们用 $\sigma_{\text{base}}$ 缩放偏差，并报告 $\pm1\sigma_{\text{base}}$ 和 $\pm2\sigma_{\text{base}}$ 区间。该噪声尺度覆盖了即使固定 seed 和数据顺序后仍会存在的残余变化，包括集合通信时序、节点/GPU 异构性、共享存储抖动以及非确定性内核调度。

**指标。** 原始恢复延迟不包含重放，在替换 rank 到达 HEALTHY 时结束；恢复时间（$T_{\text{TTR}}$）在第一次有效恢复后前向传播时结束；完全恢复时间（$T_{\text{TTFR}}$）在优化器状态挂载且更新屏障释放时结束。端到端恢复时间还计入 Restart 和 MoC-System 的重放成本；MoEGambit 的混合路径直接在步 $t$ 恢复，因此重放成本为零。

**统计处理。** 延迟实验报告 500 次注入恢复事件的均值，因为恢复成本主要由确定性 I/O 和集合传输阶段决定；质量实验使用配对比较控制 seed 和数据顺序噪声。我们以 NoFault 的 $\sigma_{\text{base}}$ 区间作为实际效应大小参考，并使用配对非参数检验评估退化；我们不要求每一次单独恢复运行都落在 $\pm1\sigma_{\text{base}}$ 区间内。

### 4.2 对比系统

我们将 MoEGambit 与三个基线对比：**NoFault**（不注入故障，作为参考轨迹）；**Restart**（标准 Megatron-LM checkpoint restart，包含完整重载和重放）；**MoC-System**（我们所知唯一已经发表的 MoE 专用容错系统 [23]）。

**MoC-System。** MoC-System 是保存侧 MoE 容错系统，通过 Partial Experts Checkpointing（PEC）降低 checkpoint 成本：每次 checkpoint 只保存选定专家子集，并在多次保存之间轮换该子集。由于没有公开参考实现，我们分别处理准确性和计时。准确性方面，我们将每个非新鲜专家重定向到对应的历史 checkpoint 分片，以复现 MoC-System 论文报告的最佳 PEC 配置（$K_{\text{pec}}=16$，$N=128$，PLT≈3.75%），并禁用 MoEGambit 的混合修复和两阶段协议。计时方面，我们使用论文报告或可由论文推导出的最佳恢复数字；由于 MoC-System 作用在保存侧，恢复仍从 checkpoint 状态恢复，并需要重放丢失迭代才能到达步 $t$。

### 4.3 策略参数

MoEGambit 使用三个阈值：$\Delta_{\min}$、$\Delta_{\max}$、$\Phi_{\max}$（表 4）。

**表 4：默认策略参数。**

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| $\Delta_{\min}$ | 混合恢复的最小间隔 | 1 迭代 |
| $\Delta_{\max}$ | 混合恢复的每次事件最大间隔 | 200 迭代 |
| $\Phi_{\max}$ | $W_{\mathrm{exp}}$ 中的最大陈旧度密度 | $1\!\times\!10^{-1}$ |
| $W_{\mathrm{exp}}$ | 暴露窗口 | 2,000 迭代 |
| $N_{\text{expert}}$ | 暴露域中的路由专家数 | 128 |

$\Delta_{\min}$ 和 $\Delta_{\max}$ 在我们的计时条件下并不触发（$T_{\text{load}}>1.2\,T_{\text{hybrid}}$，因此对每个 $\Delta\!\geq\!1$，混合恢复延迟都更低）；保留这两个阈值是为了防御 checkpoint 加载快得多的集群场景。

**$\Phi_{\max}=10^{-1}$ 校准。** 我们通过一次禁用守卫的 $3\!\times\!4$ 多 rank 突发扫描（表 5）校准该阈值，用来识别默认策略需要规避的质量退化边界。$W_{\mathrm{exp}}=2{,}000$ 覆盖稳定性实验中的 1,000 迭代预热和 1,000 迭代恢复后区间。在这一窗口下，$\Phi'(t)\!\leq\!10^{-1}$ 的七个格点均值保持在基线 $\pm 1\sigma_{\text{base}}$ 区间内；而阈值以上的五个格点，在始终使用混合恢复时偏离 $1.21$--$1.98\sigma_{\text{base}}$。因此，默认策略会把这五个超阈值格点路由到 checkpoint restart。

**表 5：守卫禁用突发扫描：始终混合恢复下的验证损失偏差（$\sigma_{\text{base}}$ 单位）。阴影格点超过 $\Phi_{\max}$ 并在默认策略下触发 restart。**

| $|F|$ | $\Delta\!=\!50$ | $\Delta\!=\!100$ | $\Delta\!=\!150$ | $\Delta\!=\!200$ |
| --- | --- | --- | --- | --- |
| $8$ | $+0.32$ | $+0.64$ | $-0.05$ | $+0.01$ |
| $16$ | $-0.18$ | $+0.71$ | **$+1.42$** | **$+1.46$** |
| $24$ | $+0.59$ | **$+1.21$** | **$+1.98$** | **$+1.95$** |

### 4.4 RQ1：单次故障恢复成本与机制分解

RQ1 衡量低延迟恢复路径的收益：MoEGambit 能节省多少延迟，以及两个机制是否可以近似独立组合。

我们在预热后注入硬故障，并运行 $2\times 2$ 析因设计，交叉比较 **(i)** 混合恢复与完整 checkpoint restart，以及 **(ii)** 两阶段与单阶段优化器挂载。每个格点聚合 500 次注入恢复事件；表 6 报告均值。延迟从故障检测开始计时，到替换 rank 产出第一次恢复后迭代为止。

**表 6：单次故障恢复时间（$2{\times}2$ 析因；每事件均值，秒，排除重放）。**

| 恢复路径 | 单阶段 | 两阶段 | $\Delta$（s） |
| --- | ---: | ---: | ---: |
| 完整 checkpoint | 36.417 | 34.238 | $-2.179$ |
| 混合（选择性） | 30.743 | **28.914** | $-1.829$ |
| $\Delta$ hybrid（s） | $-5.674$ | $-5.324$ | |

**分解。** 混合恢复主效应为 $-5.50$ s（$-15.1\%$），两阶段主效应为 $-2.00$ s（$-5.5\%$），交互项为 $0.35$ s（基线的 $0.96\%$，处于噪声范围内）。在该实验中，两个机制近似加性，联合得到最佳格点 $28.914$ s，相比完整 checkpoint restart 降低 $20.6\%$。

**结果。** 混合恢复把 NVMe 绑定的全量张量恢复替换为“peer 拉取非专家状态 + 一次专家分片读取”，将 I/O 从 $O(\text{全局 checkpoint})$ 降到 $O(|E_{\text{new}}|\!\cdot\!\text{shard})$。两阶段恢复把优化器状态恢复与恢复后的前向/反向计算重叠；在我们的测量中，这种重叠节省了近似常数的约 2 s。较小的交互项（$0.96\%$）说明两个机制主要作用于不同恢复阶段。

由于 Restart 和 MoC-System 都从 checkpoint 状态恢复，它们必须重放期望间隔 $\Delta\!=\!100$ 次迭代才能到达步 $t$（$T_{\text{replay}}\!\approx\!1031$ s）。按上述最佳加载成本计算，含重放成本为 $1067.4$ s；MoEGambit 则直接在步 $t$ 恢复，含重放端到端加速为 $1067.4/28.9=\mathbf{36.9\times}$。全文中我们分别报告原始延迟和含重放端到端时间。

### 4.5 RQ2：训练质量与稳定性

RQ2 检验恢复契约的质量侧：契约允许的混合恢复是否能避免相对于无故障训练的可检测退化？

**设置。** 每次运行先执行 1,000 次预热迭代，随后注入一次故障，并与配对的 NoFault 和 Restart 运行继续训练 1,000 次迭代。满足契约的间隔（$\Delta\!\in\!\{64,128\}$）走混合路径；更大的强制间隔 $\Delta\!\in\!\{256,512,1000,1500\}$ 用于压力测试单次事件守卫，默认会路由到 restart，除非我们故意禁用守卫。

**结果。** 对于满足契约的混合间隔，配对验证损失和 perplexity 相比 NoFault/Restart 未出现可检测退化。图 3 给出含 10 次注入故障的 10,000 迭代轨迹，图 4 展示满足契约的 50 次单故障运行相对 NoFault 均值的热力图：44/50 落在 $\pm2\sigma_{\text{base}}$ 内，23/50 落在 $\pm1\sigma_{\text{base}}$ 内。对 iter 600 的 10 次配对评估损失差异进行 Wilcoxon 符号秩检验，得到 $p\!=\!0.63$ 和 Cliff's delta $=0.07$，没有退化证据。更大的强制间隔仅用于守卫压力测试；默认策略下这些情况会路由到 restart。

由于运行按 seed、数据顺序、checkpoint 和故障时间配对，这一判据是保守的：我们按相对基线轨迹的配对偏差评价 MoEGambit，而不是看某一次噪声运行是否恰好提升了分数。

热力图关注孤立 rank/GPU 故障，这是 LLM 训练轨迹和弹性训练系统反复强调的常见生产场景 [1,2,18]。当作业规模扩大、EP 增加或专家切分更细时，单张故障 GPU 通常只拥有更小比例的专家，因此 $\Phi'(t)$ 对单次事件计入的暴露更少；RQ3 进一步覆盖了会累积暴露并触发回退的突发故障。

在所有 10,000 次迭代和注入故障中，MoEGambit 始终在运行间噪声范围内跟随 Restart；MoC-System 则积累了一个小而持续的间隔，这与 PEC 在恢复边界携带更旧专家状态的机制一致。

**下游零样本准确率。** 表 7 报告了 iter 10,000 在八个任务上的 lm-evaluation-harness [44] 结果（ARC-Easy [45]、BoolQ [46]、MathQA [47]、OpenBookQA [48]、PIQA [49]、RACE [50]、SWAG [51]、WinoGrande [52]）。

**表 7：iter $10{,}000$ 的零样本下游准确率（%；lm-evaluation-harness；标准误 ≤ 0.021）。**

| 系统 | ARC-E | BoolQ | MathQA | OBQA | PIQA | RACE | SWAG | WG | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Restart (Megatron) | 46.72 | 59.88 | 22.08 | 29.00 | **68.34** | 29.00 | 55.17 | 50.28 | 45.06 |
| MoC-System [23] | 45.41 | **59.94** | **22.28** | 28.20 | 67.68 | 29.19 | 54.00 | **50.67** | 44.67 |
| MoEGambit | **48.36** | 58.56 | 21.54 | **30.60** | 68.17 | **29.28** | **55.43** | 50.59 | **45.32** |

Restart 和 MoEGambit 在每个任务上的差距都在 $\pm 1.6$ pp 内。MoEGambit 的平均值数值上略高（$+0.26$ pp），但差距仍处于每个 checkpoint 的下游评估噪声范围 [53] 内；因此我们将其解释为质量对等，而不是精度提升。这个小的正均值可能来自评估噪声，也可能来自有界扰动对少量专家子集带来的轻微 dropout 式噪声正则化 [54,55]。这一解释也与 ST-MoE 的微调消融一致：只更新非 MoE 参数也能接近更新所有参数的效果，而稀疏模型可以容忍一定的 dropped-token 噪声 [41]。MoC-System 达到 $44.67\%$（比 Restart 低 $0.39$ pp），八个探测中有六个落后，与图 3 中持续存在的损失间隔一致。

### 4.6 RQ3：多次故障与专家加权陈旧度密度

RQ3 验证 R2 规范：$\Phi'(t)\!\leq\!\Phi_{\max}$ 守卫能否在多 rank 突发故障下识别质量退化边界？

我们评估 MoEGambit 在重复故障和多 rank 突发故障下的表现，并验证 §4.3 中的 $\Phi_{\max}=10^{-1}$ 阈值。使用 `find_multi_fault.sh`，我们在与表 5 相同的 $3\!\times\!4$ 网格上扫描 $|F|\!\in\!\{8,16,24\}$ 个突发故障以及 $\Delta\!\in\!\{50,100,150,200\}$，并加入同一 rank 在 $W_{\mathrm{exp}}$ 内重复出现的窗口轨迹。仅看间隔或始终混合的策略会把这些模式视为同类；$\Phi'(t)$ 则通过 $|E_{\text{new}}|$ 和累积窗口债务把它们区分开。

**结果。** 默认策略下，MoEGambit 对 7 个 $\Phi'(t)\!\leq\!10^{-1}$ 的格点使用混合恢复，并将 5 个超阈值格点重定向到 checkpoint restart，使格点均值损失回到 $\pm 1\sigma_{\text{base}}$ 区间内。当禁用守卫时，同样 5 个格点复现表 5 中的 $1.21$--$1.98\sigma_{\text{base}}$ 偏差。因此，超阈值场景下没有出现退化是因为策略回退起作用；专家加权窗口捕获了仅看间隔的策略会遗漏的风险。

这一结果也解释了为什么单独使用 $\Delta_{\max}$ 不够。两个 checkpoint 间隔相同的事件，可能暴露截然不同的专家比例，具体取决于哪些 rank 故障以及其他专家最近何时被修复。$\Phi'(t)$ 按间隔和受影响专家集合比例为每个事件计费，并让该计费随时间移出窗口，从而显式刻画暴露。即使单次事件间隔相同，该守卫也能区分孤立单 rank 故障和密集突发故障。

### 4.7 RQ4：无故障开销

**结果。** 在 1,000 次无故障迭代中，MoEGambit 的钩子每次迭代增加 $<6$ μs，平均步时间仅增加 $+0.1\%$；配对 $t$ 检验在 $\alpha=0.05$ 下未拒绝零均值差异。

### 4.8 RQ5：可扩展性

我们在 64 和 128 GPU 上重跑 §4.4 的单故障注入实验（dense TP=1，PP=8，EP∈{8,16}，ETP=1；每组 10 个 seed），并测试四种 64-GPU 布局，覆盖 dense TP∈{1,2}、ETP∈{1,2}、PP∈{4,8}、EP∈{4,8}、EDP∈{1,2}。**结果。** Restart 延迟随全局 checkpoint 大小增长（$36.4$ s → $47.2$ s），而 MoEGambit 增长更慢（$28.9$ s → $33.7$ s），因为混合 I/O 与 per-rank 分片大小成比例。二者比率从 $1.26\times$ 扩大到 $\mathbf{1.40\times}$，说明 GPU 规模继续扩大时收益预计会更高。

**并行敏感性。** 在四种 64-GPU 布局上，MoEGambit 相比 restart 提升 $1.26$--$3.56\times$。当 EDP≥2 时优势最大，因为存活专家 peer 允许 MoEGambit 同时从 peer 恢复非专家和专家状态；即使在 EDP=1，混合恢复仍然只需从磁盘读取失败 rank 的专家分片，因此仍快于 restart。

### 4.9 RQ6：跨模型泛化

为评估 Qwen3-30B-A3B 之外的泛化性，我们在 **DeepSeek-V2-Lite** [5]（15.7B 总参数 / 2.4B 激活参数，64 个路由专家，2 个共享专家，top-6）上重跑核心实验。共享专家是 dense-DP 复制的，走路径 P；路由专家走路径 C。策略参数沿用表 4，$N_{\text{expert}}\!=\!64$。

**结果。** Restart 耗时 $26.09$ s，MoEGambit 恢复耗时 $11.73$ s，降低 $55.0\%$（$2.22\times$）。更小的专家分片使路径 C 更快；突发扫描显示相同的 $\Phi'(t)=10^{-1}$ 阈值仍适用，而将共享专家也路由到路径 C 会使收益下降 $3.2$ pp。

### 4.10 有效性威胁

遵循标准 SE 有效性分类 [27]，我们总结主要威胁。**内部有效性。** 由于没有公开的 MoC-System 实现，我们使用其报告的最佳 PEC 配置和计时数字，分别报告原始延迟与含重放加速，并在测量 MoEGambit 机制时禁用 PEC。配对 seed、相同数据/故障步以及 NoFault $\sigma_{\text{base}}$ 区间控制了集合通信、网络、异构性、存储和调度带来的残余噪声。**构造有效性。** 我们测量损失、perplexity 和八个零样本任务；其他应用可能需要额外探测。**外部有效性。** 结果覆盖 H20-3e 互联上的两个 Megatron-LM MoE 配置；其他布局、网络架构、checkpoint 后端、检测器或辅助损失设置可能需要重新校准 $\Phi_{\max}$。

## 5 相关工作

**自适应与自愈系统。** MAPE-K 模型 [24]、架构自适应 [25] 以及 SE 路线图 [26,56,57] 都强调运行时修复应当有规范、有守卫、可审计。MoEGambit 将这一视角应用到 MoE 恢复：$\Phi'(t)$ 决定何时适应，状态类别边界决定在哪里适应，混合恢复则定义如何适应。

**检查点、拓扑适应和 peer recovery。** Checkpoint 系统降低 Young/Daly 式 restart 中的成本 $C$ [38,39]，代表系统包括 CheckFreq [11]、DeepFreeze [12]、Check-N-Run [13]、Gemini [14]、REFT [15] 和 ByteCheckpoint [16]。这些系统保留 checkpoint 一致恢复语义。Bamboo [17]、Oobleck [18]、ReCycle [19]、Varuna [58]、Parcae [59] 和 Litz [60] 等拓扑适应系统通过围绕存活 rank 重新配置来避免重载，但它们假设 rank 可互换。FlashRecovery [20] 从 peer 拉取稠密 DP 副本；MoEGambit 则把 peer recovery 应用于 dense-DP 复制的非专家状态，并在 EDP=1 时用守卫条件约束陈旧的 EP 分片专家。

**MoE 系统和容错。** GShard [29]、Switch [30]、DeepSpeed-MoE [31]、Tutel [32]、MegaBlocks [33]、FasterMoE [34]、SmartMoE [35] 以及开放 MoE 模型 [4,61,6,7,8,9,10] 主要优化路由、dispatch 和 all-to-all 效率；ST-MoE [41]、sparse-upcycling [42] 和 V-MoE [62] 则推进 MoE 架构本身。这些工作通常把专家状态新鲜度作为隐式假设。MoC-System [23] 通过部分专家 checkpoint 降低保存侧成本，但仍从 checkpoint 状态恢复；MoEGambit 关注恢复侧的重放成本、加载成本和运行时质量守卫。

## 6 讨论

**意义和部署。** 可靠的 MoE 恢复应当是状态感知且可审计的。R1-R3 暴露策略谓词、债务 $S(t)$、回退原因和质量轨迹，可支持仅审计部署，也可用于校准混合恢复。保存侧 checkpoint 优化仍然与 MoEGambit 互补，因为它们可以缩小 $\Delta$。

**局限性和未来工作。** MoEGambit 要求失败 rank 的 PP/dense-TP 坐标上存在健康 dense-DP peer；关联故障会回退到 restart。本文评估的 $\Phi_{\max}=10^{-1}$ 阈值假设辅助损失尺度相同、专家放置静态且故障为 fail-stop。动态迁移 [34]、慢性退化、静默数据损坏 [63] 和网络分区都需要额外守卫；未来工作将加入专家使用统计和在线噪声水平估计。

## 7 结论

MoEGambit 使分布式 MoE 恢复具备状态感知和可审计性：其运行时契约约束混合修复，将原始恢复延迟降低 $20.6\%$--$55.0\%$，实现 $36.9\times$ 的含重放加速，并在契约允许的故障下未观察到可检测质量退化。

## 参考文献

[1] A. Dubey et al., "The Llama 3 herd of models," arXiv:2407.21783, 2024.

[2] Z. Jiang et al., "MegaScale: Scaling large language model training to more than 10,000 GPUs," in Proc. USENIX NSDI, 2024, pp. 745–760.

[3] S. Zhang et al., "OPT: Open pre-trained transformer language models," arXiv:2205.01068, 2022.

[4] A. Q. Jiang et al., "Mixtral of experts," arXiv:2401.04088, 2024.

[5] DeepSeek-AI, "DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model," arXiv:2405.04434, 2024.

[6] DeepSeek-AI, "DeepSeek-V3 technical report," arXiv:2412.19437, 2024.

[7] Qwen Team, "Qwen3 technical report," arXiv:2505.09388, 2025.

[8] Ling Team, B. Zeng et al., "Every FLOP counts: Scaling a 300B mixture-of-experts LING LLM without premium GPUs," arXiv:2503.05139, 2025.

[9] Ling Team, B. Han et al., "Every attention matters: An efficient hybrid architecture for long-context reasoning," arXiv:2510.19338, 2025.

[10] Kimi Team, Y. Bai et al., "Kimi K2: Open agentic intelligence," arXiv:2507.20534, 2025.

[11] J. Mohan, A. Phanishayee, and V. Chidambaram, "CheckFreq: Frequent, fine-grained DNN checkpointing," in Proc. USENIX FAST, 2021, pp. 203–216.

[12] B. Nicolae, J. Li, J. Wozniak, G. Bosilca, M. Dorier, and F. Cappello, "DeepFreeze: Towards scalable asynchronous checkpointing of deep learning models," in Proc. IEEE/ACM CCGrid, 2020, pp. 172–181.

[13] A. Eisenman, K. K. Matam, S. Ingram, D. Mudigere, R. Krishnamoorthi, K. Nair, M. Smelyanskiy, and M. Annavaram, "Check-N-Run: A checkpointing system for training deep learning recommendation models," in Proc. USENIX NSDI, 2022, pp. 929–943.

[14] Z. Wang, Z. Jia, S. Zheng, Z. Zhang, X. Fu, T. S. E. Ng, and Y. Wang, "GEMINI: Fast failure recovery in distributed training with in-memory checkpoints," in Proc. ACM SOSP, 2023, pp. 364–381.

[15] Y. Wang, X. Kang, S. Shi, X. He, Z. Tang, X. Pan, Y. Zheng, X. Wu, A. C. Zhou, B. He, and X. Chu, "Fault-tolerant hybrid-parallel training at scale with reliable and efficient in-memory checkpointing," arXiv:2310.12670, 2024.

[16] B. Wan, M. Han, Y. Sheng, Y. Peng, H. Lin, M. Zhang, Z. Lai, M. Yu, J. Zhang, Z. Song, X. Liu, and C. Wu, "ByteCheckpoint: A unified checkpointing system for large foundation model development," in Proc. USENIX NSDI, 2025, pp. 559–578.

[17] J. Thorpe, P. Zhao, J. Eyolfson, Y. Qiao, Z. Jia, M. Zhang, R. Netravali, and G. H. Xu, "Bamboo: Making preemptible instances resilient for affordable training of large DNNs," in Proc. USENIX NSDI, 2023, pp. 497–513.

[18] I. Jang, Z. Yang, Z. Zhang, X. Jin, and M. Chowdhury, "Oobleck: Resilient distributed training of large models using pipeline templates," in Proc. ACM SOSP, 2023, pp. 382–395.

[19] S. Gandhi, M. Zhao, A. Skiadopoulos, and C. Kozyrakis, "ReCycle: Resilient training of large DNNs using pipeline adaptation," in Proc. ACM SOSP, 2024, pp. 211–228.

[20] H. Zhang, J. Wang, Z. Yu, Y. Zhang, X. Ji, K. Mao, J. Zhang, Y. Zhang, T. Wu, F. Jie, X. Huang, Z. Cai, J. Cheng, S. Wang, W. Li, X. Bao, H. Xu, S. Zhao, J. Li, H. Sun, Z. Zhang, Y. Xiong, and C. Li, "FlashRecovery: Fast and low-cost recovery from failures for large-scale training of LLMs," arXiv:2509.03047, 2025.

[21] C. Jin, Z. Jiang, Z. Bai, Z. Zhong, J. Liu, X. Li, N. Zheng, X. Wang, C. Xie, Q. Huang, W. Heng, Y. Ma, W. Bao, S. Zheng, Y. Peng, H. Lin, X. Liu, X. Jin, and X. Liu, "MegaScale-MoE: Large-scale communication-efficient training of mixture-of-experts models in production," arXiv:2505.11432, 2025.

[22] Z. Yan, H. Bai, X. Yao, D. Liu, T. Liu, H. Liu, P. Li, E. Wu, S. Fan, L. Tao, R. Zhang, Y. Wang, S. Xu, J. Chang, X. Chen, K. Li, Y. Bai, G. Deng, N. Zheng, V. A. Korthikanti, et al., "Scalable training of mixture-of-experts models with Megatron Core," arXiv:2603.07685, 2026.

[23] W. Cai, L. Qin, and J. Huang, "MoC-System: Efficient fault tolerance for sparse mixture-of-experts model training," in Proc. ASPLOS, 2025, pp. 655–671.

[24] J. O. Kephart and D. M. Chess, "The vision of autonomic computing," IEEE Computer, vol. 36, no. 1, pp. 41–50, 2003.

[25] D. Garlan, S.-W. Cheng, A.-C. Huang, B. Schmerl, and P. Steenkiste, "Rainbow: Architecture-based self-adaptation with reusable infrastructure," IEEE Computer, vol. 37, no. 10, pp. 46–54, 2004.

[26] M. Salehie and L. Tahvildari, "Self-adaptive software: Landscape and research challenges," ACM Trans. Auton. Adapt. Syst., vol. 4, no. 2, pp. 14:1–14:42, 2009.

[27] C. Wohlin, P. Runeson, M. Höst, M. C. Ohlsson, B. Regnell, and A. Wesslén, Experimentation in Software Engineering. Springer, 2012.

[28] A. Arcuri and L. Briand, "A hitchhiker's guide to statistical tests for assessing randomized algorithms in software engineering," Software Testing, Verification and Reliability, vol. 24, no. 3, pp. 219–250, 2014.

[29] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "GShard: Scaling giant models with conditional computation and automatic sharding," in Proc. ICLR, 2021.

[30] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," J. Mach. Learn. Res., vol. 23, no. 120, pp. 1–39, 2022.

[31] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale," in Proc. ICML, 2022.

[32] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, J. Chau, P. Cheng, F. Yang, M. Yang, and Y. Xiong, "Tutel: Adaptive mixture-of-experts at scale," in Proc. MLSys, 2023.

[33] T. Gale, D. Narayanan, C. Young, and M. Zaharia, "MegaBlocks: Efficient sparse training with mixture-of-experts," in Proc. MLSys, 2023.

[34] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "FasterMoE: Modeling and optimizing training of large-scale dynamic pre-trained models," in Proc. PPoPP, 2022, pp. 120–134.

[35] M. Zhai, J. He, Z. Ma, Z. Zong, R. Zhang, and J. Zhai, "SmartMoE: Efficiently training sparsely-activated models through combining offline and online parallelization," in Proc. USENIX ATC, 2023, pp. 961–975.

[36] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, and M. Zaharia, "Efficient large-scale language model training on GPU clusters using Megatron-LM," in Proc. SC, 2021.

[37] E. N. Elnozahy, L. Alvisi, Y.-M. Wang, and D. B. Johnson, "A survey of rollback-recovery protocols in message-passing systems," ACM Comput. Surv., vol. 34, no. 3, pp. 375–408, 2002.

[38] J. W. Young, "A first order approximation to the optimum checkpoint interval," Commun. ACM, vol. 17, no. 9, pp. 530–531, 1974.

[39] J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps," Future Generation Computer Systems, vol. 22, no. 3, pp. 303–312, 2006.

[40] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in Proc. ICLR, 2017.

[41] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus, "ST-MoE: Designing stable and transferable sparse expert models," arXiv:2202.08906, 2022.

[42] A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. Riquelme Ruiz, B. Mustafa, J. Ainslie, Y. Tay, M. Dehghani, and N. Houlsby, "Sparse upcycling: Training mixture-of-experts from dense checkpoints," in Proc. ICLR, 2023.

[43] G. Penedo, H. Kydlíček, L. Ben Allal, A. Lozhkov, M. Mitchell, C. Raffel, L. von Werra, and T. Wolf, "The FineWeb datasets: Decanting the web for the finest text data at scale," in Proc. NeurIPS Datasets and Benchmarks Track, 2024.

[44] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac'h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, "A framework for few-shot language model evaluation," Zenodo, version v0.4.0, 2023.

[45] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, "Think you have solved question answering? Try ARC, the AI2 reasoning challenge," arXiv:1803.05457, 2018.

[46] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova, "BoolQ: Exploring the surprising difficulty of natural yes/no questions," in Proc. NAACL-HLT, 2019, pp. 2924–2936.

[47] A. Amini, S. Gabriel, P. Lin, R. Koncel-Kedziorski, Y. Choi, and H. Hajishirzi, "MathQA: Towards interpretable math word problem solving with operation-based formalisms," in Proc. NAACL-HLT, 2019, pp. 2357–2367.

[48] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, "Can a suit of armor conduct electricity? A new dataset for open book question answering," in Proc. EMNLP, 2018, pp. 2381–2391.

[49] Y. Bisk, R. Zellers, R. Le Bras, J. Gao, and Y. Choi, "PIQA: Reasoning about physical commonsense in natural language," in Proc. AAAI, vol. 34, 2020, pp. 7432–7439.

[50] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy, "RACE: Large-scale ReAding Comprehension dataset from Examinations," in Proc. EMNLP, 2017, pp. 785–794.

[51] R. Zellers, Y. Bisk, R. Schwartz, and Y. Choi, "SWAG: A large-scale adversarial dataset for grounded commonsense inference," in Proc. EMNLP, 2018, pp. 93–104.

[52] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi, "WinoGrande: An adversarial Winograd Schema Challenge at scale," in Proc. AAAI, vol. 34, 2020, pp. 8732–8740.

[53] S. Biderman, H. Schoelkopf, Q. Anthony, H. Bradley, K. O'Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. van der Wal, "Pythia: A suite for analyzing large language models across training and scaling," in Proc. ICML, 2023, pp. 2397–2430.

[54] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," J. Mach. Learn. Res., vol. 15, no. 56, pp. 1929–1958, 2014.

[55] C. M. Bishop, "Training with noise is equivalent to Tikhonov regularization," Neural Computation, vol. 7, no. 1, pp. 108–116, 1995.

[56] B. H. C. Cheng, R. de Lemos, H. Giese, P. Inverardi, J. Magee et al., "Software engineering for self-adaptive systems: A research roadmap," in Software Engineering for Self-Adaptive Systems, LNCS 5525. Springer, 2009, pp. 1–26.

[57] D. Weyns, "Software engineering of self-adaptive systems," in Handbook of Software Engineering, S. Cha, R. N. Taylor, and K. Kang, Eds. Springer, 2019, pp. 399–443.

[58] S. Athlur, N. Saran, M. Sivathanu, R. Ramjee, and N. Kwatra, "Varuna: Scalable, low-cost training of massive deep learning models," in Proc. EuroSys, 2022, pp. 472–487.

[59] J. Duan, Z. Song, X. Miao, X. Xi, D. Lin, H. Xu, M. Zhang, and Z. Jia, "Parcae: Proactive, liveput-optimized DNN training on preemptible instances," in Proc. USENIX NSDI, 2024, pp. 1121–1139.

[60] A. Qiao, A. Aghayev, W. Yu, H. Chen, Q. Ho, G. A. Gibson, and E. P. Xing, "Litz: Elastic framework for high-performance distributed machine learning," in Proc. USENIX ATC, 2018, pp. 631–644.

[61] D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang, "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," arXiv:2401.06066, 2024.

[62] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, "Scaling vision with sparse mixture of experts," in Proc. NeurIPS, 2021.

[63] P. H. Hochschild, R. Govindaraju, and D. E. Culler, "Cores that don't count," in Proc. USENIX HotOS, 2021.
