# MoEGambit：基于契约的混合恢复用于 Mixture-of-Experts 训练

## 摘要

大规模 Mixture-of-Experts（MoE）训练在数千 GPU 上运行数月，rank 故障频繁发生。现有恢复机制通常通过从全局一致的 checkpoint 重启来恢复，将复制的非专家状态与 rank 独占的专家状态视为单一整体对象。该重启路径丢弃了 checkpoint 之后已完成的 GPU 工作：要从 checkpoint 步 $c$ 回到故障步 $t$，任务重新加载旧状态并重放 $t-c$ 次迭代，尽管大量当前状态可能仍存活于健康对等节点上。当专家数据并行（EDP）为 1 时，MoE 训练使这一机会尤为突出，因为非专家状态是被复制的，而专家状态可能是 rank 独占的。

我们提出 **MoEGambit**，一个面向稀疏 MoE 训练的恢复侧框架，以更细的粒度恢复状态。当运行时契约允许混合修复时，MoEGambit 从健康的稠密数据并行（dense-DP）对等节点拉取当前非专家状态，仅从 checkpoint 加载失败 rank 的专家分片，消除重放并将 checkpoint I/O 减少到没有存活对等节点的状态。该契约结合了安全修复点控制器、限定部分修复安全时机的专家加权陈旧度密度 $\Phi'(t)$，以及带结构化日志的守卫式重整合状态机。weights-first/optimizer-later 两阶段协议进一步将优化器恢复与恢复后的计算重叠。

在 Qwen3-30B-A3B 和 DeepSeek-V2-Lite 上的实验中，MoEGambit 将原始恢复延迟降低 $20.6\%$--$55.0\%$，实现了 $36.9\times$ 的含重放端到端加速。在契约允许的单次、重复和突发故障场景下，配对评估发现恢复后的验证损失、perplexity 和下游零样本准确率均无可检测的退化。

**关键词：** 大语言模型、混合专家、容错、分布式训练、故障恢复、checkpoint、运行时监控

## 1 引言

训练大语言模型是一个跨数月、数千 GPU 的软件过程，在此期间故障频繁发生。Llama 3 报告在 54 天、16,384-GPU 的预训练期间发生了 419 次意外中断，大约每三小时一次 [1]；MegaScale 记录了 10K-GPU 部署中的生产故障和落后节点 [2]；OPT-175B 日志记录了训练期间漫长的人工干预 [3]。标准恢复机制——**checkpoint restart**——重新加载最新的全局一致快照并重放所有丢失的迭代，无论哪些状态实际失效。在现代故障率下，即使每次故障的停机时间很适中，也会在训练运行中累积为大量的 GPU 时间。

稀疏 Mixture-of-Experts（MoE）架构被最先进的 LLM 广泛采用，包括 Mixtral [4]、DeepSeek-V2/V3 [5,6]、Qwen3-MoE [7]、Ling（百灵）[8]、Ring-flash-linear-2.0 [9] 和 Kimi K2 [10]。对于这些模型，恢复成本是一个结构性问题。在稠密训练中，失败的副本通常可以从健康的数据并行对等节点重建，因为完整的模型状态是被复制的。在 MoE 训练中，只有**非专家**状态（注意力、嵌入、router）以这种方式被复制；专家参数和优化器矩在专家分片后通常是 rank 独占的。重启忽略了这种区分：它回滚整个作业，重新加载所有 checkpoint 状态，并重放每个丢失的迭代。

先前的容错训练系统减少了 checkpoint 成本 [11,12,13,14,15,16]，围绕故障适应拓扑 [17,18,19]，或从对等节点恢复稠密副本 [20]。然而，它们都没有解决 MoE 的恢复侧不对称性。Checkpoint/副本系统保持了 checkpoint 一致的重启语义；拓扑适应系统依赖于 rank 可互换性；稠密对等恢复假设失败 rank 的完整状态有一个存活的对等节点。当 $\text{EDP}=1$ 时，这一假设被打破：失败 rank 上的所有专家分片都是唯一的，因此非专家状态可以从对等节点恢复，但专家状态没有存活对等节点且必须来自 checkpoint。当 $\text{EDP}>1$ 时，存在专家副本，稠密式对等恢复也可以修复专家状态；我们独特的目标是先前对等恢复方法无法覆盖的 $\text{EDP}=1$ 操作点。这个边界很重要，因为最近的 MoE 模型和生产系统使用大量专家和宽专家并行来扩展模型容量同时降低通信成本。在 Megatron Core 的 MoE 并行映射下，在固定 GPU 预算下增加 EP 或专家张量并行（ETP）会驱动 $\text{EDP}=W/(\text{PP}\times\text{EP}\times\text{ETP})$ 趋向 1 [6,7,21,22]。我们所知唯一的 MoE 专用容错系统 MoC-System [23] 通过部分专家检查点（Partial Experts Checkpointing）优化了保存侧成本，但仍从 checkpoint 状态恢复。这留下了一个恢复侧的空白：没有现有系统将对 MoE 复制状态的对等恢复与 rank 独占专家的 checkpoint 修复结合起来，同时限制部分修复引入的质量风险。

**关键观察。** 对等恢复对稠密训练来说已经成熟，但 MoE 的异构状态使恢复在 $\text{EDP}=1$ 时仅**部分**可从对等节点恢复。非专家参数（注意力、嵌入、router）在 dense-DP 维度上被复制，可以从健康的 dense-DP 对等节点拉取，就像稠密对等恢复一样。然而，专家参数和优化器状态是 EP 分片的，在 $\text{EDP}=1$ 时没有存活的对等节点。当一个 rank 失败时，替换节点因此可以从健康的 dense-DP 对等节点拉取当前步的非专家状态，仅从最新的 checkpoint 加载 rank 本地的专家分片——消除重放，同时将磁盘 I/O 减少到 rank 独占的专家体量。在 $\text{EDP}>1$ 时，MoEGambit 使用相同的状态类边界，但可以从存活的专家对等节点恢复专家状态，匹配先前对等恢复系统假设的更简单设置。

混合恢复提出了一个超出 I/O 成本的软件工程问题。它以当前非专家状态但 checkpoint 陈旧的专家状态恢复，因此恢复的专家落后模型其余部分 $\Delta=t-c$ 次迭代；重复恢复可以在专家群体中累积这种陈旧度。如果没有运行时可检查的规范说明混合路径何时安全，操作员将面临速度和模型质量之间未经审计的权衡。遵循自适应系统的监控、可审计运行时适应视图 [24,25,26]，MoE 恢复应该针对恢复后的训练轨迹来指定，而不仅仅针对进程存活。

我们提出 **MoEGambit**，一个面向稀疏 MoE 训练的恢复侧框架。MoEGambit 仅在专家陈旧度被界定和记录时才允许混合恢复。它引入了一个具有三个构件的显式恢复契约：防止部分优化器提交的安全修复点控制器（R1），带经验校准质量阈值的专家加权陈旧度密度 $\Phi'(t)$（R2），以及带结构化日志的守卫式重整合状态机（R3）。MoEGambit 然后贡献两种恢复机制：一种 MoE 感知的混合修复路径，从健康的 dense-DP 对等节点拉取当前非专家参数并仅从 checkpoint 加载失败 rank 的专家分片，以及一种 weights-first/optimizer-later 两阶段恢复协议，将优化器恢复与恢复后的计算重叠。

我们在 Qwen3-30B-A3B [7]（128 专家，64 H20 GPU）和 DeepSeek-V2-Lite [5]（64 专家）上按照软件工程（SE）实验标准评估 MoEGambit：每个恢复的运行都与一个共享相同 seed、数据顺序、checkpoint 和注入故障步骤的 NoFault baseline 配对 [27,28]，以便残留的质量差异可以专门归因于恢复而不是 seed 噪声。MoEGambit 将原始恢复延迟降低 **$20.6\%$--$55.0\%$**，并在混合路径下避免重放，实现了 **$36.9\times$** 的含重放端到端加速。在契约允许的单次、重复和突发故障场景下，配对测试显示验证损失或 perplexity 没有统计学上可检测到的退化；在 50 次单次故障热力图中，44/50 次运行落在 NoFault $\pm2\sigma_{\text{base}}$ 带内，下游零样本准确率与 Restart 无法区分，同时保持在 NoFault 噪声范围内。

本文做出三个贡献：

- **运行时恢复契约。** 三个软件工程构件——安全修复点不变量（R1）、带经验校准质量阈值的专家加权陈旧度密度 $\Phi'(t)$（R2），以及带结构化审计日志的守卫式重整合状态机（R3）——将**何时**允许混合修复与**如何**重构状态分离（§3）。
- **混合修复和两阶段恢复。** 一种从健康对等节点在步 $t$ 拉取 dense-DP 复制的非专家状态、仅从每 rank checkpoint 分片在步 $c$ 加载 EP 分片专家状态的恢复侧机制，加上一种将优化器恢复与恢复后训练重叠的 weights-first/optimizer-later 协议（§3.5-§3.6）。
- **配对故障注入评估。** 在 Qwen3-30B-A3B 和 DeepSeek-V2-Lite 上遵循 SE 实验指南 [27,28] 的配对 NoFault baseline 可复现基准，展示了 $20.6\%$--$55.0\%$ 恢复延迟降低、$36.9\times$ 端到端加速，以及在六个 RQ 上无可检测的质量退化（§4）。

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

我们首先定义使部分恢复成为可能的 MoE 状态类，然后展示为什么 checkpoint restart 未利用这种结构。

### 2.1 概念：MoE 训练作为异构状态分布式程序

分布式 MoE 训练任务 [29,30,31,32,33,22] 是在数十到数千 GPU 上执行集合通信的长期运行、有状态、多进程程序。现代 MoE 框架对稠密/注意力层和 MoE 层使用不同的映射：稠密层使用张量并行（TP）、上下文并行（CP）、流水线并行（PP）和数据并行（DP），而 MoE 层使用专家并行（EP）、专家张量并行（ETP）和专家数据并行（EDP）。与稠密 Transformer 不同，MoE 训练维护着一种**异构状态**，具有三个恢复相关的类别：

- **复制状态。** 非专家层（注意力、嵌入、层归一化）和 router 权重在共享相同 PP/dense-TP 坐标的每个 dense-DP rank 间被复制。在任意迭代 $t$，至少一个健康的对等节点持有当前步的值。
- **分片专家状态。** 专家权重及其 AdamW 一阶/二阶矩沿 EP 和 ETP 组分片。在 $128$ 专家、EP$=8$、ETP$=1$ 的布局中无专家复制时，每个 rank 独占 $128/8=16$ 个专家的固定子集；没有对等节点持有当前内存中的副本。
- **运行时元数据。** 专家目录、all-to-all 路由使用的 dispatch 拓扑 [34,35]，以及进程组视图由 rank 布局派生，任何 rank 一被替换就立即失效。

**专家数据并行（EDP）。** 按照 Megatron Core 当前的 MoE 术语 [22]，我们区分稠密张量并行（TP）和专家张量并行（ETP）。稠密层由 TP、CP、PP 和 dense-DP 组织，而 MoE 层由 PP、EP、ETP 和 EDP 组织。由于我们的实验未使用上下文并行，专家布局满足

$$W=\text{PP}\times\text{EP}\times\text{ETP}\times\text{EDP}, \quad \text{EDP}=\frac{W}{\text{PP}\times\text{EP}\times\text{ETP}}.$$

这不同于旧版的 EP-as-a-subdimension-of-DP 描述，其中 TP 有时被包含在 EDP 分母中。在当前映射下，稠密 TP 影响注意力层分片但本身不决定专家复制；ETP 是 MoE 层的张量并行度。

当 $\text{EDP}=1$ 时，每个专家分片恰好存在于一个存活 rank 上；当 $\text{EDP}>1$ 时，失败的专家分片可能有一个存活的对等副本，类似于非专家参数使用的对等副本。$\text{EDP}=1$ 是最受约束的恢复操作点，因为每个专家分片都是 rank 独占的。在我们评估的 Qwen3-30B-A3B 布局中，$W=64$，PP$=8$，EP$=8$，ETP$=1$，所以 $\text{EDP}=64/(8\cdot8\cdot1)=1$，即使模型有 128 个专家；每个 EP rank 在其流水线阶段内拥有 16 个专家。这个操作点在大型专家 MoE 系统 [6,7,21] 中是资源驱动的：大量专家数量已经将内存分布到各 rank，而额外的专家复制会增加专家和优化器内存但不增加激活计算量，因为每个 token 仅激活 $K\!\ll\!E$ 个专家。这也是 MoE 恢复与稠密恢复不同的设置：失败 rank 的专家部分没有存活对等节点，必须从 checkpoint 恢复。

图 1 说明了这种区分。

*图 1：8 GPU 下的专家分布（dense TP=1，ETP=1，PP=2）。EDP>1 时，专家有存活对等节点；EDP=1 时，每个专家分片是唯一的，必须从 checkpoint 恢复。非专家参数（绿色）在两种布局中都被 dense-DP 复制。*

软件工程后果是：**同一次故障使不同状态类以不同恢复语义失效**；标准 checkpoint restart 未利用这一边界。

### 2.2 现有流程：Checkpoint Restart 与部分恢复的隐藏风险

**动机示例。** 在我们的 64-GPU Qwen3-30B-A3B 任务中，在步 $t\!=\!200$ 检测到的故障且最新 checkpoint 在 $c\!=\!100$ 会导致标准 Megatron-LM 重新加载 56 GB 的分布式 checkpoint 并重放 100 次迭代，停机约 1067 秒。然而，失败的 rank 仅拥有 128 个专家中的 16 个；非专家层和其他 112 个专家仍存活于健康 rank 上。MoE 感知的恢复路径因此可以从对等节点拉取 3.2 GB 的非专家状态，仅加载失败 rank 的专家分片，并在 29 秒内从步 $t$ 恢复，无需重放。这一机会也引入了质量风险：恢复的专家陈旧 $\Delta=t-c$ 次迭代。

生产 LLM 训练通常使用 **checkpoint restart** [36,1,2]：故障发生后，任务重新加载步 $c$ 的最近全局一致快照并重放 $t-c$ 次丢失的迭代。经典回滚恢复和检查点间隔分析 [37,38,39] 表明，最优检查点间隔满足 $\tau^{*}\!\approx\!\sqrt{2CM}$，其中 $C$ 是每次检查点的成本，$M$ 是平均故障间隔；在 10K-GPU 集群上 $M$ 坍缩到几十分钟 [1,2]，重复故障可能在检查点 I/O 和重放中消耗大量的聚合 GPU 时间。

Restart 将整个任务视为单一的全有或全无的状态对象。表 2 展示了为什么这是浪费的：一个失败的 rank 仅使其 EP 分片的专家状态失效，而同一 PP/dense-TP 坐标上每个其他 rank 的 dense-DP 复制非专家状态在步 $t$ 仍然是当前的。一种自然的替代是**混合恢复**：从健康的 dense-DP 对等节点在当前步同步非专家状态，仅从最新 checkpoint 分片在步 $c$ 加载 rank 本地的专家。然而，混合恢复引入了 restart 没有的副作用：恢复 rank 的专家状态相对于模型其余部分陈旧 $\Delta = t-c$ 次迭代。

**表 2：EDP=1 恢复情况下步 $t$ 的 MoE 训练状态来源。**

| 状态 | 布局 | 来源 |
| --- | --- | --- |
| 非专家（attn/embed/router） | Dense-DP | 对等节点（步 $t$） |
| 专家权重 | EP/ETP | Checkpoint 分片（$c\!<\!t$） |
| 专家优化器 | EP/ETP | Checkpoint 分片（$c\!<\!t$） |
| 运行时元数据 | Rank 派生 | 重计算 |

这种陈旧度不是二元缺陷——恢复的专家仍产生有效的前向/反向信号——但它可能扰动由 MoE 辅助损失 [40,30,41,42] 驱动的负载均衡动力学。风险有两个维度：失败 rank 拥有的专家的**每次事件**间隔 $\Delta$，以及不同专家分片上重复混合恢复累积的**窗口级**债务。允许部分修复的恢复系统因此需要累积陈旧度的运行时可检查边界，而不仅仅是快速机制。

**问题陈述。** MoEGambit 从集群看门狗接收一个 fail-stop 事件 $\langle r,t,c\rangle$，其中 $r$ 是失败的逻辑 rank，$c$ 是最新 checkpoint，$t$ 是安全点处理后应执行的第一个步。如果故障在迭代中途被检测到，R1 丢弃该进行中的迭代并将 $t$ 推进到下一个有效步；因此 $\Delta=t-c$ 计算 checkpoint restart 会重放的迭代次数。MoEGambit 必须选择 **RESTART**（从 checkpoint 恢复所有状态并重放 $t-c$ 次迭代）或 **HYBRID**（从 dense-DP 对等节点恢复当前非专家状态，从失败 rank 的 checkpoint 分片恢复专家状态）。在两种情况下，恢复后的第一次 `optimizer.step()` 必须是有效的 AdamW 更新：不能将梯度应用于未初始化或部分恢复的参数。

**需求。** 我们将恢复路径形式化为包含三个软件工程需求的契约。**R1** 定义安全修复点并守护优化器提交，使得中途步故障不能损坏模型状态。**R2** 通过量化、$O(1)$ 的守卫指定混合恢复资格，捕获每次事件和累积陈旧度。**R3** 记录每个决策、守卫输入、延迟分段和状态机转换，以支持事后可审计性。MoEGambit 假设 fail-stop rank 故障、至少一个可用 checkpoint、失败 rank 非专家状态的健康 dense-DP 对等节点，以及一个替换/热备 GPU；如果对等节点条件失败，策略选择 **RESTART**。

## 3 方法

### 3.1 概览

MoEGambit 是一个运行时恢复层，拦截 rank 故障事件并从每个状态类的最新可用来源重建替换 rank 的状态，而不回滚任务的其余部分。图 2 总结了运行时架构。设计将**被保证的内容**（恢复契约，R1-R3）与**如何修复状态**（混合恢复机制和两阶段协议）分离，使契约可以独立于实现接受审计。

框架在每个故障事件上执行五个组件（图 2）：

1. **故障检测。** 外部看门狗发出故障事件 $\langle r,t,c\rangle$，标识失败 rank、当前训练步和最新 checkpoint 步。
2. **安全修复点控制器**（§3.2，R1）。安装优化器提交守卫，使没有任何 rank 应用在恢复期间计算的梯度。
3. **陈旧度密度守卫策略**（§3.3，R2）。评估 $O(1)$ 守卫 $\Phi'(t)$ 并返回确定性决策 $\pi(t)\!\in\!\{\text{HYBRID}, \text{RESTART}\}$。如果投影的陈旧度密度超过经验质量阈值，MoEGambit 回退到 checkpoint restart。
4. **MoE 感知混合恢复与两阶段协议**（§3.5-§3.6）。从健康 dense-DP 对等节点在步 $t$ 拉取非专家状态（路径 P），从步 $c$ 的 checkpoint 分片加载 rank 本地专家（路径 C），并将优化器状态恢复与恢复后的计算重叠。
5. **重整合状态机与结构化日志**（§3.7，R3）。将替换 rank 推进到 HEALTHY 并发出关联策略输入、延迟和状态机转换的结构化记录。

### 3.2 安全修复点控制器（R1）

R1 定义一个单一程序点，在该点上修复可以运行，具有一个不变量：没有优化器提交可以使用在 RECOVERING 下计算的梯度。在故障事件 $\langle r,t,c\rangle$ 上，控制器原子地将 $r$ 及其专家标记为 RECOVERING，在每个 rank 上为当前迭代的 `optimizer.step()` 安装 pre-AdamW 提交守卫，并将当前迭代标记为 DISCARD。任何在 DISCARD 下到达 `optimizer.step()` 的 rank 跳过更新；进行中的梯度被丢弃。训练随后通过混合修复（§3.5）或 restart 继续，控制器记录安全点状态。

### 3.3 陈旧度密度守卫策略（R2）

R2 定义一个 $O(1)$ 守卫 $\Phi'(t)$ 和一个确定性决策 $\pi(t)\!\in\!\{\text{HYBRID}, \text{RESTART}\}$。决策使用故障事件 $\langle r,t,c\rangle$、窗口陈旧债务 $S(t)=\sum_{h\in\mathcal{H}_{W_{\mathrm{exp}}}(t)}|E_h|\Delta_h$、谓词 $\text{PeerAvail}(r,t)$，以及在 §4 中校准的阈值 $(\Delta_{\min},\Delta_{\max},\Phi_{\max})$。其中 $W_{\mathrm{exp}}$ 是暴露窗口长度，$\mathcal{H}_{W_{\mathrm{exp}}}(t)$ 是 $[t\!-\!W_{\mathrm{exp}},t)$ 内先前混合事件的集合，$E_h$ 是事件 $h$ 恢复的专家集合，$\Delta_h$ 是其检查点间隔，$N_{\text{expert}}$ 是暴露域中的路由专家数；因此 $S(t)$ 的单位是专家-迭代。如果当前事件通过混合路径恢复，投影密度将是

$$\Phi'(t) = \frac{S(t) + |E_{\text{new}}| \cdot \Delta}{N_{\text{expert}} \cdot W_{\mathrm{exp}}}, \tag{1}$$

其中 $N_{\text{expert}}\!\cdot\!W_{\mathrm{exp}}$ 是暴露所有专家一个完整窗口的债务；$\Phi'(t)=0.1$ 因此意味着投影暴露等于该全窗口预算的 10%。该值是暴露密度，不是概率，可以在窗口内重复暴露下超过 1。运行聚合 $S(t)$ 作为单一标量维护：混合事件 $h$ 进入窗口时加 $|E_h|\Delta_h$，该事件离开窗口（超过 $W_{\mathrm{exp}}$ 迭代）时减去相同乘积。这产生了 $O(1)$ 的每迭代记账。当多个 rank 同时失败时，每个失败 rank 在策略为下一个 rank 评估之前贡献自己的 $|E_h|\Delta_h$ 项到 $S(t)$。

**直觉。** 守卫是一个可监控的暴露预算，不是全局收敛的声明。混合事件将 $|E_h|$ 个专家暴露于落后模型其余部分 $\Delta_h$ 次迭代的权重；重复事件暴露更多专家更长时间。按 $|E_h|\Delta_h$ 对每个事件计费捕获了两个因素，按 $N_{\text{expert}}\!\cdot\!W_{\mathrm{exp}}$ 归一化使 $\Phi'(t)$ 可跨模型大小和窗口长度比较。阈值 $\Phi_{\max}\!=\!10^{-1}$（§4）是相对于观察到的无故障噪声地板校准的。

决策为

$$\pi(t) = \begin{cases}
\text{HYBRID}, & \text{if } \text{PeerAvail}(r,t) \wedge \Delta_{\min} \leq \Delta \leq \Delta_{\max} \wedge \Phi'(t) \leq \Phi_{\max}, \\
\text{RESTART}, & \text{otherwise}.
\end{cases} \tag{2}$$

算法 1 编码了相同的逻辑，并在每次回退时记录第一个失败的守卫。三个阈值扮演不同角色。$\Delta_{\min}$ 是成本-效益下界，避免当重放间隔太小不足以抵消协调成本时使用混合修复；$\Delta_{\max}$ 是任何单个恢复专家分片的每次事件陈旧上界；$\Phi_{\max}$ 是累积专家暴露的窗口级上界。每个阈值绑定到可测量的来源：$\Delta_{\min}$ 绑定到 $T_{\text{load}}\!:\!T_{\text{hybrid}}\!:\!T_{\text{iter}}$ 比率，$\Delta_{\max}$ 绑定到由 Young/Daly [38,39] 限定的检查点间隔，$\Phi_{\max}$ 绑定到守卫禁用的质量扫描，$\text{PeerAvail}$ 绑定到存活进程组。日志条目包含 $\langle\pi(t),\text{reason}\rangle$ 和 $(\Delta,|E_{\text{new}}|,S(t),\Phi'(t))$（§3.7）；紧凑的原因标签标识对等节点不可用、两个间隔守卫、暴露守卫和允许的混合路径。

**算法 1：** 守卫式恢复决策 $\pi(t)$。

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

恢复契约是 MoEGambit 可以选择快速路径的可监控安全条件。对于每个恢复事件，混合恢复蕴含所有三个守卫：

$$\pi(t)=\text{HYBRID} \Rightarrow \text{PeerAvail}(r,t) \wedge \Delta_{\min}\leq\Delta\leq\Delta_{\max} \wedge \Phi'(t)\leq\Phi_{\max}, \quad \forall t.$$

R3 记录每个合取的证据，因此契约可以在线检查和离线审计。阈值 $\Phi_{\max}$ 是经验的，但 $\Phi'(t)$ 的形式来自一阶暴露边界。

**命题 1（专家暴露边界）。** 设 $\theta_e(s)$ 表示步 $s$ 后专家 $e$ 的状态。假设在恢复窗口内且非专家状态固定时，被监控的 MoE 损失分量 $\ell_{\text{aux}}$ 在每个专家状态上按坐标是 $L_{\text{aux}}$-Lipschitz 的，正常步在同一范数下改变任何专家至多为 $G$。如果事件 $h$ 从检查点步 $c_h$ 恢复专家集 $E_h$ 并在 $t_h$ 恢复，则其陈旧专家扰动相对于当前状态恢复满足

$$\delta\ell_h \leq L_{\text{aux}}\sum_{e\in E_h}\|\theta_e(t_h)-\theta_e(c_h)\| \leq L_{\text{aux}}G\,|E_h|\,(t_h-c_h).$$

因此，将当前候选事件添加到现有窗口债务后，

$$\sum_{h\in\mathcal{H}_{W_{\mathrm{exp}}}(t)\cup\{\text{new}\}}\delta\ell_h \leq L_{\text{aux}}G\,(S(t)+|E_{\text{new}}|\Delta)=L_{\text{aux}}G\,N_{\text{expert}}W_{\mathrm{exp}}\Phi'(t).$$

因此 $\Phi'(t)$ 是一个归一化暴露代理，其诱导的一阶扰动边界随 $\Phi'(t)$ 线性缩放；它不是独立的收敛定理。

### 3.5 MoE 感知混合状态恢复

当 $\pi(t)=\text{HYBRID}$ 时，替换 rank 从每个状态类的最新可用来源重建其状态（表 2），因此 I/O 与 rank 独占专家体量而非完整分布式 checkpoint 成比例。两条路径并行运行。**路径 P（对等拉取，复制状态）** 通过 NCCL/Gloo 上的单次 P2P 广播，从同一 PP/dense-TP 坐标的健康对等节点同步当前步非专家参数（和 dense-DP 复制的优化器状态）；传输在数十毫秒内完成。**路径 C（分片读取，分片专家状态）** 从步 $c$ 的每 rank 分片加载 rank-$r$ 的专家，不需要集合通信。恢复的 rank 刻意是混合的：非专家状态反映步 $t$，而专家及其优化器状态陈旧 $\Delta$——即 (1) 中 $\Phi'(t)$ 限定的量。

### 3.6 两阶段恢复协议

两阶段协议通过将路径 C 的优化器状态读取与恢复的前向/反向重叠来减少恢复时间，同时禁止在未初始化专家槽上提交。**阶段 A（权重优先）** 恢复非专家状态和专家权重，然后在 **更新屏障** 下重新加入：前向/反向贡献到全局损失，但受影响专家的梯度被缓冲。**阶段 B（优化器延后）** 并行恢复非专家和专家矩；当它完成时，屏障释放，缓冲的梯度通过一次 AdamW 步应用。屏障复用 §3.2 的 pre-`step()` 钩子，因此 R1 在整个过程中成立。日志记录 $T_{\text{TTR}}$ 和 $T_{\text{TTFR}}$；它们的差距是相对于单阶段 restart 节省的延迟。因为协议控制的是优化器状态**何时**挂载，而不是字节**从哪里**来，§4.4 的 $2{\times}2$ 析因实验独立于混合恢复测试了它。

### 3.7 重整合状态机与结构化日志（R3）

R3 要求每个恢复决策及其下游模型质量结果都是可追溯的。我们通过两个机制实现 R3：一个强制重整合协议的守卫状态机，以及一个记录每个恢复事件输入和输出的结构化日志。

**守卫状态机。** 每个替换 rank 通过四状态线性协议推进：

$$\text{RECOVERING} \xrightarrow{g_1} \text{REPAIRED} \xrightarrow{g_2} \text{BARRIER} \xrightarrow{g_3} \text{HEALTHY},$$

其中每个转换由守卫谓词控制：

- $g_1$：状态恢复完成——所有状态类（非专家通过路径 P，专家通过路径 C）已加载并对其来源校验和验证。
- $g_2$：dispatch 拓扑重推导——专家目录和 all-to-all 路由表已重算以反映替换 rank 在 EP 布局中的位置。
- $g_3$：优化器状态挂载且更新屏障释放——两阶段协议的阶段 B（§3.6）已完成，缓冲的梯度已应用。

因为每个转换都要求其守卫成立，状态机拒绝过早重整合——例如，将 token 路由到优化器状态尚未恢复的专家。

**结构化恢复日志。** 在每个恢复事件上，状态机发出一条记录，包含策略输入（$\Delta$、$|E_{\text{new}}|$、$S(t)$、$\Phi'(t)$）、决策和原因字符串、每段延迟和转换时间戳。此记录将每个恢复路径链接回 (2) 中的守卫，并支持与 §4 配对运行方法论下训练质量指标的事后关联 [27,28]。$\Phi'(t)$ 记账和日志发出的开销是每次迭代 $O(1)$ 且经验上可忽略（§4.7）。

### 3.8 实现

MoEGambit 是基于 Megatron-LM（commit `core_r0.9.0`）的 2,847 行 Python 补丁，覆盖安全点控制、策略评估、混合恢复和重整合。它拦截优化器步和广播调用，同时保持内核、检查点保存、启动器和守护进程设置不变；不使用 `--enable-moegambit` 时，训练遵循未修改的路径。我们将在发表后开源实现和评估脚本。

## 4 评估

本节评估 MoEGambit 是否在不退化恢复后训练轨迹的情况下降低恢复成本。我们提出六个研究问题：

- **RQ1：** MoEGambit 降低了多少恢复成本，混合恢复和两阶段恢复是否加性组合？
- **RQ2：** 契约允许的混合恢复是否避免了相对于无故障训练的可检测质量退化？
- **RQ3：** $\Phi'(t)$ 是否在重复和突发故障下识别质量退化？
- **RQ4：** 监控钩子在无故障训练期间增加了什么开销？
- **RQ5：** 恢复优势是否跨规模和并行布局保持？
- **RQ6：** 该机制是否泛化到不同的 MoE 模型配置？

### 4.1 实验设置

**集群和并行。** 除非另有说明，实验在 64 个 H20-3e GPU（8 节点 × 8 GPU）上运行，配置为 dense TP=1，PP=8，EP=8，ETP=1，EDP=1（Megatron Core 定义）。可扩展性和并行敏感性实验使用 RQ5 描述的覆盖。

**模型和分词器。** 我们使用 Qwen3-30B-A3B [7]（30B 总参 / 3B 激活，48 层，128 专家，top-8 路由）配合 Qwen2Tokenizer（151,936-token 字节级 BPE）。默认训练超参数见表 3。

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

**数据。** 我们使用 FineWeb [43] 的 4B-token 子集，处理为 Megatron 索引格式。每 1,000 次迭代报告保留验证集上的损失/perplexity。所有恢复实验使用**配对运行**：baseline 和 MoEGambit 共享相同的 seed、数据顺序、checkpoint 和注入故障步。

**Baseline 计时和噪声尺度。** 稳态 $T_{\text{iter}}\!\approx\!10.31$ s；完整 checkpoint 加载 $T_{\text{load}}\!\approx\!35.95$ s；混合恢复 $T_{\text{hybrid}}\!\approx\!28.91$ s；`save-interval`=200（Young/Daly 最优 [38,39]）。10 次 NoFault 运行产生 $\mu_{\text{base}}=4.8543$，$\sigma_{\text{base}}=0.024$ 在 iter-600 评估损失上；我们使用 $\sigma_{\text{base}}$ 来缩放偏差，报告 $\pm1\sigma_{\text{base}}$ 和 $\pm2\sigma_{\text{base}}$ 带。此噪声尺度捕获了集体通信时序、节点/GPU 异构性、共享存储抖动和非确定性内核调度中的残余变异。

**指标。** 原始恢复延迟排除重放，在替换 rank 到达 HEALTHY 时结束；恢复时间（$T_{\text{TTR}}$）在第一次有效恢复后前向传递时结束；完全恢复时间（$T_{\text{TTFR}}$）在优化器状态挂载且更新屏障释放时结束。端到端恢复还包括 Restart 和 MoC-System 的重放；MoEGambit 的混合路径在步 $t$ 恢复，重放成本为零。

**统计处理。** 延迟实验报告 500 次注入恢复事件的均值，因为恢复成本由确定性 I/O 和集体传输段主导；质量实验使用配对比较来控制 seed 和数据顺序噪声。我们使用 NoFault $\sigma_{\text{base}}$ 带作为实际效应大小参考，使用配对非参数检验来评估退化；我们不要求每个单独恢复的运行都在 $\pm1\sigma_{\text{base}}$ 带内。

### 4.2 对比系统

我们将 MoEGambit 与三个 baseline 对比：**NoFault**（无注入故障，参考轨迹）；**Restart**（标准 Megatron-LM checkpoint restart 带完整重载和重放）；**MoC-System**（我们所知唯一先前发表的 MoE 专用容错系统 [23]）。

**MoC-System。** MoC-System 是一个保存侧 MoE 容错系统，通过部分专家检查点（PEC）减少检查点成本：每次检查点仅存储专家的一个选定子集，并在保存间轮换该子集。由于没有公开的参考实现，我们分别处理准确性和计时。对于准确性，我们通过将每个非新鲜专家重定向到其对应的历史检查点分片来复现 MoC-System 论文报告的最佳 PEC 配置（$K_{\text{pec}}=16$，$N=128$，PLT≈3.75%），并禁用 MoEGambit 的混合修复和两阶段协议。对于计时，我们使用论文报告的或可从中推导出的最佳恢复数字；由于 MoC-System 是保存侧的，恢复仍从检查点状态恢复并重放丢失的迭代以到达步 $t$。

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

$\Delta_{\min}$ 和 $\Delta_{\max}$ 在我们的计时下是非绑定的（$T_{\text{load}}>1.2\,T_{\text{hybrid}}$，因此混合恢复对每个 $\Delta\!\geq\!1$ 都有更低延迟）；我们保留两者作为对检查点加载快得多的集群的防御性守卫。

**$\Phi_{\max}=10^{-1}$ 校准。** 我们使用一次守卫禁用的 $3\!\times\!4$ 多 rank 突发扫描（表 5）校准阈值，以识别默认策略应避免的质量退化边界。我们设置 $W_{\mathrm{exp}}=2{,}000$ 以覆盖稳定性实验中使用的 1,000 迭代预热和 1,000 迭代恢复后范围。在此窗口下，$\Phi'(t)\!\leq\!10^{-1}$ 的七个格点均值保持在 baseline $\pm 1\sigma_{\text{base}}$ 带内，而阈值以上的五个格点在始终混合恢复下偏离 $1.21$--$1.98\sigma_{\text{base}}$。默认策略因此将这五个超阈值格点路由到 checkpoint restart。

**表 5：守卫禁用突发扫描：始终混合恢复下的验证损失偏差（$\sigma_{\text{base}}$ 单位）。阴影格点超过 $\Phi_{\max}$ 并在默认策略下触发 restart。**

| $|F|$ | $\Delta\!=\!50$ | $\Delta\!=\!100$ | $\Delta\!=\!150$ | $\Delta\!=\!200$ |
| --- | --- | --- | --- | --- |
| $8$ | $+0.32$ | $+0.64$ | $-0.05$ | $+0.01$ |
| $16$ | $-0.18$ | $+0.71$ | **$+1.42$** | **$+1.46$** |
| $24$ | $+0.59$ | **$+1.21$** | **$+1.98$** | **$+1.95$** |

### 4.4 RQ1：单次故障恢复成本与机制分解

RQ1 测量低延迟恢复路径：MoEGambit 节省多少延迟，以及其两个机制是否独立组合。

我们在预热后注入硬故障，并运行 $2\times 2$ 析因设计，交叉 **(i)** 混合恢复 vs. 完整 checkpoint restart 与 **(ii)** 两阶段 vs. 单阶段优化器挂载。每个格点聚合 500 次注入恢复事件；表 6 报告均值。我们从故障检测到替换 rank 产出第一次恢复后迭代测量延迟。

**表 6：单次故障恢复时间（$2{\times}2$ 析因；每事件均值，秒，排除重放）。**

| 恢复路径 | 单阶段 | 两阶段 | $\Delta$（s） |
| --- | ---: | ---: | ---: |
| 完整 checkpoint | 36.417 | 34.238 | $-2.179$ |
| 混合（选择性） | 30.743 | **28.914** | $-1.829$ |
| $\Delta$ hybrid（s） | $-5.674$ | $-5.324$ | |

**分解。** 混合恢复主效应为 $-5.50$ s（$-15.1\%$），两阶段主效应为 $-2.00$ s（$-5.5\%$），交互为 $0.35$ s（baseline 的 $0.96\%$，在噪声内）。两个机制在此实验中近似加性，联合产生最佳格点 $28.914$ s——相对于完整 checkpoint restart 降低 $20.6\%$。

**结果。** 混合恢复将 NVMe 绑定的全张量恢复替换为对等拉取的非专家状态加一次专家分片读取，将 I/O 从 $O(\text{全局 checkpoint})$ 减少到 $O(|E_{\text{new}}|\!\cdot\!\text{shard})$。两阶段恢复将优化器状态恢复与恢复的前向/反向重叠；在我们的测量中，这种重叠节省了近似常数约 2 s。小交互（$0.96\%$）表明两个机制针对不同的恢复段。

由于 Restart 和 MoC-System 从 checkpoint 状态恢复，两者都重放期望间隔 $\Delta\!=\!100$ 次迭代以到达步 $t$（$T_{\text{replay}}\!\approx\!1031$ s）。使用上述最佳情况加载成本，含重放的成本为 $1067.4$ s；MoEGambit 直接在步 $t$ 恢复，含重放的端到端加速为 $1067.4/28.9=\mathbf{36.9\times}$。我们在全文中分别报告原始延迟和含重放的端到端时间。

### 4.5 RQ2：训练质量与稳定性

RQ2 测试恢复契约的质量面：契约允许的混合恢复是否避免了相对于无故障训练的可检测退化？

**设置。** 每次运行执行 1,000 次预热迭代，注入一次故障，然后与配对 NoFault 和 Restart 运行继续 1,000 次迭代。满足契约的间隔（$\Delta\!\in\!\{64,128\}$）使用混合路径；更大的强制间隔 $\Delta\!\in\!\{256,512,1000,1500\}$ 压力测试每次事件守卫并路由到 restart，除非守卫被故意禁用。

**结果。** 对于满足契约的混合间隔，配对验证损失和 perplexity 相对于 NoFault/Restart 没有可检测的退化。图 3 显示了 10,000 次迭代轨迹（10 次注入故障），图 4 展示了满足契约的 50 次单次故障运行围绕 NoFault 均值的热力图：44/50 落在 $\pm2\sigma_{\text{base}}$ 内，23/50 落在 $\pm1\sigma_{\text{base}}$ 内。10 次配对评估损失差异在 iter 600 上的 Wilcoxon 符号秩检验产生 $p\!=\!0.63$ 和 Cliff's delta $=0.07$，没有退化的证据。我们仅将更大的强制间隔用作守卫压力测试；在默认策略下，这些情况路由到 restart。

由于运行按 seed 配对，标准是保守的：MoEGambit 按 baseline 轨迹的配对偏差来评判，而不是看单次噪声运行是否改善分数。

在所有 10,000 次迭代和注入故障中，MoEGambit 在运行间噪声范围内跟踪 Restart；MoC-System 积累了小的持续间隔，与 PEC 跨恢复边界携带较老专家状态一致。

**下游零样本准确率。** 表 7 报告了 iter 10,000 在八个任务上的 lm-evaluation-harness [44] 结果（ARC-Easy [45]、BoolQ [46]、MathQA [47]、OpenBookQA [48]、PIQA [49]、RACE [50]、SWAG [51]、WinoGrande [52]）。

**表 7：iter $10{,}000$ 的零样本下游准确率（%；lm-evaluation-harness；标准误 ≤ 0.021）。**

| 系统 | ARC-E | BoolQ | MathQA | OBQA | PIQA | RACE | SWAG | WG | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Restart (Megatron) | 46.72 | 59.88 | 22.08 | 29.00 | **68.34** | 29.00 | 55.17 | 50.28 | 45.06 |
| MoC-System [23] | 45.41 | **59.94** | **22.28** | 28.20 | 67.68 | 29.19 | 54.00 | **50.67** | 44.67 |
| MoEGambit | **48.36** | 58.56 | 21.54 | **30.60** | 68.17 | **29.28** | **55.43** | 50.59 | **45.32** |

Restart 和 MoEGambit 在每个任务上一致在 $\pm 1.6$ pp 以内。MoEGambit 在平均上数字更高（$+0.26$ pp），但差距在每 checkpoint 下游噪声范围 [53] 内；我们将其解释为质量对等。小的正值可能反映评估噪声或有界扰动对少数专家子集的温和 dropout 式噪声正则化 [54,55]。这不是改善的证据。MoC-System 达到 $44.67\%$（比 Restart 低 $0.39$ pp），在八个探测中六个落后，与图 3 中的持续损失间隔一致。

### 4.6 RQ3：多次故障与专家加权陈旧度密度

RQ3 验证 R2 规范：$\Phi'(t)\!\leq\!\Phi_{\max}$ 守卫是否在多 rank 突发故障下正确识别质量退化边界？

我们评估了 MoEGambit 在重复和多 rank 突发故障下的表现，并从 §4.3 验证 $\Phi_{\max}=10^{-1}$ 阈值。使用 `find_multi_fault.sh`，我们在与表 5 相同的 $3\!\times\!4$ 网格上扫描 $|F|\!\in\!\{8,16,24\}$ 突发故障跨 $\Delta\!\in\!\{50,100,150,200\}$，并包括在同一 rank 上在 $W_{\mathrm{exp}}$ 内重复访问的重复窗口轨迹。仅间隔或始终混合策略会将这些模式同等对待；$\Phi'(t)$ 通过 $|E_{\text{new}}|$ 和累积窗口债务将它们分开。

**结果。** 在默认策略下，MoEGambit 对 7 个 $\Phi'(t)\!\leq\!10^{-1}$ 的格点使用混合恢复，将 5 个超阈值格点重定向到 checkpoint restart，格点均值损失回到 $\pm 1\sigma_{\text{base}}$ 带内。守卫禁用时，相同的 5 个格点复现了表 5 中的 $1.21$--$1.98\sigma_{\text{base}}$ 偏差。因此，超阈值无退化的缺失来自回退，而专家加权窗口捕获了仅间隔策略遗漏的风险。

此结果也解释了为什么 $\Delta_{\max}$ 本身不够。两个具有相同检查点间隔的事件可能暴露非常不同比例的专家群体，取决于哪些 rank 失败以及最近其他专家何时被修复。$\Phi'(t)$ 通过按间隔和受影响专家集的比例计费每个事件，然后让该计费老化出窗口来使暴露显式。守卫区分了孤立的单 rank 故障和密集突发，即使它们的每次事件间隔相同。

### 4.7 RQ4：无故障开销

**结果。** 经过 1,000 次无故障迭代，MoEGambit 的钩子每迭代增加 $<6$ μs，均值步时间仅增加 $+0.1\%$；配对 $t$ 检验在 $\alpha=0.05$ 下未拒绝零均值差异。

### 4.8 RQ5：可扩展性

我们在 64 和 128 GPU 上重跑 §4.4 的单次故障注入（dense TP=1，PP=8，EP∈{8,16}，ETP=1；每个 10 seeds），并测试了四种 64-GPU 布局覆盖 dense TP∈{1,2}，ETP∈{1,2}，PP∈{4,8}，EP∈{4,8}，EDP∈{1,2}。**结果。** Restart 随全局 checkpoint 大小增长（$36.4$ s → $47.2$ s），而 MoEGambit 增长更慢（$28.9$ s → $33.7$ s），因为混合 I/O 与每 rank 分片成比例。比率从 $1.26\times$ 扩大到 $\mathbf{1.40\times}$，表明在更大 GPU 规模上优势更大。

**并行敏感性。** 在四种 64-GPU 布局上，MoEGambit 以 $1.26$--$3.56\times$ 优于 restart。当 EDP≥2 时优势最大，因为存活专家对等节点允许 MoEGambit 从对等节点恢复非专家和专家状态；即使在 EDP=1，混合恢复仍通过仅从磁盘读取失败 rank 的专家分片更快。

### 4.9 RQ6：跨模型泛化

为评估 Qwen3-30B-A3B 之外的泛化性，我们在 **DeepSeek-V2-Lite** [5]（15.7B 总参 / 2.4B 激活，64 路由专家，2 共享专家，top-6）上重跑核心实验。共享专家是 dense-DP 复制的，使用路径 P；路由专家使用路径 C。策略参数从表 4 继承，$N_{\text{expert}}\!=\!64$。

**结果。** Restart 耗时 $26.09$ s，而 MoEGambit 恢复耗时 $11.73$ s，降低 $55.0\%$（$2.22\times$）。更小的专家分片使路径 C 更快；突发扫描显示相同的 $\Phi'(t)=10^{-1}$ 阈值，通过路径 C 路由共享专家使优势降低 $3.2$ pp。

### 4.10 有效性威胁

遵循标准 SE 有效性类别 [27]，我们总结主要威胁。**内部有效性。** 由于没有公开的 MoC-System 实现，我们使用其最佳报告的 PEC 配置和计时数字，分开报告原始延迟和含重放加速，并在测量 MoEGambit 机制时禁用 PEC。配对 seed、相同数据/故障步和 NoFault $\sigma_{\text{base}}$ 带控制了集体通信、网络、异构性、存储和调度中的残余噪声。**构造有效性。** 我们测量了损失、perplexity 和八个零样本任务；其他用途可能需要额外探测。**外部有效性。** 结果涵盖 H20-3e 互联上的两个 Megatron-LM MoE 配置；其他布局、网络架构、检查点后端、检测器或辅助损失设置可能需要重新校准 $\Phi_{\max}$。

## 5 相关工作

**自适应与自愈系统。** MAPE-K 模型 [24]，架构自适应 [25]，和 SE 路线图 [26,56,57] 主张运行时修复应该是被规范的、守卫的和可审计的。MoEGambit 将此视图应用于 MoE 恢复：$\Phi'(t)$ 控制何时适应，状态类边界决定在哪里，混合恢复定义如何。

**检查点、拓扑适应和对等恢复。** 检查点系统减少 Young/Daly 式 restart 中的成本 $C$ [37,38]，包括 CheckFreq [11]、DeepFreeze [12]、Check-N-Run [13]、Gemini [14]、REFT [15] 和 ByteCheckpoint [16]。这些系统保持 checkpoint 一致的恢复。拓扑适应系统如 Bamboo [17]、Oobleck [18]、ReCycle [19]、Varuna [58]、Parcae [59] 和 Litz [60] 通过重新配置存活 rank 来避免重载，但假设 rank 可互换。FlashRecovery [20] 从对等节点拉取稠密 DP 副本；MoEGambit 将对等恢复应用于 dense-DP 复制的非专家状态，并在 EDP=1 时守卫陈旧的 EP 分片专家。

**MoE 系统和容错。** GShard [29]、Switch [30]、DeepSpeed-MoE [31]、Tutel [32]、MegaBlocks [33]、FasterMoE [34]、SmartMoE [35]，以及开放 MoE 模型 [4,61,6,7,8,9,10] 优化路由、dispatch 和 all-to-all 效率；ST-MoE [41]、sparse-upcycling [42] 和 V-MoE [62] 发展 MoE 架构。它们将专家新鲜度保持隐式。MoC-System [23] 通过部分专家检查点减少保存侧成本但仍从 checkpoint 状态恢复；MoEGambit 针对恢复侧重放、加载成本和运行时质量守卫。

## 6 讨论

**意义和部署。** 可靠的 MoE 恢复应该是状态感知和可审计的。R1-R3 暴露策略谓词、债务 $S(t)$、回退原因和质量轨迹，支持仅审计部署和校准混合恢复。保存侧检查点优化仍然是互补的，因为它们缩小 $\Delta$。

**局限性和未来工作。** MoEGambit 要求失败 rank 的 PP/dense-TP 坐标上有一个健康的 dense-DP 对等节点；关联故障回退到 restart。评估的 $\Phi_{\max}=10^{-1}$ 阈值假设相同的辅助损失尺度、静态专家放置和 fail-stop 故障。动态迁移 [34]、慢降级、静默数据损坏 [63] 和网络分区需要额外的守卫；未来工作将添加专家使用统计和在线噪声地板估计。

## 7 结论

MoEGambit 使分布式 MoE 恢复变为状态感知和可审计的：其运行时契约管控混合修复，将原始恢复延迟降低 $20.6\%$--$55.0\%$，实现 $36.9\times$ 的含重放加速，并对允许的故障无可检测的质量退化。

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
