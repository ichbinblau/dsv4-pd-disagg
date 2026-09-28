# UMBP 让 AMD Instinct™ MI355X 在公开权威的 AgentX 排行榜上击败 NVIDIA B200 Dynamo SGLang，取得 1.5 倍于对手的每美元 TCO 产出

*2026 年 9 月*

一个 agentic（智能体）编程会话要跑几十轮，而每一轮发过来的上下文，跟上一轮几乎一模一样。智能体发来的 prompt token 里，超过 96% 是服务端早就算过的。**因此，服务这类负载的成本，取决于服务端能把多少 KV cache 重新找回来，避免重算。**

MoRI **UMBP**（Unified Memory & Bandwidth Pool，统一内存与带宽池）就是用来让这份缓存始终找得回来的。UMBP 是 AMD MoRI 团队从 agentic 场景出发、遵循第一性原理、专为 AMD 平台打造的 KV cache 基础设施；我们把这套成果贡献回 SGLang 社区，作为全新 KVCache Store Linker 的后端接入开源生态，让更广泛的生态用户都能受益。在 SemiAnalysis 的 AgentX 基准、DeepSeek-V4-Pro-0813 1.6T 模型上，MI355X 跑 UMBP + MoRI + SGLang 达到 **峰值每 1 美元 TCO 产出 6900 万 token，而 B200 跑 Dynamo SGLang 是 4600 万 —— 每美元 token 产出领先 1.5×**（SemiAnalysis Rent / 三年承诺 成本档位，B200 为 $3.7/chip/hr，MI355X 为 $2.9/chip/hr，数据截至 2026-09-25）。

![图 1](figures/fig1_tco_vs_b200.png)

*图 1：DeepSeek-V4-Pro-0813 1.6T 在 agentic 场景下的「每 1 美元 TCO 总 token 数」vs.「P90 交互速度」，取自公开的 SemiAnalysis InferenceX dashboard（Rent / 三年承诺 档位，更新于 2026-09-25）。红色是 MI355X FP4 + UMBP + MoRI + SGLang；绿色系是 NVIDIA 阵营 —— B200、B300、H200、GB200、GB300 NVL72，以及一条 Vera Rubin NVL72 的预览曲线。越靠右上越好，标签是各点的并行布局。对比 B200 FP4（Dynamo SGLang），MI355X 的峰值是每 1 美元 TCO 6900 万 token，B200 为 4600 万。*

读过我们今年 7 月与 Moonshot AI 联合发表的 [*Rebuilding Agentic AI from First Principles for AMD GPU*](https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html) 的读者，会认出这里的论点：**agentic 服务里决定成败的资源是 KV cache** —— 你能吃下多少复用上限、缓存溢出 HBM 之后存在哪里，以及调度器能否比重算更快地路由到它。那篇文章给出的答案就是 UMBP，并测得在累计命中率几乎不变的前提下，P99 TTFT 缩小 **3.2×**。

**这篇文章把同一套栈拿去和竞品对比**：在公开的第三方排行榜上，对手是 NVIDIA B200。

下面讲 UMBP 如何做到这个结果。其余优化（FP4 indexer、DP attention 重构、乐观 prefill）在靠后一节集中概括：它们同样重要，但 UMBP 才是把 TCO 曲线推得最远的那一块。

## AgentX 是什么，以及它在测什么

AgentX 是 SemiAnalysis 用真实的 agentic 编程 trace 构建的公开基准，通过公开的 InferenceX 测试框架和 dashboard 运行。它正逐步成为智能体推理评估领域的重要行业标准，并已在全球范围内获得广泛采用，使用者包括 OpenAI、Meta、Inferact、RadixArk、MiniMax、阿里 Qwen、月之暗面、智谱 GLM、Oracle 等领先企业与团队。它的跑分口径、结果和 CI 全部公开，因此**本文里的每一个数字都可以被独立复核，包括被 NVIDIA 复核**。

AgentX 回放的是完整的智能体会话：每一轮把工具调用的结果追加到已积累的上下文里，再去问一次模型。这样产生的流量有四个特征，是聊天基准不会有的。

- **长时间、多轮会话** —— 每个会话大约 43 轮，整个生命周期约有 100 万 token 的上下文流量。
- **长输入、短输出** —— 每轮中位数 142K 输入 token，对应 444 输出 token。
- **大量前缀复用** —— 超过 96% 的 prompt token 是服务端已经见过的前缀的重复。
- **子智能体（subagent）和工具调用**，会把一个用户任务扇出成好几条共享前缀的并发会话。

它上报 TTFT、P90 交互速度（单用户 tok/s）、TPGS（每 GPU 秒总 token 数，含命中缓存的 token）和 TCO（每 token 的基础设施成本）—— 最后这个就是图 1 画的东西。

由此有两个结论。在 96% 前缀复用下，*决定 prefill 开销的是缓存管理*。另外，因为智能体要等完整回复，真正要紧的是端到端延迟、由 decode 主导 —— 把机器多分一点给 prefill 并不能买到余量，**只能把这部分 prefill 计算彻底省掉**。

## 这套负载打破了什么，我们就用什么去接

7 月那篇文章已经列过一张表，把 agentic 流量打破的「聊天时代假设」逐条摊开。其中这三条，是本文这套栈必须正面回答的：

| 特点 | 它打破了哪个假设 |
|---|---|
| 同时有很多条长会话，还有子智能体 | “KV 缓存能装进 HBM。”并发稍微一高，大部分可复用的 KV 早就被从显存里赶出去了。 |
| DeepSeek-V4 稀疏注意力（DSA indexer + MLA latent KV） | “KV 缓存是按 TP rank 切开的。”MLA latent 和 indexer KV 是在每个 rank 上*复制*的。 |
| 引擎重启和滚动升级 | “缓存热度是免费的。”住在引擎进程里的 host 缓存，进程一死就跟着没了。 |

每一条都对应栈里的一层。所有结果都基于 MI355X 上的 SGLang PD 分离：prefill 和 decode 跑在不同节点，KV 通过 RDMA 在两边之间搬运。

| 层次 | 组件 | 在这次工作里的角色 |
|---|---|---|
| 服务框架 | SGLang（ROCm 版 v0.5.17 → v0.5.20） | PD 分离、DP attention、MTP/DSpark 投机解码 |
| KV 存储 | **MoRI UMBP** | 分布式、去重的 DRAM KV 池，通过 KVCache Store Linker 挂到 SGLang 的 radix tree 上 |
| KV 传输 | MoRI-IO | prefill→decode 的 KV RDMA 传输 |
| 算子 | AITER | FP4 MoE GEMM、FP4 稀疏注意力 indexer、MLA decode |
| 平台 | ROCm 7.2，AMD Instinct MI355X | 每节点 8 卡，FP4 权重 |

基准是 InferenceX 的 `agentic-coding` 场景，并开启 DRAM KV offload（`dram-utilization: 0.80`）。它作为 InferenceX CI 的一部分持续运行。

## MoRI UMBP + SGLang KVCache Store Linker

UMBP 的设计在 7 月那篇文章的 "What is UMBP" 一节里有完整交代：一个横跨 engine HBM → host DRAM → UMBP DRAM 池 → SSD 的统一逻辑缓存，建立在三条核心原则之上 —— **面向 agentic 推理的原生设计、与调度器/框架/编排层的协同设计、以及对 AMD 硬件的亲和性**。三条原则服务于同一个目标：**被 offload 出去的前缀依然是「可路由的」**，这样 router 就能依据放置位置和取回代价，挑出返回最快的那个副本。

这一节讲的是我们把这套设计接到生产级推理引擎后面、并把并发推上去之后的发现：瓶颈出在**接入方式本身**。

### HiCache 这条路的问题出在哪

在 7 月那篇文章里，UMBP 是按框架当时提供的方式接入 SGLang 的：作为 **HiCache 的 L3 存储后端**，注册名为 `mori`，用 `--hicache-storage-backend mori` 启用。这条路是能跑通的，上面引用的那些数字也正是这么跑出来的。但在 AgentX 上把并发继续推高后，它就达不到硬件所允许的水平了，瓶颈出在 UMBP 与引擎之间的 HiCache 这一层。我们找到了六个问题，而且都是结构性的：

- **浪费了可共享的 DRAM。** 不可共享的 HiCache 占着 DRAM，而这些 DRAM 本可以给可共享的 L3 后端用，等于压缩了有效共享容量。
- **数据通路绕远。** HiCache 夹在 L1 HBM 和 L3 后端中间，既增加了 load/offload 开销，又堵死了 L1 ⇔ L3 的直连通路。
- **只能按 rank 本地管理。** HiCache 是嵌在每个 rank 里的，所有 KV 决策只基于本地信息，从全局看并不最优。
- **冗余复制。** 在 MLA + TP 下，每个 TP rank 都复制一份 KV cache，各自独立地 load/offload，白白浪费内存和 PCIe 带宽。
- **没有逐层流水。** HiCache 会把 L2→L1 的加载按层重叠起来，但从外部 L3 后端抓数据进 L2 这一段没有流水化 —— 必须全部完成，计算才能开始。
- **缓存跟着引擎一起死。** HiCache 活在引擎进程里，只要重启一次，host 侧缓存就全没了。

这一点在基准上直接体现了出来：上 UMBP 之前的那一步，并发 192 的 P90 TTFT 飙到 35.3 秒，同时 GPU KV 使用率钉在 94% —— host 层已经撑满，请求全堵在重算后面排队。

### KVCache Store Linker

我们向 SGLang maintainer 提了一个方案：干脆绕开 HiCache，在 L1 HBM 和外部 KV cache store 之间走一条直连数据通路 —— 结果发现这个方向和社区自己的规划高度一致。随后 MoRI 团队与 SGLang 社区共同设计了 **KVCache Store Linker**，把 UMBP 作为一等公民后端集成进去。linker 把 SGLang 的统一 radix tree 直接连到分布式 DRAM 池上；前缀命中时，prefill 直接从 DRAM 拉 KV page，而不是重算。

Linker + UMBP 把上面六个问题全部解决了：

- **DRAM 完全可共享**，跨 DP rank、跨模型实例都可共享 —— 从而支持 DP + round-robin 的部署方式（靠跨 DP rank 的 KV cache 共享）。
- **L1 ⇔ L3 直连通路**，中间零开销，单这一项就比 HiCache 路径把 TTFT 改善最多 13%。
- **全局 KV 管理** —— UMBP 基于全局信息做放置和淘汰，比按 rank 的本地策略更有效。
- **去重 + 按 rank 拆分 load/offload**，缓解内存和 PCIe 带宽压力。TP-N 的 prefill 现在对复制的 MLA/DSA KV 只存/取一份而不是 N 份，TP8 下 8 个 key 合成 1 个；这等于把 DRAM 有效容量翻了 N 倍，host 流量也按同样倍数下降。
- **逐层流水加载**，UMBP 通过批处理、层分组、ranged API，以及为 host→device KV 加载优化过的 GPU gather kernel，把逐层带来的额外请求开销藏了起来。
- **缓存能挺过引擎重启** —— 在 UMBP standalone 模式下，KV 池住在每节点一个的独立进程里，重启和升级之后可以直接复用，无需预热。

### 结果

在 AgentX agentic-coding 场景上端到端测量，用 UMBP linker 替换 HiCache：

| 并发 | 单卡吞吐 | P90 TTFT |
|---|---|---|
| 192 | **+14%** | **–66%**（35.3 秒 → 11.9 秒） |
| 256 | +9.7% | –51% |

TTFT 的下降来自在一个 96% 前缀复用的负载上**不再重算前缀**，没有任何 kernel 改动参与其中。

**缓存够大，拓扑就能改。** 有了这层去重的 DRAM 之后，prefill 不用单纯为了装下 KV 而硬上 TP8。在并发 16–48 时，9 月 23 日的 recipe 改用 **TP4 prefill + TP8 decode（12 卡）**，替代 TP8 + TP8（16 卡）。并发 16/32/48 时，UMBP 分别供上了 30%、52%、75% 的 prompt token，需要重算的不到 3%。用少 25% 的卡，单卡吞吐反而提升 24–34% —— 这正是图 1 里 TCO 差距的重要来源之一。

![图 2](figures/fig3_umbp_tp4_prefill.png)

*图 2：左：在 UMBP linker 下，TP4 prefill 的每个 prompt token 的 KV 来自哪里 —— GPU HBM 前缀缓存、UMBP DRAM 层，还是重算。并发越高、HBM 驱逐越多，差额就由 DRAM 层补上。右：9 月 15 日 recipe（TP8 prefill + TP8 decode，16 卡）与 9 月 23 日 recipe（TP4 prefill + UMBP + TP8 decode，12 卡）的单卡吞吐对比。9 月 23 日的分支还包含乐观 prefill 和 SGLang v0.5.20 镜像；并发 16 另外用了 DSpark γ=6。*

代价是 prefill 变慢：prefill 的卡减半后，并发 16–48 的 P90 TTFT 上升 26–53%（2.2–3.6 秒 → 2.7–5.6 秒）。对于智能体要等完整回复、TTFT 只占其中一小部分的场景，我们愿意用这个代价，换在固定单用户 decode 速度下更高的单卡吞吐。

### 上游进展

linker 以及周边的 KV cache 基础设施，都在与 SGLang 社区一起公开建设：

**KVCache Store Linker** —— 为 Unified Radix Cache 增加可选的 external-cache linker 模式（`hzh0425`）；UMBP external linker，[#37578](https://github.com/sgl-project/sglang/pull/37578)（`maning00`）；UMBP 直连中对复制 MLA/DSA KV 的去重，[#38778](https://github.com/sgl-project/sglang/pull/38778)（`TianDi101`）；external linker 的 KV 加载失败改为优雅处理而不是让 scheduler 崩溃（`TianDi101`）；external-cache linker 的混合 Mamba 支持（`isytwu`）。

**KVCache Indexer** —— 进程内本地 KV indexer 以及与 Router 的集成（`wuyl1`）；在 `BlockStored` 上加 `component_types` 字段，用于按组件追踪放置位置。

**Scheduler & Router** —— bucket-aware 的策略域与原生 cache indexing（`Bo-Vincent`）；可组合的打分与准入策略，[#37731](https://github.com/sgl-project/sglang/pull/37731)（`Bo-Vincent`）。

## 其余优化，简要说明

8 月 21 日到 9 月 23 日之间，另外三项 AMD 自研改动也进了同一个 MI355X recipe；它们和 UMBP 合起来把单卡吞吐翻了一倍（见「总结」）。

- **FP4 稀疏注意力 indexer**（[sglang#37353](https://github.com/sgl-project/sglang/pull/37353)）。DeepSeek-V4 的 DSA indexer 要对每个 query token、在每一层给整段上下文打分，而且自带一份 per-token KV。把它跑在 gfx950 的 AITER FP4 kernel 上，这份 KV 从**每 token 132 B 降到 68 B** —— HBM 能装下更多并发序列，每步 decode 的 indexer 带宽更少，走 MoRI 链路的字节也更少。
- **为 PD 分离重新设计 DP attention**（[InferenceX#2823](https://github.com/SemiAnalysisAI/InferenceX/pull/2823)）。attention 在 8 个 rank 上做数据并行，而专家权重改成**按 TP 切分（EP1）**、不再用 EP8 分布，从而把 all-to-all 的 dispatch/combine 和专家路由的负载不均从关键路径上拿掉。调优开关按角色（prefill / decode）分别生效，`max-running-requests` 也改成随基准并发伸缩、不再用固定上限。这一项与 FP4 indexer、v0.5.18 镜像打包在一起，在并发 192 时是**单卡吞吐 +51%**；随并发伸缩的调度再加 **+7%**，P90 TTFT 从 33.6 秒降到 16.2 秒。
- **乐观 prefill + 请求自持的投机 KV**（[sglang#38978](https://github.com/sgl-project/sglang/pull/38978)、[sglang#40111](https://github.com/sgl-project/sglang/pull/40111)）。在 PD 分离里，请求通常要等 decode 侧 bootstrap 完才能开始 prefill；高并发下这次握手纯粹就是排队时间。让 prefill 乐观地先开跑、投机 KV 归请求自己所有而不是挂在预留的 decode 槽位上，**并发 256 时 P90 TTFT 降低 27.7%**。再去掉 DSpark prefill 槽位扩展里的一次 host 同步，并发 128–256 的 P90 TTFT 又降了 **13–16%**。
- **MLA decode 的 per-stream split-K**（[sglang#39968](https://github.com/sgl-project/sglang/pull/39968)）：按 index stream 单独选 `kv_splits`，而不是用一套配置去套 KV 长度相差好几个数量级的所有层。

![图 3](figures/fig1_pareto.png)

*图 3：整轮优化中的单卡吞吐 vs. P90 交互速度。8 月 21 日基线在所有点上都用 16 卡（1P1D，TP8 + TP8）；优化后的 recipe 在不同并发下分别用 8、12 或 16 卡，吞吐按单卡归一化。*

## 这些数字怎么看

我们希望这些结果可复现、归因也尽量公平：

- **TCO 对比的口径。** 图 1 来自公开的 InferenceX dashboard，Rent / 三年承诺 成本档位，B200 为 $3.7/chip/hr、MI355X 为 $2.9/chip/hr，数据更新于 2026-09-25。1.5× 这个数字比的是**峰值**每 1 美元 TCO 的 token 产出：MI355X 6900 万，B200 4600 万。在中段、交互速度对齐的情况下优势要小一些；在高交互速度端，B200 的曲线反而在前面。请按你自己实际的服务目标去选对比点。
- **本文对比的对象是 B200。** 图 1 上还画了 B300、GB200、GB300 NVL72 以及一条 Vera Rubin NVL72 的预览曲线，其中部分曲线位于 MI355X 之上。它们属于更新或更大整机规格的产品，单芯片 TCO 也更高（$4.25–$8.5/chip/hr，MI355X 为 $2.9）；本文给出的结论只针对 B200（Dynamo SGLang），不对其余型号作任何声明。
- **投机解码的接受率是模拟的。** InferenceX 对每个 checkpoint 把接受长度（AL）固定在一个参考值（`SGLANG_SIMULATE_ACC_LEN`），这样所有 run 都在同一接受率下比较。换到 DeepSeek-V4-Pro-0813 checkpoint 后，参考 AL 从 2.49 提到 3.01（MTP-3 / DSpark γ=3），这部分大约贡献了**并发 192 下 9% 的吞吐提升**，它不属于软件优化。9 月 23 日的 recipe 在并发 4 和 16 时跑 DSpark γ=6（AL 3.77）。
- **不同 recipe 用的卡数不同。** 8 月 21 日基线在所有并发下都用 16 卡；优化后的 recipe 在并发 4 用 8 卡、并发 16–48 用 12 卡、并发 128 及以上用 16 卡。所有吞吐都按单卡报告。
- **有些特性是打包测的。** 当一次 recipe 更新把几个特性和镜像升级混在一起时，我们直接报合并收益，而不去猜各自占多少；单项特性的数字来自各自上游 PR 里的 A/B 测量。
- **run 间波动。** 同一配置重复跑，大多数点的吞吐波动在 ±2% 以内（并发 48 最高到 11%）。

## 总结

| 指标 | 8 月 21 日 | 9 月 23 日 | 变化 |
|---|---|---|---|
| 单卡吞吐，并发 192 | 22.9k tok/s | 46.6k tok/s | **2.03×** |
| P90 TTFT，并发 192 | 33.6 s | 11.8 s | **–65%** |
| P90 交互速度，并发 192 | 23.5 tok/s/用户 | 59.3 tok/s/用户 | **2.5×** |
| 单卡吞吐峰值 | 22.9k（并发 192） | 55.8k（并发 256） | **2.4×** |
| 每 1 美元 TCO 的 token 峰值产出 | — | 6900 万（MI355X）vs. 4600 万（B200） | **1.5×** |

kernel 和并行布局依然重要。但当 96% 的 prompt token 都是算过的，**决定这些 token 存在哪里的那个系统，就决定了你的每 token 成本**。7 月我们从第一性原理论证了这一点，并在文末留下一份路线图：完成 UMBP 的集成，并把这套原语带到更多推理引擎上。KVCache Store Linker 交付了第一项：UMBP 现在直接接进引擎的 radix tree，成为一等 KV 基础设施。上面这些 AgentX 结果，就是这套设计在公开基础设施上与竞品对比的实测值。

下一项路线图：把 UMBP 带进 vLLM 生态。

## 致谢

感谢 SGLang 社区的设计评审和快速合入，感谢 SemiAnalysis 提供 AgentX 基准和 InferenceX CI 基础设施，以及 AMD AITER、MoRI 和 ROCm 团队。

## 参考

1. SemiAnalysis — AgentX 基准与 InferenceX dashboard. https://inferencex.semianalysis.com
2. vLLM — AgentX. https://vllm.ai/blog/2026-09-08-vllm-agentx
3. AMD — Rebuilding Agentic AI from First Principles for AMD GPU，与 Moonshot AI 合作（“What is UMBP”）. https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
4. sgl-project/sglang#37578 — [Unified Cache][6/N] Add UMBP external linker. https://github.com/sgl-project/sglang/pull/37578
5. sgl-project/sglang#38778 — [Unified Cache] Dedup replicated MLA/DSA KV in the UMBP direct linker. https://github.com/sgl-project/sglang/pull/38778
6. sgl-project/sglang#37731 — [Router] Add composable scoring and eligibility policies. https://github.com/sgl-project/sglang/pull/37731
7. sgl-project/sglang#37353 — [AMD] Enable FP4 indexer for DeepSeek V4. https://github.com/sgl-project/sglang/pull/37353
8. sgl-project/sglang#38978 — Reduce decode bootstrap latency with request-owned speculative KV. https://github.com/sgl-project/sglang/pull/38978
9. sgl-project/sglang#40111 — Avoid host sync in DSpark prefill slot expansion. https://github.com/sgl-project/sglang/pull/40111
10. sgl-project/sglang#39968 — [AMD] dsv4: pick kv_splits per index stream. https://github.com/sgl-project/sglang/pull/39968
11. SemiAnalysisAI/InferenceX#2823 — DeepSeek-V4 MI355X agentic PD-disaggregation recipe update. https://github.com/SemiAnalysisAI/InferenceX/pull/2823
12. SemiAnalysisAI/InferenceX#3256 — DeepSeek-V4 MI355X UMBP + DSpark recipe. https://github.com/SemiAnalysisAI/InferenceX/pull/3256
