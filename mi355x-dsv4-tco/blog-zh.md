# 十周提速 3.5 倍：AMD Instinct™ MI355X 在 DeepSeek-V4-Pro 智能体推理性价比上超越 B200 与 B300

*2026 年 11 月*

8 月 21 日，我们在 AMD Instinct MI355X 上跑了第一份完整的 DeepSeek-V4-Pro（1.6T，FP4）AgentX，单卡吞吐峰值 22.9k tokens/s。10 月 28 日，同样的硬件跑到了 79.8k。这期间一共有六次公开提交，9 月中旬以后基本一到两周就有一次。如果看 70 tok/s/用户的 P90 交互速度，差距更大：8 月是单卡 4.4k tokens/s，现在是 71.8k。

在公开的 InferenceX dashboard 上，只要是 B200 和 B300 都有公开结果的交互速度档位，MI355X 的每美元 token 产出都比它们高，最多是 B200 的 1.85 倍、B300 的 2.26 倍。每 1 美元 TCO 的 token 数峰值是 9910 万，是 B200 的 1.50 倍、B300 的 1.20 倍。这些数字按 Rent（3 年承诺）成本档位计算，数据获取于 2026-10-30。

AgentX 基准和 MoRI UMBP（这项工作底下的 KV cache 层）在我们的[上一篇文章](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html)里已经介绍过。这一篇主要讲 GPU 本身、我们在上面做的 kernel 和调度工作，以及这些改动在十周里是怎么一点点加起来的。

![图 1](figures/mi355x-progress-aug-oct.png)

*图 1：MI355X MoRI-SGLang AgentX 配方的单卡吞吐 vs. P90 交互速度，[InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) 截图，包含以下几次运行：[8 月 21 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32269076444/attempts/4)、[9 月 25 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35879254139/attempts/1)、[10 月 4 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37181045340/attempts/1)、[10 月 28 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40473082761/attempts/4)。*

## 十周六次提交

表 1 是 8 月 21 日到 10 月 28 日之间 MoRI-SGLang 的全部提交。每一行都是同一套 MI355X 集群上的一次公开 InferenceX CI 运行，两次运行之间只改了软件和部署配方。

*表 1：MI355X MoRI-SGLang AgentX 各次提交。吞吐均为单卡；「70 tok/s/用户下」用 dashboard 的插值算法从该次运行的前沿上读出；最后一列是同样 16 张卡、并发 192 的结果。*

| 运行 | 新增内容 | 单卡峰值 tok/s | 70 tok/s/用户下 | 并发 192 |
|---|---|---|---|---|
| 8 月 21 日 | 基线：SGLang v0.5.17、EAGLE MTP、HiCache、TP8 + TP8 | 22.9k | 4.4k | 22.9k，23.5 tok/s/用户 |
| 9 月 17 日 | FP4 稀疏注意力 indexer、DSpark、高并发启用 MoRI UMBP linker、按并发调整调度参数 | 47.9k | 26.3k | 44.8k，53.6 tok/s/用户 |
| 9 月 25 日 | 所有点启用 MoRI UMBP、12 卡 TP4 prefill、乐观 prefill | 55.8k | 36.5k | 46.6k，59.3 tok/s/用户 |
| 10 月 4 日 | MegaMoEv2 prefill、并发 192–512 用 FP8 KV、逐 stream split-K | 77.3k | 48.1k | 49.4k，69.3 tok/s/用户 |
| 10 月 21 日 | SGLang v0.5.22、并发 1–48 用 FP8 KV、power-of-two decode 路由、EPLB | 83.7k | 65.6k | 52.1k，80.3 tok/s/用户 |
| 10 月 28 日 | ROCm 10 | 79.8k | 71.8k | 52.1k，85.1 tok/s/用户 |

![图 2](figures/submission-timeline.png)

*图 2：表 1 中每次提交的单卡吞吐峰值，以及 70 tok/s/用户下的单卡吞吐。*

峰值涨得最多的是 9 月 17 日（2.1 倍）和 10 月 4 日（1.4 倍）。70 tok/s/用户下的吞吐则是每一次都在涨，9 月 17 日以后每次多 9–39%。后面几次运行对单用户速度的帮助比对峰值大：从 10 月 4 日到 10 月 28 日，并发 192 的吞吐只涨了 5%，P90 交互速度却从 69.3 升到了 85.1 tok/s/用户。10 月 28 日的峰值比 10 月 21 日低，也是这个原因，后面会讲到。

9 月 17 日的提升里有一部分不是我们的功劳。那次运行同时换成了 DeepSeek-V4-Pro-0813 checkpoint，它自带的 draft head 接受率更高（golden acceptance length 从 2.49 升到 3.01）。我们测下来，这部分在并发 192 下大约占 8% 的吞吐。

## MI355X 硬件

*表 2：单卡规格（取自 InferenceX GPU 规格页），以及计算 TCO 用的 Rent – 3 年承诺价格。*

| | MI355X | B200 | B300 |
|---|---|---|---|
| HBM 容量 | 288 GB HBM3E | 180 GB | 268 GB |
| HBM 带宽 | 8 TB/s | 8 TB/s | 8 TB/s |
| 稠密 FP4 算力 | 10.1 PFLOPS | 9.0 PFLOPS | 13.5 PFLOPS |
| Rent 价格（$/GPU/小时） | $2.90 | $3.70 | $4.25 |

我们的部署选择大多取决于显存容量。FP4 的 DeepSeek-V4-Pro 放进四张 MI355X 还能剩下空间给 KV cache，所以配方在并发 16–48 用 TP4 prefill，在并发 1 用 TP4 + TP4。B200 已公开的 AgentX 配方，每个角色都是八张卡。并发 192–512 时，约 94% 的 prompt token 直接从 GPU 显存命中，剩下的由 MoRI UMBP 从 DRAM 补上。

B300 的稠密 FP4 算力比 MI355X 高约三分之一。但在 70 tok/s/用户下，MI355X 单卡跑到 71.8k tok/s，比 B300 的 68.8k 略高，是 B200（49.5k）的 1.45 倍。下面讲的 kernel 工作，很大一部分就是在缩小这个模型实际跑出来的性能和硬件能力之间的差距。

## 与 NVIDIA 对比

峰值数字看不出是在延迟曲线的哪个位置拿到的，所以表 3 在固定的交互速度上比较各系统。插值用的是 dashboard 的算法，没有外推。

*表 3：每 1 美元 TCO 总 token 数（百万，Rent 档位）。「—」表示该系统在这个交互速度没有公开数据点；括号内是 MI355X 与该系统之比。NVIDIA 各系统取所有已公开框架里的最好点。*

| P90 交互速度 | MI355X | B200 | B300 | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| 70 tok/s/用户 | 89.1 | 48.1 (1.85×) | 58.3 (1.53×) | — | 98.1 (0.91×) |
| 80 tok/s/用户 | 74.3 | 45.8 (1.62×) | 53.6 (1.39×) | — | 92.5 (0.80×) |
| 100 tok/s/用户 | 44.8 | 39.1 (1.15×) | 34.8 (1.29×) | 58.2 (0.77×) | 74.1 (0.60×) |
| 120 tok/s/用户 | 30.2 | 26.9 (1.12×) | 20.2 (1.49×) | 25.3 (1.19×) | 53.0 (0.57×) |
| 150 tok/s/用户 | 23.2 | 15.9 (1.46×) | 10.3 (2.26×) | 16.4 (1.42×) | 30.9 (0.75×) |
| 峰值 | 99.1 | 66.1 (1.50×) | 82.7 (1.20×) | 95.0 (1.04×) | 112.8 (0.88×) |

![图 3](figures/iso-interactivity-ratio.png)

*图 3：相同 P90 交互速度下，MI355X 与各 NVIDIA 系统每美元 token 数之比（Rent 档位）。高于红线说明 MI355X 每 token 更便宜。*

B200 和 B300 同样是 8 卡服务器。表中每个档位 MI355X 都比它们强，每美元 token 产出是 B200 的 1.12–1.85 倍、B300 的 1.29–2.26 倍。只看单卡吞吐，在这个范围内 MI355X 最多落后其中任何一个 12%，在 70 和 150 tok/s/用户时比两者都高；剩下的差距来自价格更低。换成 Owning 成本档位，70 tok/s/用户时 MI355X 仍然是 B200 的 1.67 倍、B300 的 1.57 倍。

GB200 NVL72 和 GB300 NVL72 是 72 卡的 NVLink 整机柜。和 GB200 比，MI355X 峰值略高（9910 万对 9500 万），120–150 tok/s/用户也更好，GB200 则在 100 tok/s/用户附近领先。GB300 在每个档位都领先，我们最接近的时候是 65 tok/s/用户，达到它每美元 token 数的 0.95 倍。

![图 4](figures/tco-frontier-vs-nvidia.png)

*图 4：「每 1 美元 TCO 总 token 数」vs.「P90 交互速度」，Rent – 3 年承诺档位，[InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) 截图（数据更新于 2026-10-29）。每个系统只显示一个框架：GB300 NVL72 和 B300 为 Dynamo SGLang，GB200 NVL72 为 Dynamo vLLM，B200 为 vLLM，H200 为 Dynamo SGLang，MI355X 为 MoRI SGLang（红色）。表 3 对每个 NVIDIA 系统取最好的框架，所以表中 B300 和 GB300 的峰值比图中曲线更高。*

## 每次运行改了什么

下面这些改动都在公开的 SGLang ROCm 镜像（`lmsysorg/sglang-rocm`）和 [InferenceX 配方](https://github.com/SemiAnalysisAI/InferenceX)里。

### 9 月 17 日

DeepSeek-V4 的 DSA indexer 在每一层都要给每个 query token 对整个上下文打分，而且每个 token 有自己的一份 KV。我们把 indexer 搬到 gfx950 的 AITER FP4 kernel 上，这份 KV 从每 token 132 B 降到 68 B。HBM 里能放下更多序列，每个 decode step 读的 indexer 数据更少，MoRI 从 prefill 传到 decode 的字节也更少。同一次运行里，投机解码从 EAGLE 换成了 DSpark（一次前向就生成一整块草稿 token）；准入上限和 CUDA graph 范围改成跟着并发走，不再所有点用同一套设置。并发 128–256 的 KV 存储交给了 MoRI UMBP linker。

### 9 月 25 日

所有点都由 MoRI UMBP 服务之后，prefill 不用再为了装 KV 而上 TP8。并发 16–48 改成 TP4 prefill 加 TP8 decode，从 16 卡降到 12 卡，单卡吞吐提升 26–34%。prefill 的启动方式也改了。PD 分离里，请求一般要等 decode 完成 bootstrap 才能开始 prefill，高并发时这段等待纯粹是排队。改成乐观 prefill、由请求自己持有投机 KV 之后，并发 256 的 P90 TTFT 降了 27.7%。再去掉 DSpark prefill slot 扩展里的一次 host 同步，并发 128–256 又降了 13–16%。我们还发现 decode 一直沿用着 prefill 那边的 consistent-hashing 策略，就把 decode 改成了 round-robin。

### 10 月 4 日

并发 192–512 的 prefill 改用 AITER 的 MegaMoEv2 kernel（[sglang#35619](https://github.com/sgl-project/sglang/pull/35619)）。它用 FlyDSL 写成，把 dispatch、专家 GEMM 和 combine 合进一个 kernel，基于 MoRI 对称堆做 8 路专家并行。我们还把 KV 池拆成 FP8 的 NoPE 部分和 BF16 的 RoPE 部分。两项加起来，并发 256 的单卡吞吐提升 11%，并发 384 和 512 也能跑了，77.3k 的峰值就出在并发 512。MLA decode 的 split-K 系数改成按 index stream 分别选，因为这个模型不同层的 KV 长度能差好几个数量级。

### 10 月 21 日

FP8 KV 扩展到了并发 1–48。这几个点的智能体会话最长，HBM 也最先满。并发 192–512 的 decode 改用 power-of-two-choices 路由，让负载在各 decode DP rank 之间分得更匀。并发 512 的 MegaMoE prefill 开了专家并行负载均衡（EPLB），最忙的几个专家不再决定每一步的耗时。另外新加了一个并发 1 的测试点，用 8 个请求槽位，单用户速度到了 221 tok/s/用户。

### 10 月 28 日

镜像换成了 ROCm 10（`v0.5.22-rocm10-mi35x-20261026`）。所有高并发点的 P90 交互速度都上去了，并发 512 从 51.8 升到 60.6 tok/s/用户，单卡吞吐则从 83.7k 降到 79.8k。在 10 月 21 日和 10 月 28 日两次运行都覆盖到的交互速度上，10 月 28 日的曲线每个点都更高。

这些运行都用 MoRI 做 PD 的 KV 传输和 MoE all-to-all，8 卡节点之间走标准 RoCE；KV 池则是 MoRI UMBP 配合 SGLang 的 KVCache Store Linker，在 DRAM 里做成可共享、去重的一层。这两部分的细节见[上一篇文章](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html)。

## 下一步

在这个基准上我们还落后于 GB300 NVL72，下一轮工作主要针对它。本文提到的改动都已经进入上游或公开镜像，每一次运行都可以在公开的 InferenceX CI 里查到。

## 致谢

感谢 SGLang 社区的设计评审和快速合入，感谢 SemiAnalysis 提供 AgentX 和 InferenceX CI 基础设施，也感谢 AMD MoRI、SGLang、AITER 和 ROCm 团队。

## 参考资料

1. SemiAnalysis InferenceX，DeepSeek-V4 dashboard。https://inferencex.semianalysis.com/inference/deepseek-v4
2. SemiAnalysis AI Cloud TCO Model（$/GPU/hr 价格来源）。https://semianalysis.com/ai-cloud-tco-model/
3. AMD，*MoRI UMBP Empowers AMD Instinct™ MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard*。https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html
4. AMD，*Rebuilding Agentic AI from First Principles for AMD GPU*，与 Moonshot AI 联合发布。https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
5. 本基准的 InferenceX 配方演进：PR [#2823](https://github.com/SemiAnalysisAI/InferenceX/pull/2823)、[#3188](https://github.com/SemiAnalysisAI/InferenceX/pull/3188)、[#3256](https://github.com/SemiAnalysisAI/InferenceX/pull/3256)、[#3664](https://github.com/SemiAnalysisAI/InferenceX/pull/3664)、[#4186](https://github.com/SemiAnalysisAI/InferenceX/pull/4186)、[#4571](https://github.com/SemiAnalysisAI/InferenceX/pull/4571)。
