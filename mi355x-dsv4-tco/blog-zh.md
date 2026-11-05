# 十周提速 3.5 倍：AMD Instinct™ MI355X 在 DeepSeek-V4-Pro 智能体推理性价比上超越 B200 与 B300

*2026 年 11 月*

8 月 21 日，我们在 AMD Instinct MI355X 上提交了第一份完整的 DeepSeek-V4-Pro（1.6T，FP4）AgentX 结果，单卡吞吐峰值 22.9k tokens/s。10 月 28 日，同样的硬件峰值达到 79.8k。算上首尾，这期间一共有六次公开提交；9 月中旬以后，每一到两周就有一次新提交，而且每一次都把曲线往前推了一步。在 70 tok/s/用户的 P90 交互速度下，提升更大：单卡吞吐从 4.4k 涨到 71.8k tokens/s。

这样的迭代速度，改变了 MI355X 相对 NVIDIA 的位置。在公开的 InferenceX dashboard 上，凡是 B200 和 B300 有公开数据的交互速度档位，MI355X 的每美元 token 产出都更高，最高达到 B200 的 1.85 倍、B300 的 2.26 倍；每 1 美元 TCO 的 token 数峰值为 9910 万，是 B200 的 1.50 倍、B300 的 1.20 倍（Rent – 3 年承诺档位，数据获取于 2026-10-30）。

我们的[上一篇文章](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html)介绍了 AgentX 基准和 MoRI UMBP，也就是这项工作底层的 KV cache 层。这一篇讲的是 GPU 本身、我们在其上做的 kernel 和调度优化，以及这些优化在多短时间内累积成了现在的结果。

![图 1](figures/mi355x-progress-aug-oct.png)

*图 1：MI355X MoRI-SGLang AgentX 配方的单卡吞吐 vs. P90 交互速度，[InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) 截图，包含以下几次运行：[8 月 21 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32269076444/attempts/4)、[9 月 25 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35879254139/attempts/1)、[10 月 4 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37181045340/attempts/1)、[10 月 28 日](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40473082761/attempts/4)。*

## 十周六次提交

表 1 列出了 8 月 21 日到 10 月 28 日之间 MoRI-SGLang 的每一次提交。每一行都是同一套 MI355X 集群上的一次公开 InferenceX CI 运行，变的只有软件和部署配方。

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

有两点值得说。第一，提升不是靠某一项改动。最大的两步是 9 月 17 日（峰值 2.1 倍）和 10 月 4 日（峰值 1.4 倍），但每一次运行都提高了 70 tok/s/用户下的吞吐，9 月 17 日之后每一步又各自多出 9–39%。第二，后几次运行主要是用峰值换单用户速度。从 10 月 4 日到 10 月 28 日，并发 192 的吞吐只涨了 5%，P90 交互速度却从 69.3 升到 85.1 tok/s/用户。10 月 28 日的峰值低于 10 月 21 日，也是同样的原因（后文详述）。

关于 9 月 17 日那一步要说明一点：这次运行同时换成了 DeepSeek-V4-Pro-0813 checkpoint，它自带的 draft head 接受率更高（golden acceptance length 从 2.49 升到 3.01）。按我们的测量，这部分在并发 192 下贡献了约 8% 的吞吐，其余都来自我们的软件。

## MI355X 给 DeepSeek-V4-Pro 带来了什么

*表 2：单卡规格（取自 InferenceX GPU 规格页），以及计算 TCO 用的 Rent – 3 年承诺价格。*

| | MI355X | B200 | B300 |
|---|---|---|---|
| HBM 容量 | 288 GB HBM3E | 180 GB | 268 GB |
| HBM 带宽 | 8 TB/s | 8 TB/s | 8 TB/s |
| 稠密 FP4 算力 | 10.1 PFLOPS | 9.0 PFLOPS | 13.5 PFLOPS |
| Rent 价格（$/GPU/小时） | $2.90 | $3.70 | $4.25 |

这项工作里的大多数部署选择都是由显存容量决定的。FP4 的 DeepSeek-V4-Pro 放进四张 MI355X 还有余量留给 KV cache，所以我们的配方在并发 16–48 用 TP4 prefill，并发 1 用 TP4 + TP4，而不必每个角色都用八张卡。B200 已公开的 AgentX 配方，每个角色都用八张卡。大显存还让并发 192–512 下约 94% 的 prompt token 留在 GPU 显存里，MoRI UMBP 只需要从 DRAM 补上剩下的部分。

另一点是算力。B300 的稠密 FP4 算力比 MI355X 高约三分之一，但在 70 tok/s/用户下，MI355X 的单卡吞吐反而略高于 B300（71.8k 对 68.8k），是 B200（49.5k）的 1.45 倍。下文的 kernel 工作，主要就是让这个模型跑得更接近硬件上限。

## 相同交互速度下与 NVIDIA 对比

只比峰值，看不出各系统是在延迟曲线的哪个位置拿到的数字。表 3 在几个固定交互速度上读取各系统前沿的数值，用的是 dashboard 的插值算法，不做外推。

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

和同为 8 卡服务器的 B200、B300 相比，MI355X 在每个档位都领先。单卡吞吐上，它最多只落后这两者 12%，在 70 和 150 tok/s/用户时则领先两者；加上更低的价格，每美元 token 产出领先 B200 1.12–1.85 倍、领先 B300 1.29–2.26 倍。换成 Owning 成本档位，优势依然成立：70 tok/s/用户时是 B200 的 1.67 倍、B300 的 1.57 倍。

GB200 NVL72 和 GB300 NVL72 是 72 卡的 NVLink 整机柜。MI355X 的峰值略高于 GB200（9910 万对 9500 万），在 120–150 tok/s/用户也领先，GB200 则在 100 tok/s/用户附近更好。GB300 NVL72 在所有档位都领先；我们最接近的是高吞吐一端，65 tok/s/用户时达到它每美元 token 数的 0.95 倍。

![图 4](figures/tco-frontier-vs-nvidia.png)

*图 4：「每 1 美元 TCO 总 token 数」vs.「P90 交互速度」，Rent – 3 年承诺档位，[InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) 截图（数据更新于 2026-10-29）。每个系统只显示一个框架：GB300 NVL72 和 B300 为 Dynamo SGLang，GB200 NVL72 为 Dynamo vLLM，B200 为 vLLM，H200 为 Dynamo SGLang，MI355X 为 MoRI SGLang（红色）。表 3 对每个 NVIDIA 系统取最好的框架，所以表中 B300 和 GB300 的峰值比图中曲线更高。*

## 逐次运行的优化

以下改动都在公开的 SGLang ROCm 镜像（`lmsysorg/sglang-rocm`）和 [InferenceX 配方](https://github.com/SemiAnalysisAI/InferenceX)里。

**9 月 17 日：降低注意力的开销。** DeepSeek-V4 的 DSA indexer 在每一层都要给每个 query token 对整个上下文打分，还有自己的逐 token KV。我们把它换到 gfx950 的 AITER FP4 kernel 上，这部分 KV 从每 token 132 B 降到 68 B。HBM 能多放序列，每个 decode step 的 indexer 带宽更少，MoRI 在 prefill 和 decode 之间要传的数据也更少。同一次运行把投机解码从 EAGLE 换成了 DSpark，一次前向生成一整块草稿 token；准入上限和 CUDA graph 范围也改成随并发调整，不再所有点共用一套设置。并发 128–256 的 KV 层交给了 MoRI UMBP linker。

**9 月 25 日：每个副本用更少的卡，排队更少。** 所有点都由 MoRI UMBP 服务之后，prefill 不再需要靠 TP8 来装 KV，并发 16–48 改成 12 卡的 TP4 prefill + TP8 decode，单卡吞吐提升 26–34%。PD 分离里，请求通常要等 decode 完成 bootstrap 才开始 prefill；让 prefill 乐观先行、投机 KV 归请求自己持有之后，并发 256 下 P90 TTFT 降低 27.7%。再去掉 DSpark prefill slot 扩展里的一次 host 同步，并发 128–256 又降了 13–16%。我们还发现 decode 一直悄悄沿用了 prefill 路由的 consistent-hashing 策略，于是改成了 round-robin。

**10 月 4 日：高并发下的 MoE 和 KV 精度。** 并发 192–512 的 prefill 改用 AITER 的 MegaMoEv2 kernel（[sglang#35619](https://github.com/sgl-project/sglang/pull/35619)），用 FlyDSL 写成，把 dispatch、专家 GEMM 和 combine 融合在一起，基于 MoRI 对称堆做 8 路专家并行。KV 池拆成 FP8 的 NoPE 部分和 BF16 的 RoPE 部分。两项合起来让并发 256 的单卡吞吐提升 11%，也打开了并发 384 和 512，77.3k 的峰值就是在这里拿到的。MLA decode 的 split-K 系数改为按 index stream 分别选择，因为这个模型不同层的 KV 长度能差好几个数量级。

**10 月 21 日：照顾低并发和高并发两端。** FP8 KV 扩展到并发 1–48，这里的智能体会话最长，HBM 最先被占满。并发 192–512 的 decode 改用 power-of-two-choices 路由，让各 decode DP rank 的负载更均衡；并发 512 的 MegaMoE prefill 开启专家并行负载均衡（EPLB），最热的专家不再决定每一步的耗时。新增的并发 1 测试点使用 8 个请求槽位，把单用户速度提高到 221 tok/s/用户。

**10 月 28 日：ROCm 10。** 镜像换成 ROCm 10（`v0.5.22-rocm10-mi35x-20261026`）后，所有高并发点的 P90 交互速度都有提升。并发 512 从 51.8 升到 60.6 tok/s/用户，单卡吞吐则从 83.7k 降到 79.8k。在两次运行都覆盖的交互速度范围内，10 月 28 日的曲线处处高于 10 月 21 日。

在这些改动之下，MoRI 负责 PD 的 KV 传输和 MoE all-to-all，8 卡节点之间走标准 RoCE；MoRI UMBP 配合 SGLang 的 KVCache Store Linker，提供可共享、去重的 DRAM KV 池。这两部分在[上一篇文章](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html)里有详细介绍。

## 总结

十周、六次公开提交，同样的硬件上，DeepSeek-V4-Pro 在 MI355X 上的单卡吞吐峰值从 22.9k 升到 79.8k tokens/s，70 tok/s/用户下从 4.4k 升到 71.8k。在 B200 和 B300 都有数据的每个交互速度档位上，MI355X 的每美元 token 产出都领先，最高达到 1.85 倍和 2.26 倍；峰值是 B200 的 1.50 倍、B300 的 1.20 倍。GB300 NVL72 仍然领先，这是我们接下来要追的差距。

每卡 288 GB 显存让每个角色可以用更少的卡来跑这个模型，AITER、SGLang、MoRI 和 ROCm 的一次次发布则把这些余量变成了吞吐。所有改动都已进入上游或公开镜像，每一次运行都在公开的 InferenceX CI 里。

## 致谢

感谢 SGLang 社区的设计评审和快速合入，感谢 SemiAnalysis 提供 AgentX 和 InferenceX CI 基础设施，也感谢 AMD MoRI、SGLang、AITER 和 ROCm 团队。

## 参考资料

1. SemiAnalysis InferenceX，DeepSeek-V4 dashboard。https://inferencex.semianalysis.com/inference/deepseek-v4
2. SemiAnalysis AI Cloud TCO Model（$/GPU/hr 价格来源）。https://semianalysis.com/ai-cloud-tco-model/
3. AMD，*MoRI UMBP Empowers AMD Instinct™ MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard*。https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html
4. AMD，*Rebuilding Agentic AI from First Principles for AMD GPU*，与 Moonshot AI 联合发布。https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
5. 本基准的 InferenceX 配方演进：PR [#2823](https://github.com/SemiAnalysisAI/InferenceX/pull/2823)、[#3188](https://github.com/SemiAnalysisAI/InferenceX/pull/3188)、[#3256](https://github.com/SemiAnalysisAI/InferenceX/pull/3256)、[#3664](https://github.com/SemiAnalysisAI/InferenceX/pull/3664)、[#4186](https://github.com/SemiAnalysisAI/InferenceX/pull/4186)、[#4571](https://github.com/SemiAnalysisAI/InferenceX/pull/4571)。
