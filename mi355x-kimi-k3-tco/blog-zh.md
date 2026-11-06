# 十一周提速 4.6 倍：AMD Instinct™ MI355X 在 Kimi-K3 智能体推理性价比上超越 B200、B300 与 GB200

*2026 年 11 月*

在 SemiAnalysis 公开的 InferenceX dashboard 上，Kimi-K3（2.8T，FP4）的智能体推理测试中，只要是 B200、B300 和 GB200 NVL72 有公开结果的交互速度档位，一台 8 卡 AMD Instinct MI355X 节点的每美元 token 产出都比它们高：是 B200 的 2.0–2.6 倍、B300 的 2.0–3.0 倍、GB200 NVL72 的 1.3–2.1 倍。每 1 美元 TCO 的 token 数峰值是 2170 万，在 Kimi-K3 dashboard 上最高，是 B200 的 2.09 倍、GB300 NVL72 的 1.14 倍。

做到这一步用了十一周（8 月 12 日到 10 月 27 日），硬件没换，全靠软件，MI355X 上最好的单卡吞吐峰值从 3.8k 提升到了 17.5k tokens/s。

本文讲三件事：为什么整个模型能放进一台 MI355X 节点，这台节点和 NVIDIA 各系统比起来怎么样，以及 vLLM 和 ATOM 里撑起这些数字的 kernel 和服务端工作。

![图 1](figures/tco-frontier-vs-nvidia.png)

*图 1：Kimi-K3 FP4 AgentX，「每 1 美元 TCO 总 token 数」vs.「P90 交互速度」，Rent – 3 年承诺档位，按 dashboard 的风格用 InferenceX 公开 API 数据绘制（获取于 2026-10-30）。每条曲线是一个系统已公开框架里最好的那条；MI355X 是一台 8 卡节点，跑 vLLM 或 ATOM。虚线圈表示该点开启了 KV offload。*

## AgentX 与 Kimi-K3 简介

AgentX 是 SemiAnalysis 用真实的 agentic 编程会话做成的基准，在公开的 InferenceX CI 上运行。它回放完整的会话：每一轮把工具输出拼到上下文后面，再整个发给模型。一个会话约 43 轮，单轮输入中位数 142K token、输出 444 token，超过 96% 的 prompt token 是服务端已经处理过的前缀。

Kimi-K3 是 2.8T 参数的 MoE，896 个专家，原生支持 1M token 上下文。它的注意力由两种层混合组成：Kimi Delta Attention（KDA）层为每个请求保存一份循环状态，MLA 层保存分页 KV cache。模型附带一个用于投机解码的 DSpark draft head（Inferact/Kimi-K3-DSpark）。这种混合结构让智能体服务比纯 MLA 模型更难：会话进入下一轮时，服务端得同时恢复分页 KV 和 KDA 状态，不然就要重算整个前缀，而在 AgentX 上几乎每个请求都是这种情况。

## 与 NVIDIA 对比

表 1 在固定的交互速度上比较各系统。每个系统只在少数几个并发设置下测过，结果很少恰好落在某个交互速度上，比如 40 tok/s/用户。表里的数值取自曲线上离它最近的两个实测点之间，取法和 dashboard 画线时一样。如果某个系统在这个交互速度两侧没有实测点，就留空（「—」），不做推测。

*表 1：每 1 美元 TCO 总 token 数（百万，Rent 档位）。「—」表示该系统在这个交互速度没有公开数据点；括号内是 MI355X 与该系统之比。每个系统取所有已公开框架里的最好点。*

| P90 交互速度 | MI355X | B200 | B300¹ | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| 25 tok/s/用户 | 20.8 | 9.9 (2.10×) | 7.1 (2.95×) | 9.8 (2.12×) | — |
| 40 tok/s/用户 | 16.5 | 7.7 (2.15×) | 5.8 (2.83×) | 8.4 (1.96×) | 18.9 (0.87×) |
| 60 tok/s/用户 | 11.0 | 4.9 (2.23×) | 4.6 (2.39×) | 7.1 (1.54×) | 17.2 (0.64×) |
| 80 tok/s/用户 | 8.4 | 3.4 (2.46×) | 3.8 (2.22×) | 6.6 (1.28×) | 12.1 (0.70×) |
| 100 tok/s/用户 | 6.7 | 2.6 (2.59×) | 3.4 (1.97×) | 5.2 (1.30×) | 9.1 (0.74×) |
| 峰值 | 21.7 | 10.4 (2.09×) | 10.6 (2.05×) | 9.9 (2.19×) | 19.1 (1.14×) |

¹ B300 最近一次公开的 Kimi-K3 结果是 2026-08-16，其他系统都是 10 月测的。

![图 2](figures/iso-interactivity-ratio.png)

*图 2：相同 P90 交互速度下，MI355X 与各 NVIDIA 系统每美元 token 数之比（Rent 档位）。高于红线说明 MI355X 每 token 更便宜。*

表中每个档位 MI355X 都比 B200、B300 和 GB200 NVL72 强，每美元 token 产出是 B200 的 2.0–2.6 倍、B300 的 2.0–3.0 倍、GB200 的 1.3–2.1 倍。对 B200 的领先大部分来自吞吐而不是价格：在这个范围内，MI355X 的单卡吞吐是 B200 的 1.6–2.0 倍，比如 40 tok/s/用户时 13.3k 对 7.9k tok/s；在 25–40 tok/s/用户，也是 GB200 的 1.4–1.5 倍。换成 Owning 成本档位，MI355X 的每美元 token 产出仍然是 B200 的 1.8–2.3 倍、GB200 的 1.2–1.9 倍。

从 40 tok/s/用户往上，GB300 NVL72 领先，MI355X 的每美元 token 数是它的 0.64–0.87 倍。这些 GB300 数据点来自 NVLink 机柜内 24 到 40 卡的宽 EP 分离式部署。MI355X 的峰值 2170 万高于 GB300 的 1910 万，出现在 20.7 tok/s/用户，低于 GB300 已公开数据的最低交互速度。

## 十一周里曲线是怎么变化的

我们在 MI355X 上的第一份 Kimi-K3 AgentX 是 8 月 12 日用 vLLM 跑的，比 NVIDIA 的第一份结果晚两周，单卡吞吐峰值 3.8k tokens/s。到 10 月 27 日，vLLM 的峰值是 15.9k。ATOM 从 8 月 21 日的 4.6k 起步，10 月 15 日达到 17.5k。在 40 tok/s/用户的 P90 交互速度下，vLLM 的单卡吞吐从 2.1k 涨到 13.1k tokens/s，单用户速度从 61 升到 212 tok/s/用户。图 3 是两套服务栈各自曲线的变化，图 4 跟踪了全部 18 次提交，其中 vLLM 12 次、ATOM 6 次。每一次都是同一套 MI355X 集群上的公开 InferenceX CI 运行，每个服务实例用一台 8 卡节点。

![图 3](figures/mi355x-progress-aug-oct.png)

*图 3：MI355X 上 Kimi-K3 FP4 AgentX 的单卡吞吐 vs. P90 交互速度，vLLM 和 ATOM 各取四次提交。数据点标签是并行布局；虚线圈表示 KV offload 到了 DRAM 层。*

![图 4](figures/submission-timeline.png)

*图 4：8 月 12 日到 10 月 27 日，MI355X 每次 Kimi-K3 提交的单卡吞吐峰值。*

vLLM 最大的一跳出现在头三周：decode 上下文并行加上 DRAM KV 层，让并发 44–52 能跑起来，峰值从 3.8k 升到 10.2k。之后每次运行，40 tok/s/用户下的吞吐最多再涨 19%。ATOM 在第二次运行（第一次之后六天）峰值就接近翻倍，最后两次运行又各涨了 19–21%。

## MI355X 硬件

*表 2：单卡规格（取自 InferenceX GPU 规格页），以及计算 TCO 用的 Rent – 3 年承诺价格。*

| | MI355X | B200 | B300 | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| HBM 容量 | 288 GB HBM3E | 180 GB | 268 GB | 186 GB | 278 GB |
| HBM 带宽 | 8 TB/s | 8 TB/s | 8 TB/s | 8 TB/s | 8 TB/s |
| 稠密 FP4 算力 | 10.1 PFLOPS | 9.0 PFLOPS | 13.5 PFLOPS | 10.0 PFLOPS | 15.0 PFLOPS |
| Rent 价格（$/GPU/小时） | $2.90 | $3.70 | $4.25 | $4.00 | $5.00 |

这个模型能怎么部署，取决于显存容量。Kimi-K3 的 MXFP4 权重约 1.56 TB。MI355X 上 TP8 时每卡约 195 GB，整个模型放进一台 8 卡节点，还剩空间给 KV cache。八张 B200 只有 1.44 TB，八张 GB200 是 1.49 TB，所以已公开的 B200 和 GB200 配置都把模型拆到两台节点上，用流水线并行来跑。

算力方面，MI355X 的稠密 FP4 只比 B200 略高（10.1 对 9.0 PFLOPS），但在 40 tok/s/用户下单卡吞吐是 B200 的 1.68 倍。下面讲的 kernel 工作，很大一部分就是为了拉开这个差距。

## 关键优化

以下改动都在公开镜像（`vllm/vllm-openai-rocm` nightly 和 `rocm/atom-dev`）以及 [InferenceX 配方](https://github.com/SemiAnalysisAI/InferenceX)里。表 3 列出了实测效果最大的几项，大致按提升从大到小排列。

*表 3：主要优化及实测效果，吞吐均为单卡；「40 tok/s/用户下」从每次运行的前沿上读出。*

| 优化 | 所在组件 | 实测效果 | 上线时间 |
|---|---|---|---|
| decode 上下文并行（DCP8）、DRAM KV 层 | vLLM、ATOM、AITER | vLLM：40 tok/s/用户下吞吐 2.1k → 6.9k，峰值 3.8k → 10.2k；ATOM：峰值 4.6k → 9.0k | 8 月 27 日–9 月 4 日 |
| ReplaySSM、完整 CUDA graph、FP16 SSM 状态 | ATOM | 40 tok/s/用户下吞吐 5.6k → 8.3k（+48%） | 9 月 10 日、9 月 21 日 |
| 高并发保留 DSpark draft、gated RMSNorm prefill | ATOM | 40 tok/s/用户下吞吐 +29%；峰值 14.5k → 17.5k | 10 月 15 日 |
| DSpark draft 改用 AITER MLA | vLLM、AITER | 并发 1 的 P90 交互速度 +28.9%，吞吐 +14.5% | 9 月 17 日 |
| FlyDSL FP8 prefill attention、prefill 延迟调度 | ATOM、FlyDSL | 40 tok/s/用户下吞吐 +24%；峰值 12.2k → 14.5k | 9 月 29 日 |
| AITER Gluon KDA kernel、低延迟 MoE router GEMM | vLLM、AITER | 40 tok/s/用户下吞吐 +19%；峰值 14.9k → 16.0k | 10 月 19 日 |
| 并发 1 使用常驻 FlyDSL decode kernel | vLLM、FlyDSL | 单用户速度 178 → 211 tok/s/用户（+19%） | 10 月 26 日 |
| 兼顾尾部延迟的准入上限 | vLLM | 40 tok/s/用户下吞吐 +15%，60 tok/s/用户下 +17% | 10 月 27 日 |

### decode 上下文并行和 DRAM KV 层

上下文到了 100K token 以上，decode 受限于 KV 带宽。decode 上下文并行把 KV 读取分到全部 8 张卡上，用 all-to-all 后端，多 token verify 步骤用 AITER ASM kernel。HBM 放不下的分页 KV 放到每 rank 约 225 GB 的 DRAM 层，vLLM 用 SimpleCPUOffloadConnector，ATOM 用 LMCache，每个 rank 的缓存池都绑定在自己的 socket 上。这几项合起来，让并发 44–72 能在一台节点上跑起来。vLLM 后来把 DCP8 推到几乎每个点，那次运行让 20 tok/s/用户下的吞吐提升了 26%。

### Kimi Delta Attention 状态

每个请求的 KDA 状态约 55 MiB。ATOM 最初的配方把它和分页 KV 一起 offload 到 CPU。后来改用 ReplaySSM：从 GPU 内保存的 checkpoint 环原地重建状态，CPU 预算就可以全部留给分页 KV。同一次改动还在 DSpark verify 的批大小范围内抓取了完整的 CUDA graph，并把各 rank 绑定到对应的 socket。之后 SSM 状态从 FP32 改成 FP16，给 KV 池腾出最多 7.8% 的空间；新增的 batch size 1 CUDA graph 让单序列的 decode step 不再重放 batch size 2 的 graph，而并发 1 下 92% 的 step 原本都是这种情况。这两次改动下来，40 tok/s/用户下的吞吐从 5.6k 升到 8.3k。

### 投机解码

两套服务栈都用 DSpark draft head，草稿长度按并发设定：并发 1 用 7 个 token，批次变大后逐步减少。vLLM 里，把 draft 的注意力从 Triton MLA 换成 AITER MLA（[vllm#55966](https://github.com/vllm-project/vllm/pull/55966)）后，并发 1 的 P90 交互速度提升 28.9%，吞吐提升 14.5%。ATOM 现在在并发 48–72 也保留 4 个 token 的 draft（之前这几个点不用 draft），并为 prefill 加了 gated RMSNorm 流水线（[ROCm/ATOM#2458](https://github.com/ROCm/ATOM/pull/2458)）。这次改动让 ATOM 在 40 tok/s/用户下的吞吐提升 29%，峰值从 14.5k 升到 17.5k。

### AITER 和 FlyDSL kernel

专家层用 AITER 的 MXFP4 MoE kernel（SITUV2，A4W4 和 A8W4）加 FlyDSL stage-2 FP8。在此之外，vLLM 加入了支持任意 shape 的低延迟 MoE router GEMM（[vllm#61895](https://github.com/vllm-project/vllm/pull/61895)）、预 shuffle 的 FP4 GEMM 配置，以及只保存本 rank 分片的 latent-MoE up-projection（[vllm#59591](https://github.com/vllm-project/vllm/pull/59591)）。KDA 的 decode 和投机解码用 AITER Gluon kernel（[vllm#57392](https://github.com/vllm-project/vllm/pull/57392)），分块 prefill 也有对应的 Gluon kernel（[vllm#59552](https://github.com/vllm-project/vllm/pull/59552)）。带来 Gluon KDA decode kernel 和 router GEMM 的那个 vLLM 镜像，让 40 tok/s/用户下的吞吐提升了 19%。

ATOM 里，FlyDSL FP8 prefill attention 加上在并发 16 及以上让就绪的 prefill 等四个 decode 步再执行，40 tok/s/用户下的吞吐提升了 24%。单用户 decode 方面，vLLM 在并发 1 时把每个 MoE decoder 层跑成一个常驻的 FlyDSL kernel，TP all-reduce 在 kernel 内完成（[vllm#60683](https://github.com/vllm-project/vllm/pull/60683)），单用户速度从 178 升到 211 tok/s/用户。

### 准入控制

在一次 vLLM 运行里，最慢的 10% decode step 全都出现在最大的批次中。把并发 4–32 的运行序列数上限压到这个点以下之后，40 tok/s/用户下的吞吐提升 15%，60 tok/s/用户下提升 17%。

## 致谢

感谢 vLLM 和 Moonshot AI 社区，感谢 Inferact 提供 DSpark draft 模型，感谢 SemiAnalysis 提供 AgentX 和 InferenceX CI 基础设施，也感谢 AMD AITER、FlyDSL、ATOM、vLLM 和 ROCm 团队。

## 参考资料

1. SemiAnalysis InferenceX，Kimi-K3 dashboard。https://inferencex.semianalysis.com/inference/kimi-k3
2. SemiAnalysis AI Cloud TCO Model（$/GPU/hr 价格来源）。https://semianalysis.com/ai-cloud-tco-model/
3. AMD，*MoRI UMBP Empowers AMD Instinct™ MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard*（AgentX 基准的更多介绍）。https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html
4. MI355X Kimi-K3 的 InferenceX 配方演进：vLLM PR [#2508](https://github.com/SemiAnalysisAI/InferenceX/pull/2508)、[#2787](https://github.com/SemiAnalysisAI/InferenceX/pull/2787)、[#2810](https://github.com/SemiAnalysisAI/InferenceX/pull/2810)、[#3124](https://github.com/SemiAnalysisAI/InferenceX/pull/3124)、[#3833](https://github.com/SemiAnalysisAI/InferenceX/pull/3833)、[#3951](https://github.com/SemiAnalysisAI/InferenceX/pull/3951)、[#4215](https://github.com/SemiAnalysisAI/InferenceX/pull/4215)、[#4393](https://github.com/SemiAnalysisAI/InferenceX/pull/4393)、[#4473](https://github.com/SemiAnalysisAI/InferenceX/pull/4473)；ATOM PR [#2670](https://github.com/SemiAnalysisAI/InferenceX/pull/2670)、[#2716](https://github.com/SemiAnalysisAI/InferenceX/pull/2716)、[#2852](https://github.com/SemiAnalysisAI/InferenceX/pull/2852)、[#3207](https://github.com/SemiAnalysisAI/InferenceX/pull/3207)、[#3407](https://github.com/SemiAnalysisAI/InferenceX/pull/3407)、[#4085](https://github.com/SemiAnalysisAI/InferenceX/pull/4085)。
