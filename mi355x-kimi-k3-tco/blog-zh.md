# 用一台 AMD Instinct™ MI355X 节点部署 Kimi-K3 2.8T：AgentX 结果

*2026 年 10 月*

部署 Kimi-K3，首先要解决的是放在哪儿。MXFP4 权重约 1.56 TB。MI355X 每卡 288 GB HBM3E，TP8 下每卡约 195 GB，整个模型能放进一台 8 卡节点。8 卡 B200 节点显存不够，目前公开的 B200 和 GB200 配置都是跨两个节点、用流水线并行来跑。

在 SemiAnalysis 的公开基准 AgentX 上，这一台 MI355X 节点每 1 美元 TCO 的总 token 数峰值为 2170 万（Rent 3 年承诺档位，数据获取于 2026-10-30），是 Kimi-K3 dashboard 上最高的峰值。NVIDIA 各系统已公开的最好峰值分别是：B200 1040 万，B300 1060 万，GB200 NVL72 990 万，GB300 NVL72 1910 万。在相同交互速度下比较，25 到 100 tok/s/用户区间内，MI355X 的每美元 token 产出是 B200 的 2.0–2.6 倍。

我们在 MI355X 上的第一份 Kimi-K3 提交是 8 月 12 日，比 NVIDIA 第一份结果晚两周，当时单卡吞吐峰值 3.8k tok/s，现在是 17.5k。

![图 1](figures/tco-frontier-vs-nvidia.png)

*图 1：Kimi-K3 FP4 AgentX，「每 1 美元 TCO 总 token 数」vs.「P90 交互速度」，Rent – 3 年承诺档位。每条曲线是一个系统的 Pareto 前沿，取所有已公开框架里的最好点。红色是一台 8 卡 MI355X 节点（vLLM 或 ATOM）。数据来自 [InferenceX dashboard](https://inferencex.semianalysis.com/inference/kimi-k3)，获取于 2026-10-30。*

## 模型和基准

Kimi-K3 是 2.8T 参数的 MoE，896 个专家，原生支持 1M token 上下文。注意力由两种层混合组成：Kimi Delta Attention（KDA）层为每个请求保存一份循环状态，MLA 层保存分页 KV cache。模型附带一个用于投机解码的 DSpark draft head（Inferact/Kimi-K3-DSpark）。

正是这种混合结构，让它的 agentic 服务比纯 MLA 模型更难。会话进入下一轮时，服务端得同时恢复分页 KV 和 KDA 状态，不然就要把前缀重算一遍，而在 AgentX 里几乎每个请求都是这种情况。AgentX 回放真实的 agentic 编程会话，每个会话约 43 轮，单轮输入中位数 142K token、输出 444 token，超过 96% 的 prompt token 是服务端见过的前缀。

P90 交互速度是完整响应 P90 token 间延迟的倒数，单位 tok/s/用户。每 1 美元 TCO 的 token 数，是单卡总吞吐（含命中缓存的 token）除以 SemiAnalysis AI Cloud TCO Model 给出的每卡每小时成本：Rent 档位下 MI355X $2.90、B200 $3.70、B300 $4.25、GB200 $4.00、GB300 $5.00；Owning 档位下分别是 $1.50、$1.73、$2.26、$1.86、$2.31。

## 成本与吞吐

*表 1：相同 P90 交互速度下每 1 美元 TCO 的总 token 数（百万，Rent 档位），用 dashboard 自己的前沿样条插值，不做外推。「—」表示该系统在这里没有公开数据点；括号内是 MI355X 与该系统之比。*

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

*图 2：相同 P90 交互速度下，MI355X 与各 NVIDIA 系统每美元 token 数之比（Rent 档位）。高于 1.0 说明 MI355X 每 token 更便宜。*

对比 B200，表中整个区间内 MI355X 的每美元 token 产出都是它的 2.0–2.6 倍，其中大部分来自吞吐而不是价格：同一区间 MI355X 的单卡吞吐是 B200 的 1.6–2.0 倍，比如 40 tok/s/用户时 13.3k 对 7.9k tok/s。我们认为主要原因在显存：B200 要把模型拆到两个节点上做流水线并行，MI355X 则全部放在一个节点内。

对比 GB200 NVL72，每美元 token 产出是 1.3–2.1 倍，25–40 tok/s/用户时单卡吞吐也是它的 1.4–1.5 倍。换成 Owning 档位，对 B200 是 1.8–2.3 倍，对 GB200 是 1.2–1.9 倍。

从 40 tok/s/用户往上，GB300 NVL72 领先，MI355X 的每美元 token 数是它的 0.64–0.87 倍。这些 GB300 数据点来自 NVLink 机柜内 24 到 40 卡的宽 EP 分离式部署。MI355X 的峰值更高（2170 万对 1910 万），出现在 20.7 tok/s/用户，低于 GB300 已公开数据的最低交互速度。

每个副本只占一台 8 卡机器，还有一些表里体现不出来的好处：调度更简单，没有跨节点的流水线气泡，故障影响范围也更小。

## 十一周的软件进展

在 MI355X 上，AMD 为 Kimi-K3 维护了 vLLM 和 ATOM 两套服务栈。图 3 是两者在同样 8 张卡上随时间的变化。

![图 3](figures/mi355x-progress-aug-oct.png)

*图 3：MI355X 上 Kimi-K3 FP4 AgentX 的单卡吞吐 vs. P90 交互速度，按提交日期排列。左：vLLM；右：ATOM。*

vLLM 的单卡吞吐峰值从 3.8k 升到 15.9k tok/s；40 tok/s/用户下从 2.1k 升到 13.1k，60 tok/s/用户下从 1.1k 升到 8.8k；并发 1 的单用户速度从 61 升到 212 tok/s/用户。ATOM 的峰值从 4.6k 升到 17.5k tok/s，60 tok/s/用户下从 2.8k 升到 8.5k。

## 做了哪些改动

以下内容都在公开镜像（`vllm/vllm-openai-rocm` nightly 和 `rocm/atom-dev`）以及 [InferenceX 配方](https://github.com/SemiAnalysisAI/InferenceX)里。

Kernel 方面。专家层用 AITER 的 MXFP4 MoE kernel（SITUV2，A4W4 和 A8W4）加 FlyDSL stage-2 FP8。专家 GEMM 之外，支持任意 shape 的低延迟 MoE router GEMM（[vllm#61895](https://github.com/vllm-project/vllm/pull/61895)）、预 shuffle 的 FP4 GEMM 配置，以及 latent-MoE up-projection 只保存本 rank 分片（[vllm#59591](https://github.com/vllm-project/vllm/pull/59591)），减少了 GEMM 以外的时间和显存开销。KDA 的 decode 和投机解码用 AITER Gluon kernel（[vllm#57392](https://github.com/vllm-project/vllm/pull/57392)），分块 prefill 也有对应的 Gluon kernel（[vllm#59552](https://github.com/vllm-project/vllm/pull/59552)），另外还有一个融合的 gfx950 chunk kernel。DSpark draft 的注意力从 Triton MLA 换成 AITER MLA（[vllm#55966](https://github.com/vllm-project/vllm/pull/55966)）后，并发 1 下 P90 交互速度提升 28.9%，单卡吞吐提升 14.5%。

单用户 decode 基本是一串小 kernel 加集合通信，开销主要在 launch 和同步。现在并发 1 时，vLLM 把每个 MoE decoder 层跑成一个常驻的 FlyDSL kernel，TP all-reduce 在 kernel 内完成（[vllm#60683](https://github.com/vllm-project/vllm/pull/60683)）。另外合入了几项 FP8 融合：AttnRes 输出与逐 token FP8 输入量化（[vllm#59069](https://github.com/vllm-project/vllm/pull/59069)），GLA gate GEMM 与 MLA FP8 prefill 的 q/k/v 预处理（[vllm#59634](https://github.com/vllm-project/vllm/pull/59634)），以及 ATOM 里的 FlyDSL FP8 prefill attention。TP all-reduce 在 16 MB 及以上的消息上用 AITER 的 INT4 QuickReduce。

长上下文 decode。上下文到了 100K token 以上，decode 受限于 KV 带宽。decode 上下文并行把 KV 读取分到全部 8 张卡上，用 all-to-all 后端，多 token verify 步骤用 AITER ASM kernel。vLLM 配方现在所有并发都用 DCP8，ATOM 从并发 14 开始用。目标模型和 draft 都用 FP8 KV cache。

KDA 状态与 offload。每个请求的 KDA 状态约 55 MiB。ATOM 一开始把它 offload 到 CPU，后来改成 ReplaySSM，从 GPU 内保存的 checkpoint 环原地重建状态，CPU 预算就可以全部给分页 KV。SSM 状态从 FP32 改成 FP16，又给 KV 池腾出最多 7.8% 的空间。分页 KV 方面，两套栈都在 DRAM 里留了每 rank 约 225 GB 的一层，vLLM 用 SimpleCPUOffloadConnector，ATOM 用 LMCache，每个 rank 的缓存池绑定在自己的 socket 上。

投机解码与调度。DSpark 在并发 1 用 7 个草稿 token，批次变大后逐步减到 5、4、3；ATOM 现在在并发 48–72 也保留 draft 模型。我们在一次运行里发现最慢的 10% decode 都出现在最大的批次里，于是调低了并发 4–32 的运行序列上限，P90 交互速度随之提高。ATOM 加了 batch size 1 的 CUDA graph，因为并发 1 下 92% 的 decode step 原本在重放 batch size 2 的 graph；两套栈的 graph capture 范围也都按 DSpark verify 批大小来设。

## 总结

一台 8 卡 MI355X 节点运行 Kimi-K3 2.8T，每 1 美元 TCO 峰值 2170 万 token，是 Kimi-K3 AgentX dashboard 上最高的。相同交互速度下，每美元 token 产出是 B200 的 2.0–2.6 倍、GB200 NVL72 的 1.3–2.1 倍；从 40 tok/s/用户往上是 GB300 NVL72 领先。每卡 288 GB 让整个模型能放进一台节点，这一点 B200 和 GB200 做不到。从第一份提交到现在，单卡吞吐峰值从 3.8k 升到 17.5k tok/s，全部来自已进入上游或公开镜像的 AITER、FlyDSL、vLLM 和 ATOM 工作。

接下来，我们打算把在 DeepSeek-V4-Pro 上用的 MoRI UMBP（可共享、去重的 DRAM KV 池）用到 Kimi-K3 的分页 KV 加 KDA 状态上，并通过基于 MoRI 的 PD 分离，缩小在更高交互速度下与 GB300 NVL72 的差距。

## 致谢

感谢 vLLM 和 Moonshot AI 社区，感谢 Inferact 提供 DSpark draft 模型，感谢 SemiAnalysis 提供 AgentX 和 InferenceX CI 基础设施，也感谢 AMD AITER、FlyDSL、ATOM、vLLM 和 ROCm 团队。

## 参考资料

1. SemiAnalysis InferenceX，Kimi-K3 dashboard。https://inferencex.semianalysis.com/inference/kimi-k3
2. SemiAnalysis AI Cloud TCO Model（$/GPU/hr 价格来源）。https://semianalysis.com/ai-cloud-tco-model/
3. MI355X Kimi-K3 的 InferenceX 配方演进：vLLM PR [#2508](https://github.com/SemiAnalysisAI/InferenceX/pull/2508)、[#2787](https://github.com/SemiAnalysisAI/InferenceX/pull/2787)、[#3124](https://github.com/SemiAnalysisAI/InferenceX/pull/3124)、[#3833](https://github.com/SemiAnalysisAI/InferenceX/pull/3833)、[#3951](https://github.com/SemiAnalysisAI/InferenceX/pull/3951)、[#4215](https://github.com/SemiAnalysisAI/InferenceX/pull/4215)、[#4393](https://github.com/SemiAnalysisAI/InferenceX/pull/4393)、[#4473](https://github.com/SemiAnalysisAI/InferenceX/pull/4473)；ATOM PR [#2670](https://github.com/SemiAnalysisAI/InferenceX/pull/2670)、[#2716](https://github.com/SemiAnalysisAI/InferenceX/pull/2716)、[#2852](https://github.com/SemiAnalysisAI/InferenceX/pull/2852)、[#3207](https://github.com/SemiAnalysisAI/InferenceX/pull/3207)、[#3407](https://github.com/SemiAnalysisAI/InferenceX/pull/3407)、[#4085](https://github.com/SemiAnalysisAI/InferenceX/pull/4085)。
4. MoRI UMBP 在 DeepSeek-V4 PD 分离上的实践。https://github.com/ichbinblau/dsv4-pd-disagg

*方法说明：数字取自 InferenceX 公开 API（`/api/v1/benchmarks?model=Kimi-K3`），获取于 2026-10-30。数据快照和生成全部图表、表格的脚本在 `data/` 和 `make_figures.py`。相同交互速度下的数值用 InferenceX 的 `iso_interactivity.py` 计算，它移植自 dashboard 的插值算法。*
