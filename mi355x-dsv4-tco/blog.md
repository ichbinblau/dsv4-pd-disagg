# 3.5× in Ten Weeks: AMD Instinct™ MI355X Overtakes B200 and B300 on DeepSeek-V4-Pro Agentic Inference Cost

*November 2026*

Our first full DeepSeek-V4-Pro (1.6T, FP4) AgentX run on AMD Instinct MI355X, on Aug 21, peaked at 22.9k tokens/s per GPU. The run on Oct 28, on the same hardware, peaked at 79.8k. There were six public submissions in that window, and from mid-September on we submitted a new one every one to two weeks. At 70 tok/s/user P90 interactivity the difference is larger: 4.4k tokens/s per GPU in August, 71.8k now.

On the public InferenceX dashboard, MI355X now gets more tokens per dollar than B200 and B300 at every interactivity level where both have published results. The lead goes up to 1.85× over B200 and 2.26× over B300. Peak tokens per $1 TCO is 99.1M, 1.50× B200 and 1.20× B300. These numbers use the Rent (3-year commit) cost tier, with data retrieved on 2026-10-30.

The AgentX benchmark and MoRI UMBP, the KV cache layer this work runs on, are described in our [earlier article](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html). Here we focus on the GPU itself, the kernel and scheduling work we did on top of it, and how those changes added up over ten weeks.

![Figure 1](figures/mi355x-progress-aug-oct.png)

*Figure 1: Throughput per GPU vs. P90 interactivity for the MI355X MoRI-SGLang AgentX recipe, screenshot of the [InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) showing the runs on [Aug 21](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32269076444/attempts/4), [Sep 25](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35879254139/attempts/1), [Oct 4](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37181045340/attempts/1) and [Oct 28](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40473082761/attempts/4).*

## Six submissions in ten weeks

Table 1 has every MoRI-SGLang submission from Aug 21 to Oct 28. Each one is a public InferenceX CI run on the same MI355X cluster; between runs we changed only software and the deployment recipe.

*Table 1: MI355X MoRI-SGLang AgentX submissions. Throughput is per GPU. "At 70 tok/s/user" is read off each run's frontier with the dashboard's interpolation. The last column is the same 16 GPUs at concurrency 192.*

| Run | What landed | Peak tok/s/GPU | At 70 tok/s/user | Concurrency 192 |
|---|---|---|---|---|
| Aug 21 | Baseline: SGLang v0.5.17, EAGLE MTP, HiCache, TP8 + TP8 | 22.9k | 4.4k | 22.9k at 23.5 tok/s/user |
| Sep 17 | FP4 sparse-attention indexer, DSpark, MoRI UMBP linker at high concurrency, concurrency-scaled scheduling | 47.9k | 26.3k | 44.8k at 53.6 tok/s/user |
| Sep 25 | MoRI UMBP at every point, TP4 prefill on 12 GPUs, optimistic prefill | 55.8k | 36.5k | 46.6k at 59.3 tok/s/user |
| Oct 4 | MegaMoEv2 prefill, FP8 KV at concurrency 192–512, per-stream split-K | 77.3k | 48.1k | 49.4k at 69.3 tok/s/user |
| Oct 21 | SGLang v0.5.22, FP8 KV at concurrency 1–48, power-of-two decode routing, EPLB | 83.7k | 65.6k | 52.1k at 80.3 tok/s/user |
| Oct 28 | ROCm 10 | 79.8k | 71.8k | 52.1k at 85.1 tok/s/user |

![Figure 2](figures/submission-timeline.png)

*Figure 2: Peak throughput per GPU and throughput at 70 tok/s/user for each submission in Table 1.*

The largest jumps in peak throughput came on Sep 17 (2.1×) and Oct 4 (1.4×), but throughput at 70 tok/s/user went up in every run, by 9–39% per run after Sep 17. The later runs did more for per-user speed than for peak: from Oct 4 to Oct 28, throughput at concurrency 192 rose 5% while its P90 interactivity went from 69.3 to 85.1 tok/s/user. That is also why the Oct 28 peak is below Oct 21's, as explained further down.

Part of the Sep 17 gain is not ours. That run also switched to the DeepSeek-V4-Pro-0813 checkpoint, whose bundled draft head has a higher acceptance rate (golden acceptance length 3.01 instead of 2.49). We measured that at about 8% of throughput at concurrency 192.

## MI355X hardware

*Table 2: Per-GPU specifications from the InferenceX GPU specs page, and the Rent – 3-year commit price used for TCO.*

| | MI355X | B200 | B300 |
|---|---|---|---|
| HBM capacity | 288 GB HBM3E | 180 GB | 268 GB |
| HBM bandwidth | 8 TB/s | 8 TB/s | 8 TB/s |
| Dense FP4 | 10.1 PFLOPS | 9.0 PFLOPS | 13.5 PFLOPS |
| Rent, $/GPU-hour | $2.90 | $3.70 | $4.25 |

Memory capacity drove most of our deployment choices. DeepSeek-V4-Pro in FP4 fits on four MI355X GPUs with space left for KV cache, so the recipe uses TP4 prefill at concurrency 16–48 and TP4 + TP4 at concurrency 1. Every published B200 AgentX recipe for this model uses eight GPUs per role. At concurrency 192–512, about 94% of prompt tokens are served from GPU memory, and MoRI UMBP covers the rest from DRAM.

B300 has about a third more dense FP4 compute than MI355X. At 70 tok/s/user, though, MI355X runs at 71.8k tok/s per GPU, a bit more than B300's 68.8k and 1.45× B200's 49.5k. Much of the kernel work described below went into closing the distance between what the hardware can do and what this model was getting from it.

## Comparison with NVIDIA

A peak number does not say where on the latency curve it was reached, so Table 3 compares systems at fixed interactivity levels. We used the dashboard's interpolation and did not extrapolate.

*Table 3: Total tokens per $1 TCO in millions (Rent tier). "—" means the system has no published point at that interactivity. In parentheses: MI355X divided by that system. For NVIDIA we take the best point across all published frameworks.*

| P90 interactivity | MI355X | B200 | B300 | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| 70 tok/s/user | 89.1 | 48.1 (1.85×) | 58.3 (1.53×) | — | 98.1 (0.91×) |
| 80 tok/s/user | 74.3 | 45.8 (1.62×) | 53.6 (1.39×) | — | 92.5 (0.80×) |
| 100 tok/s/user | 44.8 | 39.1 (1.15×) | 34.8 (1.29×) | 58.2 (0.77×) | 74.1 (0.60×) |
| 120 tok/s/user | 30.2 | 26.9 (1.12×) | 20.2 (1.49×) | 25.3 (1.19×) | 53.0 (0.57×) |
| 150 tok/s/user | 23.2 | 15.9 (1.46×) | 10.3 (2.26×) | 16.4 (1.42×) | 30.9 (0.75×) |
| Peak | 99.1 | 66.1 (1.50×) | 82.7 (1.20×) | 95.0 (1.04×) | 112.8 (0.88×) |

![Figure 3](figures/iso-interactivity-ratio.png)

*Figure 3: MI355X tokens per dollar divided by each NVIDIA system's tokens per dollar at the same P90 interactivity (Rent tier). Above the red line MI355X is cheaper per token.*

B200 and B300 are also 8-GPU servers, and MI355X is ahead of both at every level in the table: 1.12–1.85× B200 and 1.29–2.26× B300 on tokens per dollar. In raw throughput per GPU, MI355X is at most 12% behind either one anywhere in this range, and ahead of both at 70 and 150 tok/s/user. The rest of the tokens-per-dollar lead comes from the lower price. On the Owning cost tier, MI355X still gets 1.67× the tokens per dollar of B200 and 1.57× of B300 at 70 tok/s/user.

GB200 NVL72 and GB300 NVL72 are 72-GPU NVLink racks. Against GB200, MI355X has a slightly higher peak (99.1M vs. 95.0M) and is ahead at 120–150 tok/s/user, while GB200 is ahead around 100 tok/s/user. GB300 is ahead at every level. The closest we get is 0.95× of its tokens per dollar, at 65 tok/s/user.

![Figure 4](figures/tco-frontier-vs-nvidia.png)

*Figure 4: Total tokens per $1 TCO vs. P90 interactivity, Rent – 3-year commit tier, screenshot of the [InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) (data updated 2026-10-29). One framework is shown per system: GB300 NVL72 and B300 on Dynamo SGLang, GB200 NVL72 on Dynamo vLLM, B200 on vLLM, H200 on Dynamo SGLang, and MI355X on MoRI SGLang (red). Table 3 uses each NVIDIA system's best framework, so its B300 and GB300 peaks are higher than the curves shown here.*

## What changed in each run

All of the changes below are in the public SGLang ROCm images (`lmsysorg/sglang-rocm`) and the [InferenceX recipes](https://github.com/SemiAnalysisAI/InferenceX).

### Sep 17

DeepSeek-V4's DSA indexer scores the whole context for every query token at every layer, and it keeps its own KV for each token. We moved the indexer onto AITER FP4 kernels for gfx950, which takes that KV from 132 B to 68 B per token. More sequences fit in HBM, each decode step reads less indexer data, and MoRI has fewer bytes to send from prefill to decode. In the same run we replaced EAGLE with DSpark for speculative decoding (DSpark drafts a block of tokens in one pass), and made admission limits and CUDA-graph ranges scale with concurrency rather than sharing one setting. The MoRI UMBP linker took over KV storage at concurrency 128–256.

### Sep 25

Once MoRI UMBP served every point, prefill no longer needed TP8 just to have room for KV. Concurrency 16–48 moved to TP4 prefill with TP8 decode, 12 GPUs instead of 16, and throughput per GPU rose 26–34%. We also changed how prefill starts. In PD disaggregation a request usually waits for decode to bootstrap it, and at high concurrency that wait is pure queueing. With optimistic prefill, where the request owns its speculative KV, P90 TTFT at concurrency 256 dropped 27.7%. Removing a host sync from DSpark prefill slot expansion took off another 13–16% at concurrency 128–256. We also found that decode had been inheriting the prefill side's consistent-hashing policy, and switched decode to round-robin.

### Oct 4

At concurrency 192–512, prefill now uses AITER's MegaMoEv2 kernel ([sglang#35619](https://github.com/sgl-project/sglang/pull/35619)). It is written in FlyDSL, fuses dispatch, expert GEMM and combine into one kernel, and runs 8-way expert parallelism over the MoRI symmetric heap. We also split the KV pool into an FP8 NoPE part and a BF16 RoPE part. With both changes, throughput per GPU at concurrency 256 rose 11%, and concurrency 384 and 512 became usable; the 77.3k peak is at 512. MLA decode now chooses its split-K factor per index stream, since KV lengths in this model vary by orders of magnitude from layer to layer.

### Oct 21

FP8 KV was extended to concurrency 1–48. Those points have the longest agent sessions, and HBM fills there first. Decode at concurrency 192–512 switched to power-of-two-choices routing so load is spread more evenly across decode DP ranks. MegaMoE prefill at concurrency 512 got expert-parallel load balancing, so the busiest experts stop setting the step time. We also added a concurrency-1 point with eight request slots, which reached 221 tok/s/user.

### Oct 28

The image moved to ROCm 10 (`v0.5.22-rocm10-mi35x-20261026`). P90 interactivity went up at all the high-concurrency points; at concurrency 512 it went from 51.8 to 60.6 tok/s/user, and throughput per GPU went down from 83.7k to 79.8k. Where the Oct 21 and Oct 28 runs cover the same interactivity, the Oct 28 curve is higher at every point.

All of these runs use MoRI for the PD KV transfer and the MoE all-to-all, over standard RoCE between 8-GPU nodes, and MoRI UMBP with the SGLang KVCache Store Linker as a shared, deduplicated DRAM KV pool. The [earlier article](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html) goes through both in detail.

## Next steps

GB300 NVL72 is the system we still trail on this benchmark, and that is where the next round of work is aimed. Every change in this post is upstream or in a public image, and every run can be checked in the public InferenceX CI.

## Acknowledgements

Thanks to the SGLang community for design reviews and quick upstreaming, to SemiAnalysis for AgentX and the InferenceX CI infrastructure, and to the AMD MoRI, SGLang, AITER and ROCm teams.

## References

1. SemiAnalysis InferenceX, DeepSeek-V4 dashboard. https://inferencex.semianalysis.com/inference/deepseek-v4
2. SemiAnalysis AI Cloud TCO Model (source of the $/GPU/hr rates). https://semianalysis.com/ai-cloud-tco-model/
3. AMD, *MoRI UMBP Empowers AMD Instinct™ MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard*. https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html
4. AMD, *Rebuilding Agentic AI from First Principles for AMD GPU*, with Moonshot AI. https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
5. InferenceX recipe history for this benchmark: PRs [#2823](https://github.com/SemiAnalysisAI/InferenceX/pull/2823), [#3188](https://github.com/SemiAnalysisAI/InferenceX/pull/3188), [#3256](https://github.com/SemiAnalysisAI/InferenceX/pull/3256), [#3664](https://github.com/SemiAnalysisAI/InferenceX/pull/3664), [#4186](https://github.com/SemiAnalysisAI/InferenceX/pull/4186), [#4571](https://github.com/SemiAnalysisAI/InferenceX/pull/4571).
