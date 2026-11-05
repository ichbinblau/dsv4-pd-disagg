# 3.5× in Ten Weeks: AMD Instinct™ MI355X Overtakes B200 and B300 on DeepSeek-V4-Pro Agentic Inference Cost

*November 2026*

On Aug 21, our first full DeepSeek-V4-Pro (1.6T, FP4) AgentX submission on AMD Instinct MI355X peaked at 22.9k tokens/s per GPU. On Oct 28, the same hardware peaked at 79.8k. Counting both ends, that is six public submissions; after mid-September a new one landed every one to two weeks, and each of them moved the curve. At 70 tok/s/user P90 interactivity, the gain is larger still: from 4.4k to 71.8k tokens/s per GPU.

That pace changed where MI355X sits against NVIDIA. On the public InferenceX dashboard, MI355X now produces more tokens per dollar than B200 and B300 at every interactivity level both systems have published, up to 1.85× B200 and 2.26× B300, and its peak of 99.1M tokens per $1 TCO is 1.50× B200 and 1.20× B300 (Rent – 3-year commit tier, data retrieved 2026-10-30).

Our [earlier article](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html) covered the AgentX benchmark and MoRI UMBP, the KV cache layer under this work. This post is about the GPU, the kernels and scheduling changes we made on top of it, and how fast those changes added up.

![Figure 1](figures/mi355x-progress-aug-oct.png)

*Figure 1: Throughput per GPU vs. P90 interactivity for the MI355X MoRI-SGLang AgentX recipe, screenshot of the [InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) showing the runs on [Aug 21](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32269076444/attempts/4), [Sep 25](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35879254139/attempts/1), [Oct 4](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37181045340/attempts/1) and [Oct 28](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40473082761/attempts/4).*

## Six submissions in ten weeks

Table 1 lists every MoRI-SGLang submission between Aug 21 and Oct 28. Each row is a public InferenceX CI run on the same MI355X cluster. Only the software and the deployment recipe changed.

*Table 1: MI355X MoRI-SGLang AgentX submissions. Throughput is per GPU; "at 70 tok/s/user" is read off that run's frontier with the dashboard's interpolation. The last column is the same 16 GPUs at concurrency 192.*

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

Two things stand out. First, no single change accounts for the gain. The biggest steps were Sep 17 (2.1× peak) and Oct 4 (1.4× peak), but every run raised throughput at 70 tok/s/user, and after Sep 17 each one added another 9–39%. Second, the later runs mostly traded raw peak for speed per user. Between Oct 4 and Oct 28, concurrency 192 throughput went up only 5%, but its P90 interactivity went from 69.3 to 85.1 tok/s/user. The Oct 28 peak is lower than Oct 21 for the same reason (more on that below).

One caveat on the Sep 17 jump: that run also moved to the DeepSeek-V4-Pro-0813 checkpoint, whose bundled draft head is accepted more often (golden acceptance length 3.01, up from 2.49). In our measurements that is worth about 8% of throughput at concurrency 192. The rest of the step is our software.

## What MI355X brings to DeepSeek-V4-Pro

*Table 2: Per-GPU specifications from the InferenceX GPU specs page, and the Rent – 3-year commit price used for TCO.*

| | MI355X | B200 | B300 |
|---|---|---|---|
| HBM capacity | 288 GB HBM3E | 180 GB | 268 GB |
| HBM bandwidth | 8 TB/s | 8 TB/s | 8 TB/s |
| Dense FP4 | 10.1 PFLOPS | 9.0 PFLOPS | 13.5 PFLOPS |
| Rent, $/GPU-hour | $2.90 | $3.70 | $4.25 |

Memory capacity shaped most of the deployment choices in this work. DeepSeek-V4-Pro in FP4 fits on four MI355X GPUs with room left for KV cache, so our recipe runs TP4 prefill at concurrency 16–48 and TP4 + TP4 at concurrency 1, instead of using eight GPUs per role. Every published B200 AgentX recipe for this model uses eight GPUs per role. The large HBM also keeps about 94% of prompt tokens in GPU memory at concurrency 192–512, so MoRI UMBP only needs to serve the remainder from DRAM.

The other point is compute. B300 has about a third more dense FP4 than MI355X. Yet at 70 tok/s/user MI355X produces slightly more tokens per GPU than B300 (71.8k vs. 68.8k) and 1.45× B200 (49.5k). Most of the kernel work below was about getting closer to the hardware's limits on this model.

## Against NVIDIA at the same interactivity

Comparing peaks hides where on the latency curve each system gets its number, so Table 3 reads each system's frontier at fixed interactivity levels, using the dashboard's interpolation and no extrapolation.

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

Against B200 and B300, the other 8-GPU servers, MI355X is ahead at every level. Per GPU, it is never more than 12% behind either system in throughput, and it is ahead of both at 70 and 150 tok/s/user; the lower price turns that into a tokens-per-dollar lead of 1.12–1.85× over B200 and 1.29–2.26× over B300. On the Owning cost tier the lead holds: 1.67× B200 and 1.57× B300 at 70 tok/s/user.

GB200 NVL72 and GB300 NVL72 are 72-GPU NVLink racks. MI355X's peak is slightly above GB200's (99.1M vs. 95.0M) and ahead at 120–150 tok/s/user, while GB200 leads around 100 tok/s/user. GB300 NVL72 is ahead at every level; we are closest at the high-throughput end, at 0.95× its tokens per dollar at 65 tok/s/user.

![Figure 4](figures/tco-frontier-vs-nvidia.png)

*Figure 4: Total tokens per $1 TCO vs. P90 interactivity, Rent – 3-year commit tier, screenshot of the [InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4) (data updated 2026-10-29). One framework is shown per system: GB300 NVL72 and B300 on Dynamo SGLang, GB200 NVL72 on Dynamo vLLM, B200 on vLLM, H200 on Dynamo SGLang, and MI355X on MoRI SGLang (red). Table 3 uses each NVIDIA system's best framework, so its B300 and GB300 peaks are higher than the curves shown here.*

## The optimizations, run by run

Everything below is in the public SGLang ROCm images (`lmsysorg/sglang-rocm`) and the [InferenceX recipes](https://github.com/SemiAnalysisAI/InferenceX).

**Sep 17: making the model cheap to attend over.** DeepSeek-V4's DSA indexer scores the full context for every query token at every layer and keeps its own per-token KV. We moved it onto AITER FP4 kernels for gfx950, which cuts that KV from 132 B to 68 B per token. That means more sequences in HBM, less indexer bandwidth per decode step, and fewer bytes for MoRI to move between prefill and decode. The same run switched speculative decoding from EAGLE to DSpark, which emits a block of draft tokens in one pass, and scaled admission limits and CUDA-graph ranges with concurrency instead of using one setting for all points. The MoRI UMBP linker took over the KV tier at concurrency 128–256.

**Sep 25: fewer GPUs per replica, less queueing.** With MoRI UMBP serving every point, prefill no longer needed TP8 just to hold KV, so concurrency 16–48 moved to TP4 prefill + TP8 decode on 12 GPUs, with 26–34% more throughput per GPU. In PD disaggregation a request normally waits for decode to bootstrap it before prefill starts; letting prefill start optimistically, with the speculative KV owned by the request, cut P90 TTFT by 27.7% at concurrency 256. Removing a host sync from DSpark prefill slot expansion cut another 13–16% at concurrency 128–256. We also found decode was silently inheriting the prefill router's consistent-hashing policy and switched it to round-robin.

**Oct 4: MoE and KV precision at high concurrency.** Prefill at concurrency 192–512 now runs AITER's MegaMoEv2 kernel ([sglang#35619](https://github.com/sgl-project/sglang/pull/35619)), written in FlyDSL, which fuses dispatch, expert GEMM and combine and runs 8-way expert parallelism over the MoRI symmetric heap. The KV pool was split into an FP8 NoPE part and a BF16 RoPE part. Together these added 11% throughput per GPU at concurrency 256 and opened up concurrency 384 and 512, which is where the 77.3k peak comes from. MLA decode now picks its split-K factor per index stream, since layers in this model have KV lengths that differ by orders of magnitude.

**Oct 21: the low- and high-concurrency ends.** FP8 KV extended down to concurrency 1–48, where agent sessions are longest and HBM fills first. Decode at concurrency 192–512 moved to power-of-two-choices routing, which evens out load across decode DP ranks, and MegaMoE prefill at concurrency 512 turned on expert-parallel load balancing so the hottest experts no longer set the step time. A new concurrency-1 point with eight request slots raised single-user speed to 221 tok/s/user.

**Oct 28: ROCm 10.** Moving the image to ROCm 10 (`v0.5.22-rocm10-mi35x-20261026`) raised P90 interactivity at every high-concurrency point. At concurrency 512 it went from 51.8 to 60.6 tok/s/user, while throughput per GPU dropped from 83.7k to 79.8k. Over the interactivity range both runs cover, the Oct 28 curve is above Oct 21 at every point.

Underneath all of this, MoRI carries both the PD KV transfer and the MoE all-to-all over standard RoCE between 8-GPU nodes, and MoRI UMBP with the SGLang KVCache Store Linker provides a shared, deduplicated DRAM KV pool. Both are covered in detail in the [earlier article](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html).

## Summary

In ten weeks and six public submissions, DeepSeek-V4-Pro on MI355X went from 22.9k to 79.8k peak tokens/s per GPU and from 4.4k to 71.8k at 70 tok/s/user, on unchanged hardware. MI355X now leads B200 and B300 on tokens per dollar at every interactivity level they share, by up to 1.85× and 2.26×, with peak tokens per dollar 1.50× B200 and 1.20× B300. GB300 NVL72 is still ahead, and the gap is what we are working on next.

The 288 GB of HBM per GPU let us run the model on fewer GPUs per role, and AITER, SGLang, MoRI and ROCm releases turned that headroom into throughput. All of it is upstream or in public images, and every run is in the public InferenceX CI.

## Acknowledgements

Thanks to the SGLang community for design reviews and quick upstreaming, to SemiAnalysis for AgentX and the InferenceX CI infrastructure, and to the AMD MoRI, SGLang, AITER and ROCm teams.

## References

1. SemiAnalysis InferenceX, DeepSeek-V4 dashboard. https://inferencex.semianalysis.com/inference/deepseek-v4
2. SemiAnalysis AI Cloud TCO Model (source of the $/GPU/hr rates). https://semianalysis.com/ai-cloud-tco-model/
3. AMD, *MoRI UMBP Empowers AMD Instinct™ MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard*. https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html
4. AMD, *Rebuilding Agentic AI from First Principles for AMD GPU*, with Moonshot AI. https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
5. InferenceX recipe history for this benchmark: PRs [#2823](https://github.com/SemiAnalysisAI/InferenceX/pull/2823), [#3188](https://github.com/SemiAnalysisAI/InferenceX/pull/3188), [#3256](https://github.com/SemiAnalysisAI/InferenceX/pull/3256), [#3664](https://github.com/SemiAnalysisAI/InferenceX/pull/3664), [#4186](https://github.com/SemiAnalysisAI/InferenceX/pull/4186), [#4571](https://github.com/SemiAnalysisAI/InferenceX/pull/4571).
