# 4.6× in Eleven Weeks: AMD Instinct™ MI355X Leads B200, B300 and GB200 on Kimi-K3 Agentic Inference Cost

*November 2026*

On SemiAnalysis's public InferenceX dashboard for Kimi-K3 (2.8T, FP4) agentic inference, a single 8-GPU AMD Instinct MI355X node now gets more tokens per dollar than B200, B300 and GB200 NVL72 at every interactivity level where they have published results: 2.0–2.6× B200, 2.0–3.0× B300 and 1.3–2.1× GB200 NVL72. Its peak of 21.7M tokens per $1 TCO is the highest on the Kimi-K3 dashboard, 2.09× B200 and 1.14× GB300 NVL72.

We got here in eleven weeks, from Aug 12 to Oct 27, with software work on the same hardware; over that time the best peak throughput per GPU on MI355X went from 3.8k to 17.5k tokens/s.

This post covers why the whole model fits on one MI355X node, how that node compares with NVIDIA's systems, and the kernel and serving work in vLLM and ATOM behind the numbers.

![Figure 1](figures/tco-frontier-vs-nvidia.png)

*Figure 1: Kimi-K3 FP4 AgentX, total tokens per $1 TCO vs. P90 interactivity, Rent – 3-year commit tier, drawn in the dashboard's style from the InferenceX public API (retrieved 2026-10-30). Each curve is one system's best published framework; MI355X is one 8-GPU node running vLLM or ATOM. Dashed circles mark points with KV offload on.*

## AgentX and Kimi-K3 in brief

AgentX is a SemiAnalysis benchmark built from real agentic coding sessions and run through the public InferenceX CI. It replays whole sessions: each turn adds tool output to the context and sends everything back to the model. A session runs about 43 turns, the median turn has 142K input tokens and 444 output tokens, and over 96% of prompt tokens repeat a prefix the server has already processed.

Kimi-K3 is a 2.8T-parameter MoE with 896 experts and a 1M-token native context. Its attention mixes two kinds of layers: Kimi Delta Attention (KDA) layers keep a recurrent state for each request, and MLA layers keep a paged KV cache. It ships with a DSpark draft head (Inferact/Kimi-K3-DSpark) for speculative decoding. The mix makes agentic serving harder than for a pure-MLA model. When a session comes back for its next turn, the server has to restore both the paged KV and the KDA state, or recompute the prefix, and on AgentX that happens on almost every request.

## Comparison with NVIDIA

Table 1 compares systems at fixed interactivity levels. Each system was measured at only a handful of concurrency settings, so its results rarely land exactly on a level such as 40 tok/s/user. We take the value from the curve between the two nearest measured points, the same way the dashboard draws its lines. When a system has no measurements on both sides of a level, we leave the cell empty ("—") instead of guessing.

*Table 1: Total tokens per $1 TCO in millions (Rent tier). "—" means the system has no published point at that interactivity. In parentheses: MI355X divided by that system. For each system we take the best point across all published frameworks.*

| P90 interactivity | MI355X | B200 | B300¹ | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| 25 tok/s/user | 20.8 | 9.9 (2.10×) | 7.1 (2.95×) | 9.8 (2.12×) | — |
| 40 tok/s/user | 16.5 | 7.7 (2.15×) | 5.8 (2.83×) | 8.4 (1.96×) | 18.9 (0.87×) |
| 60 tok/s/user | 11.0 | 4.9 (2.23×) | 4.6 (2.39×) | 7.1 (1.54×) | 17.2 (0.64×) |
| 80 tok/s/user | 8.4 | 3.4 (2.46×) | 3.8 (2.22×) | 6.6 (1.28×) | 12.1 (0.70×) |
| 100 tok/s/user | 6.7 | 2.6 (2.59×) | 3.4 (1.97×) | 5.2 (1.30×) | 9.1 (0.74×) |
| Peak | 21.7 | 10.4 (2.09×) | 10.6 (2.05×) | 9.9 (2.19×) | 19.1 (1.14×) |

¹ The most recent published B300 Kimi-K3 results are from 2026-08-16. The other systems were measured in October.

![Figure 2](figures/iso-interactivity-ratio.png)

*Figure 2: MI355X tokens per dollar divided by each NVIDIA system's tokens per dollar at the same P90 interactivity (Rent tier). Above the red line MI355X is cheaper per token.*

MI355X is ahead of B200, B300 and GB200 NVL72 at every level in the table: 2.0–2.6× B200, 2.0–3.0× B300 and 1.3–2.1× GB200 on tokens per dollar. Most of the lead over B200 comes from throughput rather than price. Per GPU, MI355X produces 1.6–2.0× B200's throughput across this range, for example 13.3k vs. 7.9k tok/s at 40 tok/s/user, and 1.4–1.5× GB200's at 25–40 tok/s/user. On the Owning cost tier, MI355X still gets 1.8–2.3× the tokens per dollar of B200 and 1.2–1.9× of GB200.

GB300 NVL72 is ahead from 40 tok/s/user up, with MI355X at 0.64–0.87× of its tokens per dollar. Those GB300 points come from wide-EP disaggregated deployments of 24 to 40 GPUs inside the NVLink rack. MI355X's peak of 21.7M is higher than GB300's 19.1M, and it is reached at 20.7 tok/s/user, below the lowest interactivity GB300 has published.

## How the curve evolved in eleven weeks

Our first Kimi-K3 AgentX run on MI355X, with vLLM on Aug 12, came two weeks after the first NVIDIA result and peaked at 3.8k tokens/s per GPU. By Oct 27 vLLM peaked at 15.9k. ATOM started on Aug 21 at 4.6k and peaked at 17.5k on Oct 15. At 40 tok/s/user P90 interactivity, vLLM went from 2.1k to 13.1k tokens/s per GPU, and its single-user speed from 61 to 212 tok/s/user. Figure 3 shows how each stack's curve moved, and Figure 4 tracks all 18 submissions, 12 with vLLM and 6 with ATOM. Each one is a public InferenceX CI run on the same MI355X cluster, one 8-GPU node per server.

![Figure 3](figures/mi355x-progress-aug-oct.png)

*Figure 3: Throughput per GPU vs. P90 interactivity for Kimi-K3 FP4 AgentX on MI355X, four submissions for each of vLLM and ATOM. Point labels give the parallelism layout; dashed circles mark KV offload to the DRAM tier.*

![Figure 4](figures/submission-timeline.png)

*Figure 4: Peak throughput per GPU of every MI355X Kimi-K3 submission, Aug 12 to Oct 27.*

vLLM's biggest jump came in its first three weeks, when decode context parallelism and a DRAM KV tier made concurrency 44–52 usable and the peak went from 3.8k to 10.2k. After that, throughput at 40 tok/s/user rose by up to 19% per run. ATOM nearly doubled its peak in its second run, six days after the first, and gained another 19–21% in each of its last two runs.

## MI355X hardware

*Table 2: Per-GPU specifications from the InferenceX GPU specs page, and the Rent – 3-year commit price used for TCO.*

| | MI355X | B200 | B300 | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| HBM capacity | 288 GB HBM3E | 180 GB | 268 GB | 186 GB | 278 GB |
| HBM bandwidth | 8 TB/s | 8 TB/s | 8 TB/s | 8 TB/s | 8 TB/s |
| Dense FP4 | 10.1 PFLOPS | 9.0 PFLOPS | 13.5 PFLOPS | 10.0 PFLOPS | 15.0 PFLOPS |
| Rent, $/GPU-hour | $2.90 | $3.70 | $4.25 | $4.00 | $5.00 |

Memory capacity decides how this model can be deployed. Kimi-K3's MXFP4 weights are about 1.56 TB. With TP8 on MI355X that is about 195 GB per GPU, so the whole model fits on one 8-GPU node with room left for KV cache. Eight B200s have 1.44 TB and eight GB200s 1.49 TB, so the published B200 and GB200 configurations split the model across two nodes with pipeline parallelism.

On compute, MI355X has a little more dense FP4 than B200 (10.1 vs. 9.0 PFLOPS), yet at 40 tok/s/user it produces 1.68× B200's throughput per GPU. Much of the kernel work described below went into that gap.

## Key optimizations

All of the changes below are in public images (`vllm/vllm-openai-rocm` nightlies and `rocm/atom-dev`) and the [InferenceX recipes](https://github.com/SemiAnalysisAI/InferenceX). Table 3 lists the ones with the largest measured effect, roughly from biggest to smallest.

*Table 3: Main optimizations and their measured effect. Throughput is per GPU; "at 40 tok/s/user" is read off each run's frontier.*

| Optimization | Where | Measured effect | Landed |
|---|---|---|---|
| Decode context parallelism (DCP8), DRAM KV tier | vLLM, ATOM, AITER | vLLM: throughput at 40 tok/s/user 2.1k → 6.9k, peak 3.8k → 10.2k; ATOM: peak 4.6k → 9.0k | Aug 27–Sep 4 |
| ReplaySSM, full CUDA graphs, FP16 SSM state | ATOM | Throughput at 40 tok/s/user 5.6k → 8.3k (+48%) | Sep 10, Sep 21 |
| DSpark draft kept on at high concurrency, gated RMSNorm prefill | ATOM | Throughput at 40 tok/s/user +29%; peak 14.5k → 17.5k | Oct 15 |
| AITER MLA for the DSpark draft | vLLM, AITER | +28.9% P90 interactivity and +14.5% throughput at concurrency 1 | Sep 17 |
| FlyDSL FP8 prefill attention, prefill delayer | ATOM, FlyDSL | Throughput at 40 tok/s/user +24%; peak 12.2k → 14.5k | Sep 29 |
| AITER Gluon KDA kernels, low-latency MoE router GEMM | vLLM, AITER | Throughput at 40 tok/s/user +19%; peak 14.9k → 16.0k | Oct 19 |
| Persistent FlyDSL decode kernels at concurrency 1 | vLLM, FlyDSL | Single-user speed 178 → 211 tok/s/user (+19%) | Oct 26 |
| Tail-aware admission caps | vLLM | Throughput at 40 tok/s/user +15%, at 60 tok/s/user +17% | Oct 27 |

### Decode context parallelism and a DRAM KV tier

At 100K+ tokens of context, decode is limited by KV bandwidth. Decode context parallelism shards the KV read across all eight GPUs, using an all-to-all backend and an AITER ASM kernel for the multi-token verify step. Paged KV that no longer fits in HBM goes to a DRAM tier of about 225 GB per rank, through SimpleCPUOffloadConnector in vLLM and LMCache in ATOM, with each rank's pool pinned to its own socket. Together these made concurrency 44–72 usable on a single node. The vLLM run that moved DCP8 to nearly every point raised throughput at 20 tok/s/user by 26%.

### Kimi Delta Attention state

Each request's KDA state is about 55 MiB. ATOM's first recipes offloaded it to CPU next to the paged KV. ReplaySSM replaced that: it rebuilds the state in place from a checkpoint ring kept in GPU memory, so the whole CPU budget can go to paged KV. The same change captured full CUDA graphs over the DSpark verify range and pinned the ranks to their sockets. Storing the SSM state in FP16 instead of FP32 then returned up to 7.8% of the KV pool, and a batch-size-1 CUDA graph stopped single-sequence decode steps from replaying the batch-size-2 graph, which had been 92% of steps at concurrency 1. Across these two changes, throughput at 40 tok/s/user went from 5.6k to 8.3k.

### Speculative decoding

Both stacks use the DSpark draft head, with the draft length set per concurrency: 7 tokens at concurrency 1, then fewer as batches grow. In vLLM, moving the draft's attention from Triton MLA to AITER MLA ([vllm#55966](https://github.com/vllm-project/vllm/pull/55966)) raised P90 interactivity by 28.9% and throughput by 14.5% at concurrency 1. ATOM now keeps a 4-token draft on at concurrency 48–72, where it previously ran without one, and added a gated RMSNorm pipeline for prefill ([ROCm/ATOM#2458](https://github.com/ROCm/ATOM/pull/2458)). That change raised ATOM's throughput at 40 tok/s/user by 29% and its peak from 14.5k to 17.5k.

### AITER and FlyDSL kernels

The experts run on AITER's MXFP4 MoE kernels (SITUV2, A4W4 and A8W4) with FlyDSL stage-2 FP8. Around them, vLLM gained a low-latency MoE router GEMM for any shape ([vllm#61895](https://github.com/vllm-project/vllm/pull/61895)), preshuffled FP4 GEMM configs, and latent-MoE up-projections that keep only the local rank's shards ([vllm#59591](https://github.com/vllm-project/vllm/pull/59591)). KDA runs on AITER Gluon kernels for decode and speculative decode ([vllm#57392](https://github.com/vllm-project/vllm/pull/57392)) and for chunked prefill ([vllm#59552](https://github.com/vllm-project/vllm/pull/59552)). The vLLM image that brought the Gluon KDA decode kernels and the router GEMM raised throughput at 40 tok/s/user by 19%.

In ATOM, FlyDSL FP8 prefill attention, together with holding a ready prefill for four decode passes at concurrency 16 and up, raised throughput at 40 tok/s/user by 24%. For single-user decode, vLLM now runs each MoE decoder layer at concurrency 1 as one persistent FlyDSL kernel with the TP all-reduce inside it ([vllm#60683](https://github.com/vllm-project/vllm/pull/60683)), which took single-user speed from 178 to 211 tok/s/user.

### Admission control

In one vLLM run, the slowest 10% of decode steps all happened in the largest batches. Capping the number of running sequences at concurrency 4–32 below that point raised throughput at 40 tok/s/user by 15% and at 60 tok/s/user by 17%.

## Acknowledgements

Thanks to the vLLM and Moonshot AI communities, to Inferact for the DSpark draft model, to SemiAnalysis for AgentX and the InferenceX CI infrastructure, and to the AMD AITER, FlyDSL, ATOM, vLLM and ROCm teams.

## References

1. SemiAnalysis InferenceX, Kimi-K3 dashboard. https://inferencex.semianalysis.com/inference/kimi-k3
2. SemiAnalysis AI Cloud TCO Model (source of the $/GPU/hr rates). https://semianalysis.com/ai-cloud-tco-model/
3. AMD, *MoRI UMBP Empowers AMD Instinct™ MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard* (more on the AgentX benchmark). https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html
4. InferenceX recipe history for MI355X Kimi-K3: vLLM PRs [#2508](https://github.com/SemiAnalysisAI/InferenceX/pull/2508), [#2787](https://github.com/SemiAnalysisAI/InferenceX/pull/2787), [#2810](https://github.com/SemiAnalysisAI/InferenceX/pull/2810), [#3124](https://github.com/SemiAnalysisAI/InferenceX/pull/3124), [#3833](https://github.com/SemiAnalysisAI/InferenceX/pull/3833), [#3951](https://github.com/SemiAnalysisAI/InferenceX/pull/3951), [#4215](https://github.com/SemiAnalysisAI/InferenceX/pull/4215), [#4393](https://github.com/SemiAnalysisAI/InferenceX/pull/4393), [#4473](https://github.com/SemiAnalysisAI/InferenceX/pull/4473); ATOM PRs [#2670](https://github.com/SemiAnalysisAI/InferenceX/pull/2670), [#2716](https://github.com/SemiAnalysisAI/InferenceX/pull/2716), [#2852](https://github.com/SemiAnalysisAI/InferenceX/pull/2852), [#3207](https://github.com/SemiAnalysisAI/InferenceX/pull/3207), [#3407](https://github.com/SemiAnalysisAI/InferenceX/pull/3407), [#4085](https://github.com/SemiAnalysisAI/InferenceX/pull/4085).
