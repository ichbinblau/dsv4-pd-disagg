# Serving Kimi-K3 2.8T on One AMD Instinct™ MI355X Node: AgentX Results

*October 2026*

The first question with Kimi-K3 is where to put it. The MXFP4 weights are about 1.56 TB. MI355X has 288 GB of HBM3E per GPU, so with TP8 each GPU holds about 195 GB and the whole model fits on a single 8-GPU node. An 8× B200 node does not have enough memory, and the published B200 and GB200 configurations run the model across two nodes with pipeline parallelism.

On SemiAnalysis's public AgentX benchmark, that single MI355X node reaches a peak of 21.7M total tokens per $1 TCO on the Rent (3-year commit) tier, retrieved 2026-10-30. That is the highest peak on the Kimi-K3 dashboard. The best published peaks for NVIDIA systems are 10.4M for B200, 10.6M for B300, 9.9M for GB200 NVL72 and 19.1M for GB300 NVL72. Compared at the same interactivity, MI355X gets 2.0–2.6× the tokens per dollar of B200 between 25 and 100 tok/s/user.

Our first MI355X Kimi-K3 submission was on Aug 12, two weeks after the first NVIDIA result, and it peaked at 3.8k tokens/s per GPU. The current peak is 17.5k.

![Figure 1](figures/tco-frontier-vs-nvidia.png)

*Figure 1: Kimi-K3 FP4 AgentX, total tokens per $1 TCO vs. P90 interactivity, Rent – 3-year commit tier. Each curve is one system's Pareto frontier, using the best point across all published frameworks. Red is one 8× MI355X node running vLLM or ATOM. Data from the [InferenceX dashboard](https://inferencex.semianalysis.com/inference/kimi-k3), retrieved 2026-10-30.*

## The model and the benchmark

Kimi-K3 is a 2.8T-parameter MoE with 896 experts and 1M-token native context. Its attention stack mixes two layer types: Kimi Delta Attention (KDA) layers, which keep a recurrent state per request, and MLA layers, which keep a paged KV cache. It comes with a DSpark draft head (Inferact/Kimi-K3-DSpark) for speculative decoding.

The mixed attention is what makes agentic serving harder here than for a pure-MLA model. When a session returns for its next turn, the server has to restore the paged KV and the KDA state, or recompute the prefix. In AgentX that is nearly every request. The benchmark replays real agentic coding sessions of about 43 turns; the median turn has 142K input tokens and 444 output tokens, and over 96% of prompt tokens repeat a prefix the server has seen before.

P90 interactivity is 1 divided by the P90 inter-token latency over the full response, in tok/s/user. Tokens per $1 TCO is total throughput per GPU, including cached tokens, divided by the hourly GPU cost from the SemiAnalysis AI Cloud TCO Model: on the Rent tier $2.90 for MI355X, $3.70 for B200, $4.25 for B300, $4.00 for GB200 and $5.00 for GB300; on the Owning tier $1.50, $1.73, $2.26, $1.86 and $2.31.

## Cost and throughput

*Table 1: Total tokens per $1 TCO in millions (Rent tier) at the same P90 interactivity, interpolated with the dashboard's own frontier spline and not extrapolated. "—" means the system has no published point there. The number in parentheses is MI355X divided by that system.*

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

*Figure 2: MI355X tokens per dollar divided by each NVIDIA system's tokens per dollar at the same P90 interactivity (Rent tier). Bars above 1.0 mean MI355X is cheaper per token.*

Against B200, MI355X gets 2.0–2.6× the tokens per dollar across the whole range in the table. Most of that is throughput rather than price: MI355X produces 1.6–2.0× the tokens per second per GPU of B200 over the same range, for example 13.3k vs. 7.9k at 40 tok/s/user. We think the main reason is the memory layout. B200 has to split the model across two nodes with pipeline parallelism, while MI355X keeps all of it inside one node.

Against GB200 NVL72, MI355X gets 1.3–2.1× the tokens per dollar, and 1.4–1.5× the throughput per GPU at 25–40 tok/s/user. On the Owning tier the ratios are 1.8–2.3× for B200 and 1.2–1.9× for GB200.

GB300 NVL72 is ahead from 40 tok/s/user upward, where MI355X is at 0.64–0.87× of its tokens per dollar. Those GB300 points come from wide-EP disaggregated deployments of 24 to 40 GPUs inside the NVLink rack. MI355X does have the higher peak (21.7M vs. 19.1M), reached at 20.7 tok/s/user, which is below the lowest interactivity GB300 has published.

Running one replica per 8-GPU box also has practical benefits that do not show up in these numbers: simpler scheduling, no cross-node pipeline bubbles, and a smaller failure domain.

## Eleven weeks of software

AMD supports two serving stacks for Kimi-K3 on MI355X, vLLM and ATOM. Figure 3 shows both over time on the same 8 GPUs.

![Figure 3](figures/mi355x-progress-aug-oct.png)

*Figure 3: Throughput per GPU vs. P90 interactivity for Kimi-K3 FP4 AgentX on MI355X by submission date. Left: vLLM. Right: ATOM.*

With vLLM, peak throughput per GPU went from 3.8k to 15.9k tok/s. At 40 tok/s/user it went from 2.1k to 13.1k, and at 60 tok/s/user from 1.1k to 8.8k. Single-user speed at concurrency 1 went from 61 to 212 tok/s/user. With ATOM, peak throughput went from 4.6k to 17.5k tok/s, and at 60 tok/s/user from 2.8k to 8.5k.

## What changed

All of the following is in public images (`vllm/vllm-openai-rocm` nightlies and `rocm/atom-dev`) and in the [InferenceX recipes](https://github.com/SemiAnalysisAI/InferenceX).

Kernels. The experts use AITER's MXFP4 MoE kernels (SITUV2, A4W4 and A8W4) with FlyDSL stage-2 FP8. Around them, a low-latency MoE router GEMM that handles any shape ([vllm#61895](https://github.com/vllm-project/vllm/pull/61895)), preshuffled FP4 GEMM configs, and latent-MoE up-projections that keep only the local rank's shards ([vllm#59591](https://github.com/vllm-project/vllm/pull/59591)) reduce the time and memory spent outside the expert GEMMs. KDA runs on AITER Gluon kernels for decode and speculative decode ([vllm#57392](https://github.com/vllm-project/vllm/pull/57392)) and for chunked prefill ([vllm#59552](https://github.com/vllm-project/vllm/pull/59552)), alongside a fused gfx950 chunk kernel. Moving the DSpark draft's attention from Triton MLA to AITER MLA ([vllm#55966](https://github.com/vllm-project/vllm/pull/55966)) raised P90 interactivity by 28.9% and throughput per GPU by 14.5% at concurrency 1.

Single-user decode is mostly small kernels and collectives, so launch and sync overhead dominates. At concurrency 1, vLLM now runs each MoE decoder layer as one persistent FlyDSL kernel with the TP all-reduce done inside it ([vllm#60683](https://github.com/vllm-project/vllm/pull/60683)). Several FP8 fusions also landed: AttnRes output with per-token FP8 input quantization ([vllm#59069](https://github.com/vllm-project/vllm/pull/59069)), the GLA gate GEMM with MLA FP8 prefill q/k/v prep ([vllm#59634](https://github.com/vllm-project/vllm/pull/59634)), and FlyDSL FP8 prefill attention in ATOM. TP all-reduce uses AITER's INT4 QuickReduce for messages of 16 MB and up.

Long-context decode. At 100K+ tokens of context, decode is limited by KV bandwidth. Decode context parallelism shards the KV read across all 8 GPUs, using an all-to-all backend and an AITER ASM kernel for the multi-token verify step. The vLLM recipe now uses DCP8 at every concurrency, and ATOM uses it from concurrency 14 up. Both target and draft use an FP8 KV cache.

KDA state and offload. Each request's KDA state is about 55 MiB. ATOM first offloaded it to CPU, then replaced that with ReplaySSM, which rebuilds the state in place from a checkpoint ring kept in GPU memory, so the whole CPU budget can go to paged KV. Storing the SSM state in FP16 instead of FP32 gave back up to 7.8% of the KV pool. For paged KV, both stacks keep a DRAM tier of about 225 GB per rank, through SimpleCPUOffloadConnector in vLLM and LMCache in ATOM, with each rank's pool pinned to its own socket.

Speculative decoding and scheduling. DSpark uses 7 draft tokens at concurrency 1 and fewer as batches grow (5, 4, then 3), and ATOM now keeps the draft model on at concurrency 48–72. In one run we saw that the slowest 10% of decodes happened in the largest batches, so we lowered the running-sequence caps at concurrency 4–32, which raised P90 interactivity. ATOM added a batch-size-1 CUDA graph, since 92% of decode steps at concurrency 1 had been replaying the batch-size-2 graph, and both stacks size their graph capture to the DSpark verify batch.

## Summary

One 8× MI355X node serving Kimi-K3 2.8T peaks at 21.7M tokens per $1 TCO, the highest on the Kimi-K3 AgentX dashboard. At the same interactivity it gets 2.0–2.6× the tokens per dollar of B200 and 1.3–2.1× of GB200 NVL72. GB300 NVL72 is ahead from 40 tok/s/user upward. The 288 GB per GPU is what lets the whole model fit on one node, which B200 and GB200 cannot do. Since the first submission, peak throughput per GPU has gone from 3.8k to 17.5k tok/s, all from AITER, FlyDSL, vLLM and ATOM work that is upstream or in public images.

Next, we want to bring MoRI UMBP, the shared and deduplicated DRAM KV pool we use for DeepSeek-V4-Pro, to Kimi-K3's mix of paged KV and KDA state, and to use MoRI-based PD disaggregation to close the gap with GB300 NVL72 at higher interactivity.

## Acknowledgements

Thanks to the vLLM and Moonshot AI communities, to Inferact for the DSpark draft model, to SemiAnalysis for AgentX and the InferenceX CI infrastructure, and to the AMD AITER, FlyDSL, ATOM, vLLM and ROCm teams.

## References

1. SemiAnalysis InferenceX, Kimi-K3 dashboard. https://inferencex.semianalysis.com/inference/kimi-k3
2. SemiAnalysis AI Cloud TCO Model (source of the $/GPU/hr rates). https://semianalysis.com/ai-cloud-tco-model/
3. InferenceX recipe history for MI355X Kimi-K3: vLLM PRs [#2508](https://github.com/SemiAnalysisAI/InferenceX/pull/2508), [#2787](https://github.com/SemiAnalysisAI/InferenceX/pull/2787), [#3124](https://github.com/SemiAnalysisAI/InferenceX/pull/3124), [#3833](https://github.com/SemiAnalysisAI/InferenceX/pull/3833), [#3951](https://github.com/SemiAnalysisAI/InferenceX/pull/3951), [#4215](https://github.com/SemiAnalysisAI/InferenceX/pull/4215), [#4393](https://github.com/SemiAnalysisAI/InferenceX/pull/4393), [#4473](https://github.com/SemiAnalysisAI/InferenceX/pull/4473); ATOM PRs [#2670](https://github.com/SemiAnalysisAI/InferenceX/pull/2670), [#2716](https://github.com/SemiAnalysisAI/InferenceX/pull/2716), [#2852](https://github.com/SemiAnalysisAI/InferenceX/pull/2852), [#3207](https://github.com/SemiAnalysisAI/InferenceX/pull/3207), [#3407](https://github.com/SemiAnalysisAI/InferenceX/pull/3407), [#4085](https://github.com/SemiAnalysisAI/InferenceX/pull/4085).
4. MoRI UMBP on DeepSeek-V4 PD disaggregation. https://github.com/ichbinblau/dsv4-pd-disagg/tree/main/dsv4-umbp

*Methodology: numbers come from the InferenceX public API (`/api/v1/benchmarks?model=Kimi-K3`), retrieved 2026-10-30. The snapshots and the script that generates every figure and table are in `data/` and `make_figures.py`. Same-interactivity values use InferenceX's `iso_interactivity.py`, which ports the dashboard's interpolation.*
