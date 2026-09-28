# MoRI UMBP Empowers AMD Instinct™ MI355X to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard

*September 2026*

Agentic applications run long multi-turn sessions, which makes the cost of serving them depend overwhelmingly on how much KV cache the server can reuse instead of recomputing.

In July, together with Moonshot AI, we introduced **UMBP** (Unified Memory & Bandwidth Pool) in [*Rebuilding Agentic AI from First Principles for AMD GPU*](https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html). UMBP is a KV cache infrastructure built by the AMD MoRI team from first principles, starting from the agentic workload itself and purpose-built for the AMD platform. Over the past month we have contributed that work to the SGLang community, where it lands in the open-source ecosystem as the backend behind the new KVCache Store Linker, so the whole community benefits. On the public SemiAnalysis AgentX benchmark with DeepSeek-V4-Pro-0813 1.6T, AMD MI355X — running MoRI disaggregation with UMBP as the unified KV cache pool, on top of AMD's ongoing optimization work in the SGLang community — now **surpasses NVIDIA GB200 NVL72, B300 and GB300 NVL72 at selected operating points**, and peaks at **69M total tokens per $1 TCO** (SemiAnalysis Rent / 3-Year-Commit cost tier, MI355X at $2.9/chip/hr, as of 2026-09-25).

![Figure 1](figures/fig1_tco_vs_b200.png)

*Figure 1: DeepSeek-V4-Pro-0813 1.6T agentic total tokens per $1 TCO vs. P90 interactivity, from the public SemiAnalysis InferenceX dashboard (Rent / 3-Year-Commit tier, updated 2026-09-25). Red is MI355X FP4 with UMBP + MoRI + SGLang; the green curves are the NVIDIA field — B200, B300, H200, GB200 and GB300 NVL72, plus a Vera Rubin NVL72 preview. Up and to the right is better; labels show the parallelism layout of each point.*

At the operating points below, MI355X leads on tokens per dollar, and against B300 and GB200 NVL72 also on throughput per chip. Each MI355X point is paired with a measured NVIDIA point at the same or lower P90 interactivity, so in every row MI355X is at least as fast per user.

| NVIDIA system (dashboard series, $/chip/hr) | P90 interactivity, tok/s/user (MI355X vs. NVIDIA) | Tokens per $1 TCO (MI355X vs. NVIDIA) | Throughput per chip, tok/s (MI355X vs. NVIDIA) |
|---|---|---|---|
| B300 (SGLang), $4.25 | 54.1 vs. 53.9 | 69.3M vs. 41.9M (**1.65×**) | 55.8k vs. 49.5k (**1.13×**) |
| B300 (SGLang), $4.25 | 115.9 vs. 104.8 | 19.8M vs. 10.3M (**1.93×**) | 15.9k vs. 12.1k (**1.32×**) |
| GB200 NVL72 (Dynamo vLLM), $4 | 102.5 vs. 84.7 | 26.6M vs. 4.7M (**5.6×**) | 21.4k vs. 5.3k (**4.1×**) |
| GB200 NVL72 (Dynamo vLLM), $4 | 152.7 vs. 135.0 | 10.2M vs. 2.7M (**3.8×**) | 8.2k vs. 3.0k (**2.7×**) |
| GB300 NVL72 (Dynamo SGLang), $5 | 152.7 vs. 146.0 | 10.2M vs. 8.9M (**1.15×**) | — |

Rows are paired by P90 interactivity, not by concurrency; the two sides may run on different numbers of chips. By row, the concurrency is c256 vs. c128, c32 vs. c16, c48 vs. c8, c16 vs. c4 and c16 vs. c8 (MI355X vs. NVIDIA). The MI355X points come from the [Sep 25 run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35879254139/attempts/1); the NVIDIA points are the latest dashboard runs as of 2026-09-25: B300 from 2026-09-15, GB200 NVL72 from 2026-08-18 and GB300 NVL72 from 2026-09-10. Against GB300 NVL72 the lead is in tokens per dollar only.

The rest of this post covers in detail how UMBP produces that result.

## A Quick Recap of the Public AgentX Benchmark

AgentX is a public benchmark built by SemiAnalysis from real-world agentic coding traces, served through the public InferenceX harness and dashboard. It is fast becoming an industry standard for agentic inference evaluation, adopted worldwide by leading teams including OpenAI, Meta, Inferact, RadixArk, MiniMax, Alibaba Qwen, Moonshot AI, Zhipu GLM and Oracle. Its methodology, results and CI are all public, so **every number in this post can be checked independently, including by NVIDIA.**

AgentX replays whole agent sessions: each turn appends tool results to the accumulated context and re-queries the model. That traffic has four properties chat benchmarks do not produce.

- **Long-running, multi-turn sessions** — roughly 43 turns per session, around 1M tokens of context traffic over a session's life.
- **Long contexts, short outputs** — median 142K input tokens against 444 output tokens per turn.
- **Extensive prefix reuse** — above 96% of prompt tokens are repeats of a prefix the server has already seen.
- **Subagents and tool calls**, which fan a single user task out into several concurrent sessions sharing a prefix.

It reports TTFT, P90 interactivity (per-user tok/s), TPGS (total tokens per GPU-second, counting cached tokens), and TCO (infrastructure cost per token) — the last of which is what Figure 1 plots.

At 96% prefix reuse, *cache management determines prefill cost*. And because the agent blocks on the full response, the latency that matters is end-to-end, dominated by decode. AgentX therefore presents three challenges we had to solve:

- **The KV cache does not fit in HBM.** With many long sessions live at once and subagents fanning out, most reusable KV has already been evicted from GPU memory at moderate concurrency.
- **The KV cache is not sharded across TP ranks.** Under DeepSeek-V4 sparse attention, the MLA latent and DSA indexer KV are replicated in full on every rank.
- **Cache warmth is not free.** A host cache that lives inside the engine process is lost on every restart, and on every rolling upgrade.

## MoRI UMBP + the SGLang KVCache Store Linker

To address those challenges we went back over the problems UMBP had been hitting. UMBP was integrated into SGLang as a **HiCache L3 storage backend**, and on AgentX we made the following observations:

- **Wasted shareable DRAM.** Unsharable HiCache occupies DRAM that could otherwise serve the shareable L3 backend, shrinking effective shareable capacity.
- **Indirect data path.** Sitting between L1 HBM and the L3 backend, HiCache adds load/offload overhead and blocks a direct L1 ⇔ L3 data path.
- **Per-rank local management.** HiCache is embedded per rank and makes all KV decisions on local information only, which is globally suboptimal.
- **Redundant replication.** Under MLA + TP, each TP rank replicates the KV cache and loads/offloads independently, wasting memory and PCIe bandwidth.
- **No layer-wise pipelining.** HiCache overlaps its L2→L1 load layer by layer, but the fetch from the external L3 backend into L2 is not pipelined — it must complete before compute starts.
- **Cache dies with the engine.** HiCache lives in the engine process, so any restart discards the host-side cache.

### The KVCache Store Linker

We proposed to the SGLang maintainers an option to bypass HiCache entirely with a direct data path between L1 HBM and external KV cache stores — a direction that turned out to align closely with the community's own plans. The MoRI team then co-designed the **KVCache Store Linker** with the SGLang community, integrating UMBP as a first-class backend. The linker connects SGLang's unified radix tree straight to the distributed DRAM pool. On a prefix match, prefill pulls KV pages from DRAM instead of recomputing them.

Linker + UMBP resolves all six issues above:

- **Fully shareable DRAM** across DP ranks and model instances — enabling DP + round-robin deployments via cross-DP-rank KV cache sharing.
- **Direct L1 ⇔ L3 path** with no intermediate overhead, improving TTFT by up to 13% over the HiCache path on its own.
- **Global KV management** — UMBP places and evicts KV based on global information, more effective than per-rank local policies.
- **Deduplication + split load/offload by rank**, alleviating memory and PCIe bandwidth pressure. A TP-N prefill now stores and fetches one copy of the replicated MLA/DSA KV instead of N; at TP8, eight keys become one. That multiplies effective DRAM capacity and cuts host traffic by the same factor.
- **Layer-wise pipelined loading**, with UMBP hiding the added per-layer request overhead via batching, layer grouping, a ranged API, and an optimized GPU gather kernel for host-to-device KV loading.
- **Cache survives engine restarts** — in UMBP standalone mode the KV pool lives in a separate per-node process, so restarts and upgrades reuse it with no warm-up.

### Results

Replacing HiCache with the UMBP linker, measured end-to-end on the AgentX agentic-coding scenario:

| Concurrency | Throughput/GPU | P90 TTFT |
|---|---|---|
| 192 | **+14%** | **–66%** (35.3 s → 11.9 s) |
| 256 | +9.7% | –51% |

The TTFT reduction comes from eliminating prefix recomputation on a workload with 96% prefix reuse. No kernel changed.

**A large enough cache changes the topology.** Once the deduplicated DRAM tier is in place, prefill no longer needs TP8 simply to hold KV. At concurrency 16–48 we run a **TP4 prefill with a TP8 decode (12 GPUs)** instead of TP8 + TP8 (16 GPUs). UMBP serves 30%, 52% and 75% of prompt tokens at concurrency 16, 32 and 48, while less than 3% are recomputed. Throughput per GPU rises 24–34% on 25% fewer GPUs — and four of the five leads over NVIDIA in the table above come from these TP4-prefill points.

![Figure 2](figures/fig3_umbp_tp4_prefill.png)

*Figure 2: Left: where the TP4 prefill finds the KV for each prompt token under the UMBP linker — GPU HBM prefix cache, UMBP DRAM tier, or recomputation. As concurrency grows and HBM evicts more, the DRAM tier absorbs the difference. Right: throughput per GPU of the [Sep 15 recipe](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34926284365) (TP8 prefill + TP8 decode, 16 GPUs) vs. the Sep 25 recipe (TP4 prefill + UMBP + TP8 decode, 12 GPUs). The Sep 25 arms also include optimistic prefill and the SGLang v0.5.20 image; concurrency 16 additionally uses DSpark γ=6.*

The trade-off is prefill time. With half the prefill GPUs, P90 TTFT at concurrency 16–48 rises by 26–53% (2.2–3.6 s → 2.7–5.6 s). For agentic workloads, where the agent waits for the full response and TTFT is a small fraction of it, we take that trade for more throughput per GPU at a fixed per-user decode speed.

## Further Optimizations

Alongside this, the AMD SGLang team continues to deliver optimizations for DeepSeek-V4-Pro, covering the following areas.

- **FP4 sparse-attention indexer** ([sglang#37353](https://github.com/sgl-project/sglang/pull/37353)). DeepSeek-V4's DSA indexer scores the whole context for every query token at every layer, and carries its own per-token KV. Running it on AITER FP4 kernels on gfx950 cuts that KV from **132 B to 68 B per token** — more concurrent sequences in HBM, less indexer bandwidth per decode step, and fewer bytes over the MoRI link.
- **Optimistic prefill with request-owned speculative KV** ([sglang#38978](https://github.com/sgl-project/sglang/pull/38978), [sglang#40111](https://github.com/sgl-project/sglang/pull/40111)). In PD disaggregation a request normally waits for decode to bootstrap it before prefill can start; at high concurrency that handshake is pure queueing time. Letting prefill start optimistically, with the speculative KV owned by the request rather than a pre-reserved decode slot, cut **P90 TTFT by 27.7% at concurrency 256**. Removing a host sync from DSpark prefill slot expansion cut P90 TTFT a further **13–16%** at concurrency 128–256.
- **Per-stream split-K for MLA decode** ([sglang#39968](https://github.com/sgl-project/sglang/pull/39968)) picks the split-K factor per index stream instead of applying one setting to layers whose KV lengths differ by orders of magnitude.

![Figure 3](figures/fig1_pareto_0821_vs_0925.png)

*Figure 3: Throughput per GPU vs. P90 interactivity across the optimization campaign. The Aug 21 baseline uses 16 GPUs (1P1D, TP8 + TP8) at every point; the optimized recipes pick 8, 12 or 16 GPUs per concurrency and report throughput normalized per GPU.*

## Summary

This work produced two results.

**First, against NVIDIA Blackwell systems.** On the public AgentX leaderboard, MI355X surpasses GB200 NVL72, B300 and GB300 NVL72 at selected operating points, by up to **5.6×** in tokens per dollar (see the table after Figure 1).

**Second, against ourselves a month earlier.** Same benchmark, same concurrency of 192:

| Metric | Aug 21 | Sep 25 | Change |
|---|---|---|---|
| Throughput/GPU | 22.9k tok/s | 46.6k tok/s | **2.03×** |
| P90 TTFT | 33.6 s | 11.8 s | **–65%** |
| P90 interactivity | 23.5 tok/s/user | 59.3 tok/s/user | **2.5×** |

Peak throughput per GPU also rose from 22.9k to 55.8k (**2.4×**), at concurrency 256.

In an agentic workload, 96% of prompt tokens have already been computed, so the cost per token is set by the system that manages where those tokens live. UMBP is that system: it turns DRAM into a deduplicated KV pool that is shareable across instances and survives engine restarts, wired straight into SGLang's radix tree through the KVCache Store Linker.

The first item on the July roadmap was completing the UMBP integration; that is now delivered. The next is bringing UMBP to the vLLM ecosystem.

## Acknowledgements

We thank the SGLang community for design reviews and fast upstreaming, SemiAnalysis for the AgentX benchmark and InferenceX CI infrastructure, and the AMD AITER, MoRI and ROCm teams.

## References

1. SemiAnalysis — AgentX benchmark and InferenceX dashboard. https://inferencex.semianalysis.com
2. vLLM — AgentX. https://vllm.ai/blog/2026-09-08-vllm-agentx
3. AMD — Rebuilding Agentic AI from First Principles for AMD GPU, together with Moonshot AI ("What is UMBP"). https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
