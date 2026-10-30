# DeepSeek-V4-Pro on AMD Instinct™ MI355X: AgentX Results and the Work Behind Them

*October 2026*

We have been running DeepSeek-V4-Pro (1.6T, FP4) on AMD Instinct MI355X in SemiAnalysis's public AgentX benchmark since August. The serving stack is SGLang with MoRI for prefill/decode disaggregation and MoRI UMBP as the KV cache pool. As of the latest run on Oct 28, the InferenceX dashboard shows MI355X at a peak of 99.1M total tokens per $1 TCO. The best published B200 result is 66.1M and the best B300 result is 82.7M, so MI355X produces 1.50× and 1.20× as many tokens per dollar. GB200 NVL72 peaks at 95.0M. All of these use the Rent (3-year commit) cost tier, with data retrieved on 2026-10-30.

Peak numbers hide where on the latency curve each system gets them, so most of this post compares systems at the same interactivity. At 70 tok/s/user, MI355X gets 1.85× the tokens per dollar of B200 and 1.53× of B300. GB300 NVL72 is still ahead of us on this benchmark, and we show by how much below.

The other thing we want to show is how much of this came from software in a short time. On Aug 21 this recipe peaked at 22.9k tokens/s per GPU. On Oct 28 it peaked at 79.8k, on the same GPUs.

![Figure 1](figures/tco-frontier-vs-nvidia.png)

*Figure 1: DeepSeek-V4-Pro FP4 AgentX, total tokens per $1 TCO vs. P90 interactivity, Rent – 3-year commit tier. Each curve is one system's Pareto frontier. For NVIDIA systems we took the best point across every published framework (SGLang, vLLM, Dynamo-SGLang, Dynamo-vLLM, Dynamo-TRT). Red is MI355X with MoRI-SGLang and MoRI UMBP. Data from the [InferenceX dashboard](https://inferencex.semianalysis.com/inference/deepseek-v4), retrieved 2026-10-30.*

## About the benchmark

AgentX replays real agentic coding sessions. Each turn appends tool output to the context and sends the whole thing back to the model. A session runs about 43 turns; the median turn has 142K input tokens and 444 output tokens, and over 96% of prompt tokens are a prefix the server has already processed. Recipes, logs and results are all public, so anyone can check the numbers here.

Two definitions from the dashboard are used throughout. P90 interactivity is 1 divided by the P90 inter-token latency over the full response, in tok/s/user; since an agent waits for the full response before acting, this is the latency that matters. Tokens per $1 TCO is total throughput per GPU, cached tokens included, divided by the hourly GPU cost from the SemiAnalysis AI Cloud TCO Model. On the Rent tier those costs are $2.90 for MI355X, $3.70 for B200, $4.25 for B300, $4.00 for GB200 and $5.00 for GB300 per GPU-hour. On the Owning tier they are $1.50, $1.73, $2.26, $1.86 and $2.31.

## Comparison at the same interactivity

Table 1 reads each system's frontier at fixed interactivity levels. We used the same spline interpolation the dashboard uses for its charts and did not extrapolate past any system's measured range.

*Table 1: Total tokens per $1 TCO in millions (Rent tier). "—" means the system has no published point at that interactivity. The number in parentheses is MI355X divided by that system.*

| P90 interactivity | MI355X | B200 | B300 | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|---|
| 70 tok/s/user | 89.1 | 48.1 (1.85×) | 58.3 (1.53×) | — | 98.1 (0.91×) |
| 80 tok/s/user | 74.3 | 45.8 (1.62×) | 53.6 (1.39×) | — | 92.5 (0.80×) |
| 100 tok/s/user | 44.8 | 39.1 (1.15×) | 34.8 (1.29×) | 58.2 (0.77×) | 74.1 (0.60×) |
| 120 tok/s/user | 30.2 | 26.9 (1.12×) | 20.2 (1.49×) | 25.3 (1.19×) | 53.0 (0.57×) |
| 150 tok/s/user | 23.2 | 15.9 (1.46×) | 10.3 (2.26×) | 16.4 (1.42×) | 30.9 (0.75×) |
| Peak | 99.1 | 66.1 (1.50×) | 82.7 (1.20×) | 95.0 (1.04×) | 112.8 (0.88×) |

![Figure 2](figures/iso-interactivity-ratio.png)

*Figure 2: MI355X tokens per dollar divided by each NVIDIA system's tokens per dollar at the same P90 interactivity (Rent tier). Bars above 1.0 mean MI355X is cheaper per token.*

B200 and B300 are the closest comparison, since they are also 8-GPU servers. MI355X is ahead of both at every interactivity level in the table, by 1.12–1.85× over B200 and 1.29–2.26× over B300. The lower GPU price accounts for part of that, but not all of it. At 70 tok/s/user MI355X also has 1.45× the throughput per GPU of B200 (71.8k vs. 49.5k tok/s) and slightly more than B300 (68.8k). Switching to the Owning tier, MI355X still gets 1.67× the tokens per dollar of B200 and 1.57× of B300 at 70 tok/s/user.

GB200 NVL72 is a mixed picture. MI355X has a slightly higher peak (99.1M vs. 95.0M) and is 1.19–1.42× better at 120–150 tok/s/user, but GB200 is ahead around 100 tok/s/user. Keep in mind that GB200 runs on a 72-GPU NVLink rack, while our runs use 8-GPU nodes connected over RoCE.

GB300 NVL72 is ahead at every interactivity level, with MI355X at 0.57–0.95× of its tokens per dollar. We are closest at the high-throughput end: 0.95× at 65 tok/s/user and 0.91× at 70. The gap has been shrinking with each run.

## Ten weeks on the same hardware

Figure 3 shows each major submission of the MoRI-SGLang recipe on MI355X between August and October.

![Figure 3](figures/mi355x-progress-aug-oct.png)

*Figure 3: Throughput per GPU vs. P90 interactivity for the MI355X MoRI-SGLang AgentX recipe on [Aug 21](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32269076444/attempts/4), [Sep 25](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35879254139/attempts/1), [Oct 4](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37181045340/attempts/1) and [Oct 28](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40473082761/attempts/4).*

Peak throughput per GPU went from 22.9k to 79.8k tok/s (3.5×), and peak tokens per dollar from 28.5M to 99.1M. The change is larger at higher interactivity. In August the recipe only reached 70 tok/s/user at very low concurrency, where it produced 4.4k tok/s per GPU. It now produces 71.8k at that interactivity.

Figure 4 looks at a single point, concurrency 192 on 16 GPUs. Throughput per GPU went up 2.3×, P90 TTFT went from 33.6 s to 12.3 s, and P90 interactivity went from 23.5 to 85.1 tok/s/user.

![Figure 4](figures/mi355x-c192-aug-vs-oct.png)

*Figure 4: 16 MI355X GPUs at concurrency 192, Aug 21 vs. Oct 28.*

## MoRI UMBP

With over 96% of prompt tokens already computed, the cost of serving AgentX depends mostly on where the cached KV lives and how fast it can be brought back. We ran into three problems. First, the KV does not fit in HBM: with many long sessions open and subagents running in parallel, most reusable KV is evicted from GPU memory even at moderate concurrency. Second, with DeepSeek-V4's sparse attention, the MLA latent and the DSA indexer KV are replicated in full on every TP rank. Third, a host cache inside the engine process is lost every time the engine restarts or is upgraded.

MoRI UMBP (Unified Memory & Bandwidth Pool) is the KV cache layer the AMD MoRI team built for this kind of workload. We first described it with Moonshot AI in [*Rebuilding Agentic AI from First Principles for AMD GPU*](https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html). Together with the SGLang maintainers we then designed the KVCache Store Linker, which connects SGLang's radix tree directly to a distributed DRAM pool instead of going through the per-rank L2 host cache. On a prefix hit, prefill loads the KV pages from DRAM rather than recomputing them.

A few properties of this design matter for AgentX. The DRAM pool is shared across DP ranks and across model instances, so round-robin DP routing still gets cache hits. KV moves directly between HBM and DRAM and is loaded layer by layer, with batching, layer grouping, a ranged API and a GPU gather kernel to keep per-layer request overhead low. Replicated MLA/DSA KV is stored once instead of once per TP rank, so at TP8 the effective DRAM capacity is 8× larger and PCIe traffic is 8× smaller. Placement and eviction use global information instead of per-rank decisions. In standalone mode the pool runs in its own per-node process, so it survives engine restarts.

The linker helped even in a run where DRAM served almost nothing. In an on/off comparison with the same recipe (1P1D, TP8 + TP8, 16 GPUs), HBM served about 95.6% of prompt tokens and DRAM under 1%. Turning MoRI UMBP on still gave 8.3% more throughput per GPU and 35% lower P90 TTFT at concurrency 256 (Figure 5). The gain comes from removing the intermediate host tier and its load/offload overhead.

![Figure 5](figures/umbp-on-off-pareto.png)

*Figure 5: Throughput per GPU vs. P90 interactivity with MoRI UMBP off and on, same recipe, concurrency 128 and 256.*

The larger effective cache also let us change the deployment. Prefill no longer needs TP8 just to hold KV, so at concurrency 16–48 we run TP4 prefill with TP8 decode on 12 GPUs instead of TP8 + TP8 on 16. Throughput per GPU went up 26–34% with 25% fewer GPUs (Figure 6). In the Oct 28 run, FP8 KV at low concurrency keeps more KV in HBM, and MoRI UMBP still serves 38% and 57% of prompt tokens at concurrency 32 and 48. About 7% are recomputed.

![Figure 6](figures/umbp-tp4-prefill.png)

*Figure 6: Left: where TP4 prefill finds the KV for each prompt token (GPU HBM, MoRI UMBP DRAM, or recomputed). Right: throughput per GPU for TP8 + TP8 on 16 GPUs vs. TP4 prefill + MoRI UMBP + TP8 decode on 12 GPUs.*

## AITER and SGLang changes

Everything here is in the public SGLang ROCm images (`lmsysorg/sglang-rocm`) and the [InferenceX recipes](https://github.com/SemiAnalysisAI/InferenceX), roughly in the order it landed.

On the attention side, the largest change was running DeepSeek-V4's DSA indexer on AITER FP4 kernels for gfx950. The indexer scores the full context for every query token at every layer and keeps its own per-token KV, which this change shrinks from 132 B to 68 B per token. That leaves room for more sequences in HBM, cuts indexer bandwidth per decode step, and reduces what MoRI has to transfer between prefill and decode. We also split the KV pool into an FP8 NoPE part and a BF16 RoPE part on the unified-KV kernels. This started at concurrency 192–512 and now covers concurrency 1–48 as well. For MLA decode, the split-K factor is now picked per index stream (for example the compress-ratio-128 HCA stream), because layers in the same model have KV lengths that differ by orders of magnitude.

For MoE, prefill at concurrency 192–512 runs AITER's MegaMoEv2 kernel ([sglang#35619](https://github.com/sgl-project/sglang/pull/35619)), written in FlyDSL, which fuses dispatch, GEMM and combine and uses 8-way expert parallelism over the MoRI symmetric heap. Together with FP8 KV this added 11% throughput per GPU at concurrency 256 and let the curve extend to concurrency 384 and 512. At concurrency 512 we also turned on expert-parallel load balancing for the MegaMoE prefill.

In SGLang's scheduler, we moved speculative decoding from EAGLE/MTP to DSpark, which produces a block of draft tokens in one pass using the draft head that ships with DeepSeek-V4-Pro-0813; the draft length is 6 at low concurrency and 3 at high concurrency. In PD disaggregation a request normally waits for decode to bootstrap it before prefill begins, which at high concurrency is just queueing. Letting prefill start optimistically, with the speculative KV owned by the request, cut P90 TTFT by 27.7% at concurrency 256, and removing a host sync from DSpark prefill slot expansion cut another 13–16% at concurrency 128–256. Decode routing had been silently inheriting the prefill's consistent-hashing policy; it now uses round-robin, and from Oct 21 power-of-two-choices at concurrency 192–512.

MoRI handles both the PD KV transfer and the MoE all-to-all, over standard RoCE between 8-GPU nodes. Finally, the Oct 28 image (`v0.5.22-rocm10-mi35x-20261026`) moved to ROCm 10. Compared with the ROCm 7.2 build from a week earlier, P90 interactivity went up at every high-concurrency point; at concurrency 512 it went from 51.8 to 60.6 tok/s/user, while throughput per GPU went down slightly, from 83.7k to 79.8k tok/s.

## Summary

On DeepSeek-V4-Pro AgentX, MI355X peaks at 99.1M tokens per $1 TCO, which is 1.50× B200, 1.20× B300 and slightly above GB200 NVL72. At the same interactivity it reaches up to 1.85× B200 and 2.26× B300. GB300 NVL72 is still ahead, and the closest we get is 0.95× at 65 tok/s/user. Over ten weeks on the same GPUs, peak throughput per GPU went up 3.5×.

MoRI UMBP and the KVCache Store Linker are upstream in SGLang, the AITER kernels ship in the ROCm SGLang images, and every recipe and log is in the public InferenceX CI. Next we plan to bring MoRI UMBP to ATOM, vLLM and llm-d, and to apply the same approach to newer agentic models.

## Acknowledgements

Thanks to the SGLang community for design reviews and quick upstreaming, to SemiAnalysis for AgentX and the InferenceX CI infrastructure, and to the AMD MoRI, SGLang, AITER and ROCm teams.

## References

1. SemiAnalysis InferenceX, DeepSeek-V4 dashboard. https://inferencex.semianalysis.com/inference/deepseek-v4
2. SemiAnalysis AI Cloud TCO Model (source of the $/GPU/hr rates). https://semianalysis.com/ai-cloud-tco-model/
3. AMD, *Rebuilding Agentic AI from First Principles for AMD GPU*, with Moonshot AI. https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html
4. MoRI UMBP on DeepSeek-V4 PD disaggregation (earlier post). https://github.com/ichbinblau/dsv4-pd-disagg/tree/main/dsv4-umbp
5. InferenceX recipe history for this benchmark: PRs [#2823](https://github.com/SemiAnalysisAI/InferenceX/pull/2823), [#3188](https://github.com/SemiAnalysisAI/InferenceX/pull/3188), [#3256](https://github.com/SemiAnalysisAI/InferenceX/pull/3256), [#3664](https://github.com/SemiAnalysisAI/InferenceX/pull/3664), [#4186](https://github.com/SemiAnalysisAI/InferenceX/pull/4186), [#4571](https://github.com/SemiAnalysisAI/InferenceX/pull/4571).

*Methodology: numbers come from the InferenceX public API (`/api/v1/benchmarks?model=DeepSeek-V4-Pro`), retrieved 2026-10-30. The snapshots and the script that generates every figure and table are in `data/` and `make_figures.py`. Same-interactivity values use InferenceX's `iso_interactivity.py`, which ports the dashboard's interpolation.*
