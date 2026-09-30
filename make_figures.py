"""Regenerate the blog figures.

Numbers are copied from the InferenceX site API (tput_per_gpu,
p90_full_response_intvty, p90_ttft, server_*_cache_hit_rate) for:
  baseline  official InferenceX run of 2026-08-21 (v0.5.17 image)
  pr2823    run 34926284365 (InferenceX #2823, merged 2026-09-16)
  pr3256    run 35879254139 (InferenceX #3256)
  s1..s4    intermediate PR #2823 sweeps 33837253408, 34368415051,
            34550987258, 34615909904
"""
import os

import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

AMD_RED = "#ED1C24"
DARK = "#2B2B2B"
GREY = "#8C8C8C"
LIGHT = "#D9D9D9"
# softer red / amber / green: baseline or worse -> better
SOFT_RED = "#E15A4E"
SOFT_AMBER = "#EFA73C"
SOFT_GREEN = "#3A9E48"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# conc: (tput/GPU tok/s, P90 interactivity tok/s/user)
BASELINE = {1: (653, 109.9), 2: (729, 104.1), 4: (1310, 93.3), 8: (2349, 87.4),
            16: (4455, 69.8), 32: (8544, 54.6), 48: (11519, 49.2),
            64: (13713, 40.4), 96: (16634, 38.0), 128: (19544, 35.3),
            192: (22932, 23.5)}
PR3256 = {4: (2974, 164.0), 16: (8246, 152.8), 32: (15948, 115.9),
          48: (21407, 102.4), 192: (46563, 59.3), 256: (55848, 54.1)}


def fig1_pareto():
    fig, ax = plt.subplots(figsize=(9, 5.6))
    series = [
        (BASELINE, "Aug 21 baseline (SGLang v0.5.17)", GREY, "o", "--"),
        (PR3256, "Sep 23: + TP4 prefill on UMBP, optimistic prefill, HCA split-K", AMD_RED, "D", "-"),
    ]
    offsets = [(5, 4), (0, 8)]
    for (data, label, color, marker, ls), off in zip(series, offsets):
        pts = sorted(data.items(), key=lambda kv: kv[1][1])
        xs = [v[1] for _, v in pts]
        ys = [v[0] for _, v in pts]
        ax.plot(xs, ys, ls, color=color, marker=marker, lw=2, ms=6, label=label)
        for c, (y, x) in data.items():
            ax.annotate(f"c{c}", (x, y), textcoords="offset points", xytext=off,
                        ha="left" if data is BASELINE else "center", fontsize=8, color=color)
    ax.set_xlabel("P90 interactivity (tok/s/user)")
    ax.set_ylabel("Token throughput per GPU (tok/s)")
    ax.set_title("DeepSeek-V4-Pro FP4 agentic PD disaggregation on AMD Instinct MI355X")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_pareto.png"), dpi=200)
    plt.close(fig)


def fig2_waterfall():
    # concurrency 192, 16 GPUs (1P1D, TP8/DP8) at every step
    steps = [
        ("Aug 21\nbaseline", 22932, 33.56, "base"),
        ("FP4 indexer\n+ DP/EP1\nredesign", 34704, 20.50, "amd"),
        ("Concurrency-\nscaled\nscheduling", 37254, 16.21, "amd"),
        ("v0.5.19\nkernels", 38247, 35.28, "amd"),
        ("UMBP linker\nreplaces\nHiCache", 43683, 11.91, "amd"),
        ("Pro-0813\nacceptance\n(calibration)", 47415, 13.47, "cal"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})
    prev = 0
    for i, (label, val, _, kind) in enumerate(steps):
        if kind == "base":
            ax.bar(i, val, color=GREY)
        else:
            color = AMD_RED if kind == "amd" else LIGHT
            hatch = None if kind == "amd" else "//"
            ax.bar(i, val - prev, bottom=prev, color=color, hatch=hatch, edgecolor=DARK, lw=0.5)
            ax.text(i, val + 600, f"{val / 1000:.1f}k\n(+{(val / prev - 1) * 100:.0f}%)",
                    ha="center", fontsize=9)
        if kind == "base":
            ax.text(i, val / 2, f"{val / 1000:.1f}k", ha="center", va="center",
                    fontsize=9, color="white")
        prev = val
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("Token throughput per GPU (tok/s)")
    ax.set_title("Throughput per GPU at concurrency 192 (2.07x)")
    ax.set_ylim(0, 54000)

    xs = range(len(steps))
    ttft = [s[2] for s in steps]
    ax2.plot(xs, ttft, "-o", color=AMD_RED, lw=2)
    for x, y in zip(xs, ttft):
        ax2.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8)
    ax2.annotate("HiCache host tier\nunder pressure", (3, 35.28), textcoords="offset points",
                 xytext=(-95, -45), fontsize=8, color=GREY,
                 arrowprops=dict(arrowstyle="->", color=GREY))
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([s[0] for s in steps], fontsize=7)
    ax2.set_ylabel("P90 TTFT (s)")
    ax2.set_title("P90 TTFT at concurrency 192 (-60%)")
    ax2.set_ylim(0, 42)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_c192_breakdown.png"), dpi=200)
    plt.close(fig)


def fig3_umbp():
    concs = ["c16", "c32", "c48"]
    gpu_hit = [67.5, 45.7, 21.8]
    dram_hit = [29.7, 51.7, 75.4]
    miss = [2.8, 2.5, 2.8]
    tp8 = [6133, 12738, 17272]   # Sep 15: TP8 prefill + TP8 decode, 16 GPUs
    tp4 = [8246, 15948, 21407]   # Sep 25: TP4 prefill + TP8 decode + MoRI UMBP, 12 GPUs

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(concs))
    ax.bar(x, gpu_hit, color=SOFT_GREEN, label="GPU HBM prefix hit")
    ax.bar(x, dram_hit, bottom=gpu_hit, color=SOFT_AMBER, label="MoRI UMBP DRAM hit")
    ax.bar(x, miss, bottom=[a + b for a, b in zip(gpu_hit, dram_hit)], color=SOFT_RED,
           label="Recomputed")
    for i, d in enumerate(dram_hit):
        ax.text(i, gpu_hit[i] + d / 2, f"{d:.0f}%", ha="center", va="center",
                color=DARK, fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(concs)
    ax.set_ylabel("Share of prompt tokens (%)")
    ax.set_title("Where TP4 prefill finds its KV")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)

    w = 0.38
    ax2.bar([i - w / 2 for i in x], tp8, w, color=SOFT_RED, label="Sep 15: TP8 prefill, 16 GPUs")
    ax2.bar([i + w / 2 for i in x], tp4, w, color=SOFT_GREEN, label="Sep 25: TP4 prefill + MoRI UMBP, 12 GPUs")
    for i in x:
        ax2.text(i + w / 2, tp4[i] + 400, f"+{(tp4[i] / tp8[i] - 1) * 100:.0f}%",
                 ha="center", fontsize=9)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(concs)
    ax2.set_ylabel("Token throughput per GPU (tok/s)")
    ax2.set_title("Same load on 25% fewer GPUs")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_umbp_tp4_prefill.png"), dpi=200)
    plt.close(fig)


def fig5_c192_summary():
    # concurrency 192: Aug 21 baseline vs. Sep 25 (run 35879254139)
    panels = [
        ("Throughput per GPU", "tok/s", 22932, 46563, lambda v: f"{v / 1000:.1f}k", "2.03×"),
        ("P90 TTFT (lower is better)", "s", 33.6, 11.8, lambda v: f"{v:.1f} s", "–65%"),
        ("P90 interactivity", "tok/s/user", 23.5, 59.3, lambda v: f"{v:.1f}", "2.5×"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, unit, before, after, fmt, change) in zip(axes, panels):
        ax.bar([0, 1], [before, after], 0.6, color=[SOFT_RED, SOFT_GREEN])
        top = max(before, after)
        for x, v in zip([0, 1], [before, after]):
            ax.text(x, v + top * 0.02, fmt(v), ha="center", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Aug 21", "Sep 25"])
        ax.set_ylabel(unit)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(f"{title}: {change}")
        ax.grid(axis="x", visible=False)
    fig.suptitle("DeepSeek-V4-Pro FP4 agentic on MI355X at concurrency 192", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_c192_aug_vs_sep.png"), dpi=200)
    plt.close(fig)


def fig6_hicache_vs_umbp():
    # Controlled A/B, 2026-09-17 (hicache_vs_umbp_dspark_g3_ab_20260917): same
    # 1P1D TP8+TP8 recipe, consistent_hashing router, 600 GB DRAM; only the KV
    # offload backend differs. conc: (tput/GPU tok/s, P90 interactivity tok/s/user)
    # c256 UMBP result JSON is truncated in the archive: 815,254 tok/s / 16 GPUs
    # and 1000 / 22.75 ms P90 TPOT from the run summary.
    hicache = {128: (30698, 61.77), 256: (47054, 42.44)}
    umbp = {128: (31518, 61.74), 256: (50953, 43.96)}
    dark_red = "#A32A22"  # deuteranopia-safe against SOFT_GREEN

    fig, ax = plt.subplots(figsize=(9, 5.6))
    series = [
        (hicache, "A: SGLang HiCache", dark_red, "o", "--", (-12, -4), "right"),
        (umbp, "B: MoRI UMBP linker", SOFT_GREEN, "D", "-", (10, -4), "left"),
    ]
    for data, label, color, marker, ls, off, ha in series:
        pts = sorted(data.items(), key=lambda kv: kv[1][1])
        ax.plot([v[1] for _, v in pts], [v[0] for _, v in pts], ls, color=color,
                marker=marker, lw=2, ms=8, mec="white", mew=1.5, label=label)
        for c, (y, x) in data.items():
            ax.annotate(f"c{c}", (x, y), textcoords="offset points", xytext=off,
                        ha=ha, fontsize=9, color=DARK)
    for c in (128, 256):
        (ya, xa), (yb, xb) = hicache[c], umbp[c]
        ttft = {128: "–34%", 256: "–35%"}[c]
        above = c == 256
        ax.annotate(f"+{(yb / ya - 1) * 100:.1f}% tput/GPU\n{ttft} P90 TTFT",
                    (xb, yb if above else ya), textcoords="offset points",
                    xytext=(0, 14 if above else -22), ha="center",
                    va="bottom" if above else "top", fontsize=9, color=DARK)
    ax.set_xlim(37, 66)
    ax.set_ylim(24000, 56000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("P90 interactivity (tok/s/user)")
    ax.set_ylabel("Token throughput per GPU (tok/s)")
    ax.set_title("HiCache vs. MoRI UMBP linker, same recipe "
                 "(DeepSeek-V4-Pro FP4, 1P1D TP8 + TP8, 16× MI355X)", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_hicache_vs_umbp_pareto.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_pareto()
    fig2_waterfall()
    fig3_umbp()
    fig5_c192_summary()
    fig6_hicache_vs_umbp()
    print("wrote", sorted(os.listdir(OUT)))
