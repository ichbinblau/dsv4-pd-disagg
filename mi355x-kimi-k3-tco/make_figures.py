"""Regenerate the figures and tables for the MI355X Kimi-K3 TCO blog.

Data: InferenceX public API snapshots saved under data/
  kimik3_latest_20261030.json  /api/v1/benchmarks?model=Kimi-K3 (retrieved 2026-10-30)
  kimik3_<date>.json           same endpoint with &date=<date>&exact=true

Conventions match the dashboard:
  * P90 interactivity = 1 / p90_full_response_itl (agentic rows)
  * tokens per $1 TCO = tput_per_gpu * 3600 / ($/GPU/hr), HW_REGISTRY rates
  * iso-interactivity values use iso_interactivity.py, the 1:1 port of the
    chart's Pareto-frontier + monotone-spline interpolation
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from iso_interactivity import interpolate_metric, pareto_front_upper_left

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# $/GPU/hr from packages/constants/src/gpu-keys.ts (InferenceX-app, 2026-10-30)
COST_RENT = {"mi355x": 2.9, "b200": 3.7, "b300": 4.25, "gb200": 4.0, "gb300": 5.0}
COST_OWN = {"mi355x": 1.5, "b200": 1.73, "b300": 2.26, "gb200": 1.86, "gb300": 2.31}
LABEL = {"mi355x": "MI355X (vLLM / ATOM, 8 GPUs)", "b200": "B200", "b300": "B300",
         "gb200": "GB200 NVL72", "gb300": "GB300 NVL72"}
SHORT = {"mi355x": "MI355X", "b200": "B200", "b300": "B300", "gb200": "GB200 NVL72",
         "gb300": "GB300 NVL72"}
# validated with dataviz validate_palette.js (light): all hard checks pass;
# the two low-contrast hues are always direct-labeled
COLOR = {"mi355x": "#e34948", "b200": "#2a78d6", "b300": "#1baf7a",
         "gb200": "#eda100", "gb300": "#6250d6"}
MARKER = {"mi355x": "D", "b200": "o", "b300": "s", "gb200": "^", "gb300": "v"}
INK = "#2b2b2b"
MUTED = "#8c8c8c"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.edgecolor": MUTED,
})


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return [r for r in json.load(f) if r["benchmark_type"] == "agentic_traces"]


def points(rows):
    out = []
    for r in rows:
        itl = r["metrics"].get("p90_full_response_itl")
        if itl:
            out.append({"interactivity": 1 / itl, "throughput": r["metrics"]["tput_per_gpu"],
                        "conc": r["conc"], "row": r})
    return out


LATEST = load("kimik3_latest_20261030.json")


def series(hw):
    """Best of every published framework for each system."""
    return points([r for r in LATEST if r["hardware"] == hw])


def frontier(pts):
    f = pareto_front_upper_left(pts, lambda p: p["interactivity"], lambda p: p["throughput"])
    return sorted(f, key=lambda p: p["interactivity"])


def iso(pts, iv):
    return interpolate_metric(pts, target_iv=iv, metric_key="throughput")


def fig_tco_frontier():
    fig, ax = plt.subplots(figsize=(10, 6))
    for hw in ["b200", "b300", "gb200", "gb300", "mi355x"]:
        f = frontier(series(hw))
        xs = [p["interactivity"] for p in f]
        ys = [p["throughput"] * 3600 / COST_RENT[hw] / 1e6 for p in f]
        ax.plot(xs, ys, "-", color=COLOR[hw], marker=MARKER[hw],
                lw=2.6 if hw == "mi355x" else 1.8, ms=7, mec="white", mew=1.2,
                label=LABEL[hw], zorder=3 if hw == "mi355x" else 2)
        i = max(range(len(ys)), key=ys.__getitem__)
        off = {"b300": (-4, 10), "b200": (8, 2), "gb200": (6, -14)}.get(hw, (6, 6))
        ax.annotate(f"{SHORT[hw]}  {ys[i]:.1f}M", (xs[i], ys[i]), textcoords="offset points",
                    xytext=off, fontsize=9, color=INK)
    ax.set_xlabel("P90 interactivity (tok/s/user)")
    ax.set_ylabel("Total tokens per $1 TCO (millions)")
    ax.set_title("Kimi-K3 2.8T FP4, AgentX: tokens per $1 TCO (Rent – 3-year commit)", fontsize=12)
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 25)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Source: InferenceX public API, 2026-10-30. Pareto frontier per system, "
             "best of all published frameworks.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "tco-frontier-vs-nvidia.png"), dpi=200)
    plt.close(fig)


def fig_iso_ratio(tiers):
    mi = series("mi355x")
    hws = ["b200", "b300", "gb200", "gb300"]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.2
    for j, hw in enumerate(hws):
        other = series(hw)
        for i, iv in enumerate(tiers):
            a, b = iso(mi, iv), iso(other, iv)
            x = i + (j - 1.5) * w
            if a is None or b is None:
                ax.text(x, 0.06, "n/a", ha="center", fontsize=7, color=MUTED, rotation=90)
                continue
            r = (a / COST_RENT["mi355x"]) / (b / COST_RENT[hw])
            ax.bar(x, r, w * 0.9, color=COLOR[hw])
            ax.text(x, r + 0.04, f"{r:.2f}×", ha="center", fontsize=8, color=INK)
    ax.axhline(1.0, color=INK, lw=1)
    ax.text(len(tiers) - 0.45, 1.03, "parity", fontsize=8, color=INK, ha="right")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t} tok/s/user" for t in tiers])
    ax.set_ylabel(r"MI355X tokens/\$ ÷ NVIDIA tokens/\$")
    ax.set_title("MI355X tokens-per-dollar advantage at the same P90 interactivity (Rent tier)",
                 fontsize=12)
    ax.set_ylim(0, 3.6)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=COLOR[hw], label=LABEL[hw]) for hw in hws],
              loc="upper right", frameon=False, fontsize=9, ncol=4)
    fig.text(0.01, 0.01, "n/a = that system has no published point at this interactivity "
             "(the chart does not extrapolate). B300 results date from 2026-08-16.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "iso-interactivity-ratio.png"), dpi=200)
    plt.close(fig)


PROGRESS = {
    "vllm": [("2026-08-12", "Aug 12  first submission", "#c3c2b7"),
             ("2026-09-10", "Sep 10  + DCP8, DRAM offload", "#86b6ef"),
             ("2026-10-12", "Oct 12  + AITER MLA draft attention", "#2a78d6"),
             ("2026-10-27", "Oct 27  + Gluon KDA, MonoKernel decode", "#e34948")],
    "atom": [("2026-08-21", "Aug 21  first submission", "#c3c2b7"),
             ("2026-09-10", "Sep 10  + DCP8, LMCache, ReplaySSM", "#86b6ef"),
             ("2026-09-29", "Sep 29  + FlyDSL FP8 prefill attention", "#2a78d6"),
             ("2026-10-15", "Oct 15  + DSpark at high conc, gated RMSNorm", "#e34948")],
}


def mi_at(date, fw):
    return points([r for r in load(f"kimik3_{date}.json")
                   if r["hardware"] == "mi355x" and r["framework"] == fw])


def fig_progress():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (fw, title) in zip(axes, [("vllm", "vLLM"), ("atom", "ATOM")]):
        for date, label, color in PROGRESS[fw]:
            f = frontier(mi_at(date, fw))
            ax.plot([p["interactivity"] for p in f], [p["throughput"] for p in f], "-o",
                    color=color, lw=2.4 if color == "#e34948" else 1.8, ms=6, mec="white",
                    mew=1.2, label=label)
        ax.set_title(f"MI355X + {title}", fontsize=11)
        ax.set_xlabel("P90 interactivity (tok/s/user)")
        ax.set_xlim(0, 220)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].set_ylabel("Token throughput per GPU (tok/s)")
    axes[0].set_ylim(0, 19000)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    fig.suptitle("Kimi-K3 FP4 AgentX on one 8× MI355X node, ten weeks of software", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mi355x-progress-aug-oct.png"), dpi=200)
    plt.close(fig)


def tables(tiers):
    print("## iso-interactivity tokens per $1 TCO (millions), Rent; ratios rent/own/perf")
    mi = series("mi355x")
    for iv in tiers:
        a = iso(mi, iv)
        line = f"iv {iv:>3}: MI355X " + (f"{a * 3600 / 2.9 / 1e6:5.1f}M ({a / 1e3:.1f}k)" if a else "  --  ")
        for hw in ["b200", "b300", "gb200", "gb300"]:
            b = iso(series(hw), iv)
            if a and b:
                rr = (a / COST_RENT["mi355x"]) / (b / COST_RENT[hw])
                ro = (a / COST_OWN["mi355x"]) / (b / COST_OWN[hw])
                line += f" | {hw} {b * 3600 / COST_RENT[hw] / 1e6:5.1f}M ({b / 1e3:.1f}k) r{rr:.2f} o{ro:.2f} p{a / b:.2f}"
            else:
                line += f" | {hw} --"
        print(line)
    print("## peak tokens per $ (Rent | Own)")
    for hw in COST_RENT:
        p = max(series(hw), key=lambda q: q["throughput"])
        print(f"{hw}: {p['throughput'] * 3600 / COST_RENT[hw] / 1e6:.1f}M | "
              f"{p['throughput'] * 3600 / COST_OWN[hw] / 1e6:.1f}M at {p['interactivity']:.1f} "
              f"tok/s/user, {p['row']['framework']} c{p['conc']} tput {p['throughput']:.0f}")
    print("## MI355X progress, iso-interactivity throughput/GPU")
    for fw, first, last in [("vllm", "2026-08-12", "2026-10-27"), ("atom", "2026-08-21", "2026-10-15")]:
        for iv in [20, 30, 40, 50, 60, 80, 100]:
            a, b = iso(mi_at(first, fw), iv), iso(mi_at(last, fw), iv)
            print(fw, iv, a and round(a), b and round(b), (round(b / a, 2) if a and b else None))


if __name__ == "__main__":
    fig_tco_frontier()
    fig_iso_ratio([25, 30, 40, 60, 80, 100])
    fig_progress()
    tables([20, 25, 30, 40, 50, 60, 80, 100, 120])
    print("wrote", sorted(os.listdir(OUT)))
