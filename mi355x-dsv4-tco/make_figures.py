"""Regenerate the figures and tables for the MI355X DeepSeek-V4-Pro TCO blog.

Data: InferenceX public API snapshots saved under data/
  dsv4_latest_20261030.json  /api/v1/benchmarks?model=DeepSeek-V4-Pro (retrieved 2026-10-30)
  dsv4_<date>.json           same endpoint with &date=<date>&exact=true

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
LABEL = {"mi355x": "MI355X (MoRI-SGLang + MoRI UMBP)", "b200": "B200", "b300": "B300",
         "gb200": "GB200 NVL72", "gb300": "GB300 NVL72"}
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


LATEST = load("dsv4_latest_20261030.json")


def series(hw):
    """MI355X: the MoRI-SGLang PD line. NVIDIA: best of every published framework."""
    rows = [r for r in LATEST if r["hardware"] == hw]
    if hw == "mi355x":
        rows = [r for r in rows if r["framework"] == "mori-sglang"]
    return points(rows)


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
        lw = 2.6 if hw == "mi355x" else 1.8
        ax.plot(xs, ys, "-", color=COLOR[hw], marker=MARKER[hw], lw=lw, ms=7,
                mec="white", mew=1.2, label=LABEL[hw], zorder=3 if hw == "mi355x" else 2)
        i = max(range(len(ys)), key=ys.__getitem__)
        ax.annotate(f"{LABEL[hw].split(' (')[0]}  {ys[i]:.0f}M", (xs[i], ys[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9, color=INK)
    ax.set_xlabel("P90 interactivity (tok/s/user)")
    ax.set_ylabel("Total tokens per $1 TCO (millions)")
    ax.set_title("DeepSeek-V4-Pro FP4, AgentX: tokens per $1 TCO (Rent – 3-year commit)", fontsize=12)
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 125)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Source: InferenceX public API, 2026-10-30. Pareto frontier per system; "
             "NVIDIA = best of all published frameworks.", fontsize=8, color=MUTED)
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
                ax.text(x, 0.04, "n/a", ha="center", fontsize=7, color=MUTED, rotation=90)
                continue
            r = (a / COST_RENT["mi355x"]) / (b / COST_RENT[hw])
            ax.bar(x, r, w * 0.9, color=COLOR[hw], label=None)
            ax.text(x, r + 0.03, f"{r:.2f}×", ha="center", fontsize=8, color=INK)
    ax.axhline(1.0, color=INK, lw=1)
    ax.text(len(tiers) - 0.45, 1.02, "parity", fontsize=8, color=INK, ha="right")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t} tok/s/user" for t in tiers])
    ax.set_ylabel(r"MI355X tokens/\$ ÷ NVIDIA tokens/\$")
    ax.set_title("MI355X tokens-per-dollar advantage at the same P90 interactivity (Rent tier)",
                 fontsize=12)
    ax.set_ylim(0, 2.6)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=COLOR[hw], label=LABEL[hw]) for hw in hws],
              loc="upper left", frameon=False, fontsize=9, ncol=4)
    fig.text(0.01, 0.01, "n/a = that system has no published point at this interactivity "
             "(the chart does not extrapolate).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "iso-interactivity-ratio.png"), dpi=200)
    plt.close(fig)


PROGRESS = [
    ("2026-08-21", "Aug 21  baseline (v0.5.17)", "#c3c2b7"),
    ("2026-09-25", "Sep 25  + MoRI UMBP linker, TP4 prefill, DSpark", "#86b6ef"),
    ("2026-10-04", "Oct 4  + MegaMoE prefill, FP8 KV", "#2a78d6"),
    ("2026-10-28", "Oct 28  + FP8 KV at low conc, EPLB, P2C routing, ROCm 10", "#e34948"),
]


def mori(date):
    return points([r for r in load(f"dsv4_{date}.json")
                   if r["hardware"] == "mi355x" and r["framework"] == "mori-sglang"])


def fig_progress():
    fig, ax = plt.subplots(figsize=(10, 6))
    for date, label, color in PROGRESS:
        f = frontier(mori(date))
        ax.plot([p["interactivity"] for p in f], [p["throughput"] for p in f], "-o",
                color=color, lw=2.4 if date == "2026-10-28" else 1.8, ms=6, mec="white",
                mew=1.2, label=label)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("P90 interactivity (tok/s/user)")
    ax.set_ylabel("Token throughput per GPU (tok/s)")
    ax.set_title("DeepSeek-V4-Pro FP4 AgentX on MI355X, ten weeks of software", fontsize=12)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 85000)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mi355x-progress-aug-oct.png"), dpi=200)
    plt.close(fig)


def fig_c192():
    def at(date, conc):
        p = [q for q in mori(date) if q["conc"] == conc][0]
        return p["throughput"], p["row"]["metrics"]["p90_ttft"], p["interactivity"]
    a, b = at("2026-08-21", 192), at("2026-10-28", 192)
    panels = [("Throughput per GPU", "tok/s", a[0], b[0], lambda v: f"{v / 1000:.1f}k"),
              ("P90 TTFT (lower is better)", "s", a[1], b[1], lambda v: f"{v:.1f} s"),
              ("P90 interactivity", "tok/s/user", a[2], b[2], lambda v: f"{v:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, unit, before, after, fmt) in zip(axes, panels):
        ax.bar([0, 1], [before, after], 0.6, color=["#c3c2b7", "#e34948"])
        top = max(before, after)
        for x, v in zip([0, 1], [before, after]):
            ax.text(x, v + top * 0.02, fmt(v), ha="center", fontsize=10, color=INK)
        change = f"{after / before:.1f}×" if "TTFT" not in title else f"–{(1 - after / before) * 100:.0f}%"
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Aug 21", "Oct 28"])
        ax.set_ylabel(unit)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(f"{title}: {change}", fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Same 16 MI355X GPUs, same concurrency 192", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mi355x-c192-aug-vs-oct.png"), dpi=200)
    plt.close(fig)


def tables(tiers):
    print("## iso-interactivity tokens per $1 TCO (millions), Rent | Own; tput/GPU")
    mi = series("mi355x")
    for iv in tiers:
        a = iso(mi, iv)
        line = f"iv {iv:>3}: MI355X " + (f"{a * 3600 / 2.9 / 1e6:6.1f}|{a * 3600 / 1.5 / 1e6:6.1f} ({a / 1e3:.1f}k)" if a else "   --  ")
        for hw in ["b200", "b300", "gb200", "gb300"]:
            b = iso(series(hw), iv)
            if b and a:
                rr = (a / COST_RENT["mi355x"]) / (b / COST_RENT[hw])
                ro = (a / COST_OWN["mi355x"]) / (b / COST_OWN[hw])
                line += f" | {hw} {b * 3600 / COST_RENT[hw] / 1e6:6.1f}M ({b / 1e3:.1f}k) rent {rr:.2f}x own {ro:.2f}x perf {a / b:.2f}x"
            else:
                line += f" | {hw} --"
        print(line)
    print("## peak tokens per $ (Rent | Own)")
    for hw in COST_RENT:
        p = max(series(hw), key=lambda q: q["throughput"])
        print(f"{hw}: {p['throughput'] * 3600 / COST_RENT[hw] / 1e6:.1f}M | "
              f"{p['throughput'] * 3600 / COST_OWN[hw] / 1e6:.1f}M at {p['interactivity']:.1f} "
              f"tok/s/user, {p['row']['framework']} c{p['conc']}")
    print("## MI355X progress, iso-interactivity throughput/GPU")
    for iv in [60, 70, 80, 100]:
        a, b = iso(mori("2026-08-21"), iv), iso(mori("2026-10-28"), iv)
        print(iv, a, b, (b / a) if a and b else None)


if __name__ == "__main__":
    tiers = [70, 80, 100, 120, 150]
    fig_tco_frontier()
    fig_iso_ratio(tiers)
    fig_progress()
    fig_c192()
    tables([60, 65, 70, 80, 90, 100, 120, 150])
    print("wrote", sorted(os.listdir(OUT)))
