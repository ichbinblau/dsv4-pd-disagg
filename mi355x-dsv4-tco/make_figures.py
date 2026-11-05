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
import textwrap

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import Bbox
from matplotlib.patches import Circle, FancyBboxPatch, Patch

from iso_interactivity import (hermite_interpolate, interpolate_metric, monotone_slopes,
                               pareto_front_upper_left)

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


# InferenceX dashboard colors, sampled from the Figure 1 / Figure 3 screenshots
DASH_GREEN = {"b200": "#007636", "b300": "#278733", "gb200": "#5a9830", "gb300": "#83a730"}
DASH_RED = {"2026-08-21": "#f36358", "2026-10-28": "#970004"}
DASH_FONT = "Liberation Sans"
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    _path = os.path.join("/usr/share/fonts/truetype/liberation", _f)
    if os.path.exists(_path):
        font_manager.fontManager.addfont(_path)


def _dash_axes(ax):
    ax.set_facecolor("#eaebec")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#4a4a4a")
    ax.grid(True, axis="y", color="#d3d6db", lw=0.9, alpha=1)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4a4a4a", labelsize=10)


def fig_iso_ratio(tiers):
    """Figure 2: bars grouped by NVIDIA system. Systems are named on the axis, so the
    dashboard's four greens only tie the chart to Figure 1; they do not carry identity."""
    mi = series("mi355x")
    hws = ["b200", "b300", "gb200", "gb300"]
    with plt.rc_context({"font.family": DASH_FONT}):
        fig, ax = plt.subplots(figsize=(12, 5.8), facecolor="#eaebec")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20)
        _dash_axes(ax)
        n = len(tiers)
        w = 0.8 / n
        for g, hw in enumerate(hws):
            other = series(hw)
            for i, iv in enumerate(tiers):
                x = g + (i - (n - 1) / 2) * w
                ax.text(x, -0.035, str(iv), transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=8.5, color="#6b6e74")
                a_, b_ = iso(mi, iv), iso(other, iv)
                if a_ is None or b_ is None:
                    ax.text(x, 0.05, "n/a", ha="center", va="bottom", fontsize=8,
                            color="#8c8f94", rotation=90)
                    continue
                r = (a_ / COST_RENT["mi355x"]) / (b_ / COST_RENT[hw])
                ax.bar(x, r, w * 0.82, color=DASH_GREEN[hw], zorder=3)
                ax.text(x, r + 0.035, f"{r:.2f}×", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color="#1f1f1f", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.12", fc="#eaebec", ec="none"))
            ax.text(g, -0.11, LABEL[hw], transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=12, fontweight="bold", color="#1f1f1f")
        ax.axhline(1.0, color="#c53732", lw=1.3, ls=(0, (4, 3)), zorder=4)
        ax.text(len(hws) - 0.5, 1.03, "parity", ha="right", va="bottom", fontsize=9,
                color="#c53732", fontweight="bold")
        ax.set_xlim(-0.5, len(hws) - 0.5)
        ax.set_xticks([])
        ax.set_ylim(0, 2.5)
        ax.set_ylabel(r"MI355X tokens/\$ ÷ NVIDIA tokens/\$", fontsize=11, color="#2b2b2b")
        fig.text(0.02, 0.92, r"MI355X Tokens per \$1 TCO vs. NVIDIA at the Same P90 Interactivity",
                 fontsize=15, fontweight="bold", color="#1f1f1f")
        fig.text(0.02, 0.865, "DeepSeek V4 Pro 0813 1.6T, Agentic, Rent – 3 Year Commit. "
                 "Above the red line MI355X is cheaper per token.", fontsize=10.5, color="#4a4a4a")
        fig.text(0.07, 0.035, "Small numbers under the bars are P90 interactivity in tok/s/user. "
                 "n/a = no published point at that interactivity (no extrapolation).",
                 fontsize=9, color="#6b6e74")
        fig.savefig(os.path.join(OUT, "iso-interactivity-ratio.png"), dpi=170, facecolor="#eaebec")
        plt.close(fig)


PROGRESS = [
    ("2026-08-21", "Baseline (SGLang v0.5.17)", "#f4a39a"),
    ("2026-09-25", "+ MoRI UMBP linker, TP4 prefill, DSpark", "#ec6a5c"),
    ("2026-10-04", "+ MegaMoE prefill, FP8 KV", "#cf3027"),
    ("2026-10-28", "+ FP8 KV at low conc, EPLB, P2C routing, ROCm 10", "#9e0b0b"),
]
# callout per date: anchor concurrency, box position (data coords), alignment, full label?
# The two ends carry the full series name like the dashboard; the middle two show the
# date only so they fit beside their curves (the legend names the series).
CALLOUT = {"2026-08-21": (32, (59, 8500), "left", True),
           "2026-09-25": (256, (49, 55800), "right", False),
           "2026-10-04": (512, (43.5, 77300), "right", False),
           "2026-10-28": (256, (86, 65500), "left", True)}

DASH_BG = "#ebecef"
DASH_GRID = "#d3d6db"
DASH_AXIS = "#4a4a4a"
DASH_MUTED = "#6b6e74"


def mori(date):
    return points([r for r in load(f"dsv4_{date}.json")
                   if r["hardware"] == "mi355x" and r["framework"] == "mori-sglang"])


def _segment_label(tp, ep, dpa, pp=None, dcp=None):
    """Port of configSegmentLabel() in InferenceX-app parallelism-label.ts."""
    suffix = (f"PP{pp}" if pp and pp > 1 else "") + (f"/DCP{dcp}" if dcp and dcp > 1 else "")
    if ep and ep > 1 and tp == ep:
        return f"{'DEP' if dpa else 'TEP'}{tp}{suffix}"
    prefix = "DPA" if dpa else ""
    if not ep or ep <= 1:
        return f"{prefix}TP{tp}{suffix}"
    return f"{prefix}EP{ep}{suffix}"


def parallelism_label(r):
    """Disaggregated multi-node branch of parallelismLabel() in parallelism-label.ts."""
    m = r["metrics"]
    pre = _segment_label(r["prefill_tp"], r["prefill_ep"], r["prefill_dp_attention"],
                         m.get("prefill_pp"), m.get("prefill_dcp_size"))
    dec = _segment_label(r["decode_tp"], r["decode_ep"], r["decode_dp_attention"],
                         m.get("decode_pp"), m.get("decode_dcp_size"))
    pw = r["prefill_num_workers"] if r["prefill_num_workers"] is not None else 1
    dw = r["decode_num_workers"] if r["decode_num_workers"] is not None else 1
    return f"{pw}x{pre}+{dw}x{dec}"


def _smooth(f):
    """Dense samples of the dashboard's monotone cubic (d3.curveMonotoneX) through f."""
    xs = [p["interactivity"] for p in f]
    ys = [p["throughput"] for p in f]
    if len(xs) < 2:
        return xs, ys
    m = monotone_slopes(xs, ys)
    out_x, out_y = [], []
    for i in range(len(xs) - 1):
        for k in range(24):
            x = xs[i] + (xs[i + 1] - xs[i]) * k / 24
            out_x.append(x)
            out_y.append(hermite_interpolate(xs, ys, m, x))
    out_x.append(xs[-1])
    out_y.append(ys[-1])
    return out_x, out_y


def fig_progress():
    """Dashboard-style throughput-per-chip chart of the MoRI-SGLang recipe over time."""
    fig = plt.figure(figsize=(13.4, 7.6), facecolor=DASH_BG)
    ax = fig.add_axes((0.065, 0.12, 0.70, 0.76), facecolor=DASH_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(DASH_AXIS)
    ax.grid(True, color=DASH_GRID, lw=0.9, alpha=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=DASH_AXIS, labelsize=10)
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 90000)
    ax.set_xticks(range(0, 261, 20))
    ax.set_yticks(range(0, 90001, 10000))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))
    ax.set_xlabel("P90 Interactivity (tok/s/user)", color="#2b2b2b", fontsize=12, labelpad=12)
    ax.set_ylabel("Token Throughput per Chip (tok/s/chip)", color="#2b2b2b", fontsize=12, labelpad=10)
    fig.text(0.02, 0.945, "DeepSeek V4 Pro 0813 1.6T Agentic Token Throughput per Chip "
             "vs. P90 Interactivity", fontsize=15, fontweight="bold", color="#1f1f1f")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = []

    def place(x, y, text, candidates, **kw):
        for dx, dy, ha, va in candidates:
            t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, **kw)
            bb = t.get_window_extent(renderer).expanded(1.06, 1.25)
            inside = ax.get_window_extent(renderer)
            if (bb.x0 >= inside.x0 and bb.x1 <= inside.x1 + 40 and bb.y1 <= inside.y1
                    and bb.y0 >= inside.y0
                    and not any(bb.overlaps(o) for o in taken)):
                taken.append(bb)
                return t
            t.remove()
        return None

    series = []
    for date, desc, color in PROGRESS:
        f = frontier(mori(date))
        series.append((date, desc, color, f))
        sx, sy = _smooth(f)
        ax.plot(sx, sy, "-", color=color, lw=1.8, zorder=3, solid_capstyle="round")
        ax.scatter([p["interactivity"] for p in f], [p["throughput"] for p in f], s=34,
                   color=color, zorder=5, linewidths=0)
        ax.scatter([p["interactivity"] for p in f], [p["throughput"] for p in f], s=230,
                   facecolors="none", edgecolors="#2b2b2b", linewidths=0.9,
                   linestyles=(0, (2.2, 2.2)), zorder=4)

    # dashed rings are obstacles for the parallelism labels
    for _, _, _, f in series:
        for p in f:
            cx, cy = ax.transData.transform((p["interactivity"], p["throughput"]))
            r = 8 * fig.dpi / 72
            taken.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))

    # callouts first so the parallelism labels route around them
    for date, desc, color, f in series:
        conc, (bx, by), ha, full = CALLOUT[date]
        p = [q for q in f if q["conc"] == conc][0]
        text = f"MI355X (MoRI SGLang) • {date}" if full else date
        t = ax.annotate(text, (p["interactivity"], p["throughput"]),
                        xytext=(bx, by), textcoords="data", ha=ha, va="center",
                        fontsize=10.5, fontweight="bold", color="white", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.25", fc=color, ec="none"),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=0, shrinkB=5))
        taken.append(t.get_window_extent(renderer).expanded(1.02, 1.15))

    for date, desc, color, f in reversed(series):
        for p in f:
            place(p["interactivity"], p["throughput"], parallelism_label(p["row"]),
                  [(0, 9, "center", "bottom"), (0, -10, "center", "top"),
                   (10, 3, "left", "bottom"), (-10, 3, "right", "bottom"),
                   (10, -4, "left", "top"), (-10, -4, "right", "top")],
                  fontsize=8, fontweight="bold", color="#1f1f1f", zorder=6)

    # legend panel, styled after the dashboard's side panel
    panel = FancyBboxPatch((0.785, 0.385), 0.198, 0.495, boxstyle="round,pad=0,rounding_size=0.012",
                           transform=fig.transFigure, facecolor=DASH_BG, edgecolor="#c4c7cc", lw=1.0)
    fig.patches.append(panel)
    fig.text(0.800, 0.835, "MI355X (MoRI SGLang)", fontsize=13, color="#1f1f1f")
    y = 0.775
    for date, desc, color, _ in series:
        fig.patches.append(Circle((0.807, y + 0.006), 0.0055, transform=fig.transFigure,
                                  facecolor=color, edgecolor="none"))
        fig.text(0.820, y, date, fontsize=12, color="#1f1f1f")
        lines = textwrap.wrap(desc, 36)
        for j, line in enumerate(lines):
            fig.text(0.820, y - 0.030 - 0.022 * j, line, fontsize=8.5, color=DASH_MUTED)
        y -= 0.068 + 0.022 * len(lines)
    fig.text(0.065, 0.025, "Optimal (Pareto) points only. Labels show the prefill + decode "
             "parallelism layout. Source: InferenceX public API, retrieved 2026-10-30.",
             fontsize=9, color=DASH_MUTED)
    fig.savefig(os.path.join(OUT, "mi355x-progress-aug-oct.png"), dpi=150, facecolor=DASH_BG)
    plt.close(fig)


def fig_c192():
    """Figure 4: same 16 GPUs at concurrency 192, Aug 21 vs Oct 28, dashboard colors."""
    def at(date, conc):
        p = [q for q in mori(date) if q["conc"] == conc][0]
        return p["throughput"], p["row"]["metrics"]["p90_ttft"], p["interactivity"]
    a, b = at("2026-08-21", 192), at("2026-10-28", 192)
    panels = [("Throughput per GPU", "tok/s", a[0], b[0], lambda v: f"{v / 1000:.1f}k",
               f"{b[0] / a[0]:.1f}× higher"),
              ("P90 TTFT", "s", a[1], b[1], lambda v: f"{v:.1f} s",
               f"{a[1] / b[1]:.1f}× lower"),
              ("P90 interactivity", "tok/s/user", a[2], b[2], lambda v: f"{v:.1f}",
               f"{b[2] / a[2]:.1f}× higher")]
    colors = [DASH_RED["2026-08-21"], DASH_RED["2026-10-28"]]
    with plt.rc_context({"font.family": DASH_FONT}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor="#eaebec")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.12, wspace=0.32)
        for ax, (title, unit, before, after, fmt, change) in zip(axes, panels):
            _dash_axes(ax)
            ax.bar([0, 1], [before, after], 0.58, color=colors, zorder=3)
            top = max(before, after)
            for x, v in zip([0, 1], [before, after]):
                ax.text(x, v + top * 0.025, fmt(v), ha="center", va="bottom", fontsize=10.5,
                        fontweight="bold", color="#1f1f1f")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["2026-08-21", "2026-10-28"], fontsize=10.5)
            ax.set_ylabel(unit, fontsize=10.5, color="#2b2b2b")
            ax.set_ylim(0, top * 1.2)
            if unit == "tok/s":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(
                    lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))
            ax.set_title(title, fontsize=12, fontweight="bold", color="#1f1f1f", loc="left")
            ax.text(1.0, 1.02, change, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, fontweight="bold", color=DASH_RED["2026-10-28"])
        fig.text(0.02, 0.9, "MI355X (MoRI SGLang): Same 16 Chips at Concurrency 192, "
                 "2026-08-21 vs. 2026-10-28", fontsize=15, fontweight="bold", color="#1f1f1f")
        fig.savefig(os.path.join(OUT, "mi355x-c192-aug-vs-oct.png"), dpi=170, facecolor="#eaebec")
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
    # Figure 1 is now a dashboard screenshot; fig_tco_frontier() would overwrite it.
    fig_iso_ratio(tiers)
    # Figure 3 is now a dashboard screenshot; fig_progress() would overwrite it.
    fig_c192()
    tables([60, 65, 70, 80, 90, 100, 120, 150])
    print("wrote", sorted(os.listdir(OUT)))
