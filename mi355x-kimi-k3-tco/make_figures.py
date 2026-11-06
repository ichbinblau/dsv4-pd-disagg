"""Regenerate the figures and tables for the MI355X Kimi-K3 TCO blog.

Data: InferenceX public API snapshots saved under data/
  kimik3_latest_20261030.json  /api/v1/benchmarks?model=Kimi-K3 (retrieved 2026-10-30)
  kimik3_<date>.json           same endpoint with &date=<date>&exact=true

Conventions match the dashboard:
  * P90 interactivity = 1 / p90_full_response_itl (agentic rows)
  * tokens per $1 TCO = tput_per_gpu * 3600 / ($/GPU/hr), HW_REGISTRY rates
  * iso-interactivity values use iso_interactivity.py, the 1:1 port of the
    chart's Pareto-frontier + monotone-spline interpolation
  * point labels follow parallelismLabel() in InferenceX-app parallelism-label.ts

Figures use the dashboard's look: grey canvas, NVIDIA greens and MI355X reds
sampled from dashboard screenshots, dashed rings on points with KV offload on.
"""
import json
import os
from datetime import date as _date

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.transforms import Bbox

from iso_interactivity import (hermite_interpolate, interpolate_metric, monotone_slopes,
                               pareto_front_upper_left)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# $/GPU/hr from packages/constants/src/gpu-keys.ts (InferenceX-app, 2026-10-30)
COST_RENT = {"mi355x": 2.9, "b200": 3.7, "b300": 4.25, "gb200": 4.0, "gb300": 5.0}
COST_OWN = {"mi355x": 1.5, "b200": 1.73, "b300": 2.26, "gb200": 1.86, "gb300": 2.31}
LABEL = {"mi355x": "MI355X", "b200": "B200", "b300": "B300", "gb200": "GB200 NVL72",
         "gb300": "GB300 NVL72"}

# dashboard colors, sampled from the DeepSeek-V4 dashboard screenshots
BG = "#eaebec"
GRID = "#d3d6db"
AXIS = "#4a4a4a"
INK = "#1f1f1f"
MUTED = "#6b6e74"
COLOR = {"mi355x": "#c53732", "b200": "#007636", "b300": "#278733", "gb200": "#5a9830",
         "gb300": "#83a730"}
REDS = ["#f4a39a", "#ec6a5c", "#cf3027", "#970004"]   # oldest -> newest
FONT = "Liberation Sans"
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    _p = os.path.join("/usr/share/fonts/truetype/liberation", _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)


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
    """Best of every published framework for each system, as the dashboard does."""
    return points([r for r in LATEST if r["hardware"] == hw])


def stack(date, fw):
    return points([r for r in load(f"kimik3_{date}.json")
                   if r["hardware"] == "mi355x" and r["framework"] == fw])


def frontier(pts):
    f = pareto_front_upper_left(pts, lambda p: p["interactivity"], lambda p: p["throughput"])
    return sorted(f, key=lambda p: p["interactivity"])


def iso(pts, iv):
    return interpolate_metric(pts, target_iv=iv, metric_key="throughput")


def smooth(xs, ys):
    """Dense samples of the dashboard's monotone cubic (d3.curveMonotoneX)."""
    if len(xs) < 2:
        return xs, ys
    m = monotone_slopes(xs, ys)
    ox, oy = [], []
    for i in range(len(xs) - 1):
        for k in range(24):
            x = xs[i] + (xs[i + 1] - xs[i]) * k / 24
            ox.append(x)
            oy.append(hermite_interpolate(xs, ys, m, x))
    return ox + [xs[-1]], oy + [ys[-1]]


def parallelism_label(r):
    """Aggregated branch of parallelismLabel() in parallelism-label.ts."""
    tp, ep = r["decode_tp"] or r["prefill_tp"], r["decode_ep"] or r["prefill_ep"]
    dcp = r["metrics"].get("dcp_size")
    suffix = f"/DCP{dcp}" if dcp and dcp > 1 else ""
    if ep and ep > 1 and tp == ep:
        return f"{'DEP' if r['decode_dp_attention'] else 'TEP'}{tp}{suffix}"
    return f"{'DPA' if r['decode_dp_attention'] else ''}TP{tp}{suffix}"


def dash_axes(ax, xgrid=False):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, lw=0.9)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, lw=0.9)
    else:
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS, labelsize=10)


def title(fig, text, sub=None, y=0.93):
    fig.text(0.02, y, text, fontsize=15, fontweight="bold", color=INK)
    if sub:
        fig.text(0.02, y - 0.05, sub, fontsize=10.5, color=AXIS)


def draw_curve(ax, f, color, lw=1.9):
    xs = [p["interactivity"] for p in f]
    ys = [p["y"] for p in f]
    sx, sy = smooth(xs, ys)
    ax.plot(sx, sy, "-", color=color, lw=lw, zorder=3, solid_capstyle="round")
    ax.scatter(xs, ys, s=30, color=color, zorder=5, linewidths=0)
    halo = [p for p in f if p["row"]["offload_mode"] == "on"]
    ax.scatter([p["interactivity"] for p in halo], [p["y"] for p in halo], s=210,
               facecolors="none", edgecolors="#2b2b2b", linewidths=0.9,
               linestyles=(0, (2.2, 2.2)), zorder=4)


def callout(ax, text, xy, xytext, color, ha="left"):
    return ax.annotate(text, xy, xytext=xytext, textcoords="data", ha=ha, va="center",
                       fontsize=10.5, fontweight="bold", color="white", zorder=7,
                       bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.25", fc=color, ec="none"),
                       arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=0, shrinkB=5))


def legend_panel(fig, x, y, w, h, header, entries):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=BG,
                                      edgecolor="#c4c7cc", lw=1.0))
    fig.text(x + 0.015, y + h - 0.05, header, fontsize=12.5, color=INK)
    yy = y + h - 0.11
    for label, color in entries:
        fig.patches.append(Circle((x + 0.022, yy + 0.007), 0.0055, transform=fig.transFigure,
                                  facecolor=color, edgecolor="none"))
        fig.text(x + 0.035, yy, label, fontsize=11, color=INK)
        yy -= 0.058


def fig_tco_frontier():
    """Figure 1: tokens per $1 TCO frontier per system (best framework for each)."""
    order = ["b200", "b300", "gb200", "gb300", "mi355x"]
    # callout per system: anchor interactivity (nearest frontier point), box position
    anchors = {"mi355x": (20.7, (60, 22.5), "left"), "gb300": (34.2, (95, 19.4), "left"),
               "gb200": (101.3, (140, 4.6), "left"), "b200": (69.7, (6, 1.3), "left"),
               "b300": (21.6, (6, 3.3), "left")}
    with plt.rc_context({"font.family": FONT}):
        fig = plt.figure(figsize=(13.4, 7.6), facecolor=BG)
        ax = fig.add_axes((0.065, 0.11, 0.70, 0.75))
        dash_axes(ax, xgrid=True)
        for hw in order:
            f = frontier(series(hw))
            for p in f:
                p["y"] = p["throughput"] * 3600 / COST_RENT[hw] / 1e6
            draw_curve(ax, f, COLOR[hw], lw=2.4 if hw == "mi355x" else 1.9)
            target, xy, ha = anchors[hw]
            p = min(f, key=lambda q: abs(q["interactivity"] - target))
            callout(ax, f"{LABEL[hw]} FP4", (p["interactivity"], p["y"]), xy, COLOR[hw], ha)
        ax.set_xlim(0, 440)
        ax.set_ylim(0, 25)
        ax.set_xticks(range(0, 441, 40))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.0f}M"))
        ax.set_xlabel("P90 Interactivity (tok/s/user)", fontsize=12, color="#2b2b2b", labelpad=10)
        ax.set_ylabel(r"Total Tokens per \$1 TCO (tok/\$)", fontsize=12, color="#2b2b2b", labelpad=8)
        title(fig, r"Kimi K3 2.8T Agentic Total Tokens per \$1 TCO vs. P90 Interactivity",
              "Rent – 3 Year Commit. Each curve is one system's best published framework.")
        legend_panel(fig, 0.785, 0.46, 0.198, 0.40, "Systems",
                     [(LABEL[hw], COLOR[hw]) for hw in ["gb300", "gb200", "b300", "b200", "mi355x"]])
        fig.text(0.065, 0.025, "Dashed circles mark points with KV offload on. "
                 "Source: InferenceX public API, retrieved 2026-10-30.", fontsize=9, color=MUTED)
        fig.savefig(os.path.join(OUT, "tco-frontier-vs-nvidia.png"), dpi=150, facecolor=BG)
        plt.close(fig)


def fig_iso_ratio(tiers):
    """Figure 2: bars grouped by NVIDIA system; the greens tie it to Figure 1, the
    system names on the axis carry identity."""
    mi = series("mi355x")
    hws = ["b200", "b300", "gb200", "gb300"]
    with plt.rc_context({"font.family": FONT}):
        fig, ax = plt.subplots(figsize=(12, 5.8), facecolor=BG)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20)
        dash_axes(ax)
        n = len(tiers)
        w = 0.8 / n
        for g, hw in enumerate(hws):
            other = series(hw)
            for i, iv in enumerate(tiers):
                x = g + (i - (n - 1) / 2) * w
                ax.text(x, -0.035, str(iv), transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=8.5, color=MUTED)
                a, b = iso(mi, iv), iso(other, iv)
                if a is None or b is None:
                    ax.text(x, 0.06, "n/a", ha="center", va="bottom", fontsize=8,
                            color="#8c8f94", rotation=90)
                    continue
                r = (a / COST_RENT["mi355x"]) / (b / COST_RENT[hw])
                ax.bar(x, r, w * 0.82, color=COLOR[hw], zorder=3)
                ax.text(x, r + 0.04, f"{r:.2f}×", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.12", fc=BG, ec="none"))
            ax.text(g, -0.11, LABEL[hw], transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=12, fontweight="bold", color=INK)
        ax.axhline(1.0, color=COLOR["mi355x"], lw=1.3, ls=(0, (4, 3)), zorder=4)
        ax.text(len(hws) - 0.5, 1.04, "parity", ha="right", va="bottom", fontsize=9,
                color=COLOR["mi355x"], fontweight="bold")
        ax.set_xlim(-0.5, len(hws) - 0.5)
        ax.set_xticks([])
        ax.set_ylim(0, 3.3)
        ax.set_ylabel(r"MI355X tokens/\$ ÷ NVIDIA tokens/\$", fontsize=11, color="#2b2b2b")
        title(fig, r"MI355X Tokens per \$1 TCO vs. NVIDIA at the Same P90 Interactivity",
              "Kimi K3 2.8T, Agentic, Rent – 3 Year Commit. Above the red line MI355X is cheaper per token.",
              y=0.92)
        fig.text(0.07, 0.035, "Small numbers under the bars are P90 interactivity in tok/s/user. "
                 "n/a = no published point at that interactivity (no extrapolation). "
                 "B300 results date from 2026-08-16.", fontsize=9, color=MUTED)
        fig.savefig(os.path.join(OUT, "iso-interactivity-ratio.png"), dpi=170, facecolor=BG)
        plt.close(fig)


PROGRESS = {
    "vllm": ["2026-08-12", "2026-09-17", "2026-10-15", "2026-10-27"],
    "atom": ["2026-08-21", "2026-09-10", "2026-09-29", "2026-10-15"],
}
def fig_progress():
    """Figure 3: frontier of each MI355X stack at four of its submissions."""
    with plt.rc_context({"font.family": FONT}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=BG, sharey=True)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.06)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, fw, name in [(axes[0], "vllm", "vLLM"), (axes[1], "atom", "ATOM")]:
            dash_axes(ax, xgrid=True)
            ax.set_xlim(0, 230)
            ax.set_ylim(0, 19500)
            taken = []
            curves = []
            for date, color in zip(PROGRESS[fw], REDS):
                f = frontier(stack(date, fw))
                for p in f:
                    p["y"] = p["throughput"]
                draw_curve(ax, f, color, lw=2.3 if color == REDS[-1] else 1.8)
                curves.append((date, color, f))
                for p in f:
                    cx, cy = ax.transData.transform((p["interactivity"], p["y"]))
                    r = 8 * fig.dpi / 72
                    taken.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))
            handles = [Line2D([], [], marker="o", ls="-", lw=2, ms=7, color=color, label=date)
                       for date, color, _ in curves]
            leg = ax.legend(handles=handles, loc="upper right", fontsize=10.5, frameon=True,
                            facecolor=BG, edgecolor="#c4c7cc", framealpha=1, borderpad=0.8,
                            labelspacing=0.6, handlelength=1.6)
            leg.get_frame().set_boxstyle("round,pad=0.2,rounding_size=0.8")
            fig.canvas.draw()
            taken.append(leg.get_window_extent(renderer).expanded(1.03, 1.05))
            inside = ax.get_window_extent(renderer)
            for date, color, f in reversed(curves):
                for p in f:
                    for dx, dy, ha, va in [(0, 9, "center", "bottom"), (0, -10, "center", "top"),
                                           (10, 3, "left", "bottom"), (-10, 3, "right", "bottom")]:
                        t = ax.annotate(parallelism_label(p["row"]), (p["interactivity"], p["y"]),
                                        textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                                        fontsize=7.5, fontweight="bold", color=INK, zorder=6)
                        bb = t.get_window_extent(renderer).expanded(1.06, 1.25)
                        if (bb.x0 >= inside.x0 and bb.x1 <= inside.x1 and bb.y0 >= inside.y0
                                and bb.y1 <= inside.y1 and not any(bb.overlaps(o) for o in taken)):
                            taken.append(bb)
                            break
                        t.remove()
            ax.set_title(f"MI355X ({name})", fontsize=12.5, fontweight="bold", color=INK, loc="left")
            ax.set_xlabel("P90 Interactivity (tok/s/user)", fontsize=11.5, color="#2b2b2b")
        axes[0].set_ylabel("Token Throughput per Chip (tok/s/chip)", fontsize=11.5, color="#2b2b2b")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
        title(fig, "Kimi K3 2.8T Agentic Token Throughput per Chip vs. P90 Interactivity",
              "One 8× MI355X node. Each line is one submission; labels show the parallelism layout.")
        fig.savefig(os.path.join(OUT, "mi355x-progress-aug-oct.png"), dpi=150, facecolor=BG)
        plt.close(fig)


def fig_timeline():
    """Figure 4: peak throughput per GPU of every vLLM and ATOM submission."""
    start = _date(2026, 8, 12)
    with plt.rc_context({"font.family": FONT}):
        fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=BG)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.12)
        dash_axes(ax, xgrid=False)
        dates_by_fw = {}
        for f in sorted(os.listdir(DATA)):
            if not f.startswith("kimik3_2026-"):
                continue
            d = f[7:17]
            for fw in ("vllm", "atom"):
                pts = stack(d, fw)
                if pts:
                    dates_by_fw.setdefault(fw, []).append((d, max(p["throughput"] for p in pts)))
        handles = []
        for fw, color, name in [("vllm", "#970004", "vLLM"), ("atom", "#f36358", "ATOM")]:
            xs = [(_date.fromisoformat(d) - start).days for d, _ in dates_by_fw[fw]]
            ys = [y for _, y in dates_by_fw[fw]]
            ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=6.5, mec=BG, mew=1.5, zorder=4)
            handles.append(Line2D([], [], marker="o", ls="-", lw=2.2, ms=7, color=color, label=name))
            for i, (x, y) in enumerate(zip(xs, ys)):
                if i in (0, len(xs) - 1) or y == max(ys):
                    # ATOM's first point sits just under the vLLM line, so label it below
                    dy = -15 if (fw == "atom" and i == 0) else 11
                    ax.annotate(f"{y / 1000:.1f}k", (x, y), textcoords="offset points",
                                xytext=(0, dy), ha="center", va="center", fontsize=9.5,
                                fontweight="bold", color=INK, zorder=6,
                                bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none"))
        leg = ax.legend(handles=handles, loc="upper left", fontsize=11, frameon=True, facecolor=BG,
                        edgecolor="#c4c7cc", framealpha=1, borderpad=0.8, handlelength=1.8)
        leg.get_frame().set_boxstyle("round,pad=0.2,rounding_size=0.8")
        ticks = [_date(2026, 8, 12), _date(2026, 9, 1), _date(2026, 9, 15), _date(2026, 10, 1),
                 _date(2026, 10, 15), _date(2026, 10, 27)]
        ax.set_xticks([(t - start).days for t in ticks])
        ax.set_xticklabels([t.strftime("%b %-d") for t in ticks], fontsize=10)
        ax.set_xlim(-3, (ticks[-1] - start).days + 12)
        ax.set_ylim(0, 20000)
        ax.set_yticks(range(0, 20001, 4000))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))
        ax.set_ylabel("Peak token throughput per GPU (tok/s)", fontsize=11, color="#2b2b2b")
        title(fig, "Every MI355X Kimi K3 Submission, Aug 12 to Oct 27",
              "Peak throughput per GPU of each public InferenceX run, one 8× MI355X node.", y=0.92)
        fig.savefig(os.path.join(OUT, "submission-timeline.png"), dpi=170, facecolor=BG)
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


if __name__ == "__main__":
    fig_tco_frontier()
    fig_iso_ratio([25, 30, 40, 60, 80, 100])
    fig_progress()
    fig_timeline()
    tables([25, 30, 40, 50, 60, 80, 100])
    print("wrote", sorted(os.listdir(OUT)))
