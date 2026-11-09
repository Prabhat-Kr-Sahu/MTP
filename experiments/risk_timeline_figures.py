#!/usr/bin/env python3
"""Figures and summary tables for RISK_TIMELINE.md, from the committed CSVs.

Reads experiments/results/risk_timeline/{scores,truth,tracks}.csv and
plotted_cases.json (written by `plot_risk_timeline.py --plot`), writes
experiments/figures/risk_timeline_doc_*.png and results/risk_timeline/summary.csv.
Local and model-free: seconds.

    python experiments/risk_timeline_figures.py
"""
import json
import sys
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from experiments.plot_risk_timeline import (  # noqa: E402
    AFTER, BASELINES, BEFORE, FIGURES, LABEL_DIST, MODELS, OUT, TITLES, case_rows,
)
from src.config import DT  # noqa: E402

INK, INK_2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                          "#c3c2b7", "#fcfcfb")
BASE_C, TUNED_C, RED = "#2a78d6", "#eb6834", "#e34948"
BLUES = LinearSegmentedColormap.from_list("blue", [
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CASES = ("collision", "missed", "near_miss", "safe", "false_alarm")
SHORT = {"collision": "Collision (detected)", "missed": "Collision (missed)",
         "near_miss": "Near miss", "safe": "No collision (safe)", "false_alarm": "False alarm"}
GRID_T = np.round(np.arange(-BEFORE, AFTER + 1) * DT, 1)
TUNED = tuple(m for m in MODELS if m.endswith("_tuned"))


def _style(ax, zero=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    if zero:
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")


def _save(fig, name):
    path = FIGURES / f"risk_timeline_doc_{name}.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def _pair(scores, case, info, source="cv"):
    return case_rows(scores[scores.source == source], case, info)


def _wide(rows, metric="p_coll_gauss"):
    """(models x time) on the full 0.1 s grid; NaN where the pair was not scored."""
    return rows.pivot_table(index="model", columns="t", values=metric).reindex(
        index=list(MODELS), columns=GRID_T)


def ref_time(case, info, scores, truth):
    """Bird's-eye snapshot time: t* for the events, closest point for safe,
    the median-model peak for the false alarm."""
    if case == "safe":
        d = truth[(truth.case == info["from_window"]) & (truth.neighbour == info["neighbour"])]
        return float(d.loc[d.dist.idxmin(), "t"])
    if case == "false_alarm":
        return float(_pair(scores, case, info).groupby("t").p_coll_gauss.median().idxmax())
    return 0.0


# --------------------------------------------------------------------------- #
#  Summary table
# --------------------------------------------------------------------------- #
def summary(scores, cases):
    rows = []
    for case, info in cases.items():
        windows = scores[scores.case == info.get("from_window", case)].t.nunique()
        for source in ("cv", "pred"):
            pair = _pair(scores, case, info, source)
            for model, r in pair.groupby("model"):
                r = r.set_index("t").p_coll_gauss
                hit = r[r >= 0.1]
                rows.append({"case": case, "source": source, "model": model,
                             "coverage": len(r) / windows, "peak": r.max(),
                             "t_peak": r.idxmax(), "first_ge_0.1": hit.index.min() if len(hit) else np.nan,
                             "p_at_0": r.get(0.0, np.nan)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    return out


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_overview(scores, truth, cases):
    """One column per case: median and IQR across baselines and across tuned,
    above the true distance."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.6), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.12,
                                          "wspace": 0.12})
    top = 0
    for j, case in enumerate(CASES):
        info = cases[case]
        wide = _wide(_pair(scores, case, info))
        ax = axes[0, j]
        _style(ax)
        for group, colour, style in ((BASELINES, BASE_C, "-"), (TUNED, TUNED_C, "--")):
            block = wide.loc[list(group)]
            q1, med, q3 = (block.quantile(q) for q in (0.25, 0.5, 0.75))
            ax.fill_between(GRID_T, q1, q3, color=colour, alpha=0.18, linewidth=0)
            ax.plot(GRID_T, med, color=colour, linestyle=style, linewidth=2)
            top = max(top, float(np.nanmax(q3.to_numpy())))
        ax.set_title(SHORT[case], fontsize=10, color=INK, loc="left")
        d = truth[(truth.case == info.get("from_window", case))
                  & (truth.neighbour == info["neighbour"])].sort_values("t")
        ax = axes[1, j]
        _style(ax)
        ax.plot(d.t, d.dist, color=INK, linewidth=1.6)
        ax.axhline(LABEL_DIST, color=INK_2, linewidth=0.9, linestyle="--")
        ax.set_ylim(0, 60)
        ax.set_xlabel("time (s)", color=INK_2, fontsize=9)
        if j:
            axes[0, j].tick_params(labelleft=False)
            ax.tick_params(labelleft=False)
    for ax in axes[0]:
        ax.set_ylim(0, top * 1.08)
    axes[0, 0].set_ylabel("p_coll_gauss (cv)", color=INK_2, fontsize=9)
    axes[1, 0].set_ylabel("true distance (m)", color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], color=BASE_C, linewidth=2, label="8 baselines: median, IQR band"),
               plt.Line2D([], [], color=TUNED_C, linewidth=2, linestyle="--",
                          label="7 tuned: median, IQR band"),
               plt.Line2D([], [], color=INK_2, linewidth=0.9, linestyle="--",
                          label="2 m collision label")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=3,
               bbox_to_anchor=(0.99, 1.02), labelcolor=INK_2)
    fig.suptitle("Vehicle 1102724: risk over time in all five cases",
                 x=0.07, y=1.02, ha="left", fontsize=12, color=INK)
    _save(fig, "overview")


METRIC_ROWS = (  # name, axis label, riskier direction
    ("p_coll_gauss", "p_coll_gauss\nprobability", "up"),
    ("p_coll_mc", "p_coll_mc\nprobability", "up"),
    ("s_field_cal", "s_field_cal\nfield value", "up"),
    ("min_dist", "min_dist\nm", "down"),
    ("ttc_true", "ttc_true, s\ngap = no contact\nclipped at 5 s", "down"),
    ("tet", "tet\ns below 2 m", "up"),
)


def fig_metrics(scores, truth, cases, source):
    """Six headline risk measures x five cases: median and IQR across the
    baselines and across the tuned models, one y-range per measure."""
    metrics = [m for m in METRIC_ROWS if m[0] in scores.columns]
    fig, axes = plt.subplots(len(metrics) + 1, 5, figsize=(17, 2.1 * (len(metrics) + 1)),
                             facecolor=SURFACE, sharex=True,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.1})
    for i, (metric, label, riskier) in enumerate(metrics):
        top = 0.0
        for j, case in enumerate(CASES):
            ax = axes[i, j]
            _style(ax)
            rows = _pair(scores, case, cases[case], source)
            if rows.empty:
                ax.text(0.5, 0.5, "never scored", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=MUTED)
            else:
                wide = _wide(rows.replace([np.inf, -np.inf], np.nan), metric)
                for group, colour, style in ((BASELINES, BASE_C, "-"), (TUNED, TUNED_C, "--")):
                    block = wide.loc[list(group)]
                    q1, med, q3 = (block.quantile(q) for q in (0.25, 0.5, 0.75))
                    ax.fill_between(GRID_T, q1, q3, color=colour, alpha=0.18, linewidth=0)
                    ax.plot(GRID_T, med, color=colour, linestyle=style, linewidth=1.8)
                    if q3.notna().any():
                        top = max(top, float(np.nanmax(q3.to_numpy())))
            if i == 0:
                ax.set_title(SHORT[case], fontsize=10, color=INK, loc="left")
            if j:
                ax.tick_params(labelleft=False)
        for ax in axes[i]:
            # ttc_true can exceed the 5 s horizon by orders of magnitude; clip there.
            ax.set_ylim(0, 5.0 if metric == "ttc_true" else (top * 1.08 if top > 0 else 1))
        arrow = "↑ riskier" if riskier == "up" else "↓ riskier"
        axes[i, 0].set_ylabel(f"{label}\n{arrow}", color=INK_2, fontsize=8)
    for j, case in enumerate(CASES):
        info = cases[case]
        ax = axes[-1, j]
        _style(ax)
        d = truth[(truth.case == info.get("from_window", case))
                  & (truth.neighbour == info["neighbour"])].sort_values("t")
        ax.plot(d.t, d.dist, color=INK, linewidth=1.6)
        ax.axhline(LABEL_DIST, color=INK_2, linewidth=0.9, linestyle="--")
        ax.set_ylim(0, 60)
        ax.set_xlabel("time (s)", color=INK_2, fontsize=9)
        if j:
            ax.tick_params(labelleft=False)
    axes[-1, 0].set_ylabel("true distance\nm", color=INK_2, fontsize=8)
    handles = [plt.Line2D([], [], color=BASE_C, linewidth=2, label="8 baselines: median, IQR band"),
               plt.Line2D([], [], color=TUNED_C, linewidth=2, linestyle="--",
                          label="7 tuned: median, IQR band")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=2,
               bbox_to_anchor=(0.99, 0.915), labelcolor=INK_2)
    fig.suptitle(f"Six risk measures over time, vehicle 1102724, source {source}",
                 x=0.07, y=0.915, ha="left", fontsize=12, color=INK)
    _save(fig, f"metrics_{source}")


def fig_heatmap(scores, cases):
    """Every model x every 0.1 s window, one panel per case, one colour scale."""
    wides = {c: _wide(_pair(scores, c, cases[c])) for c in CASES}
    vmax = max(float(np.nanmax(w.to_numpy())) for w in wides.values() if w.notna().any().any())
    cmap = BLUES.copy()
    cmap.set_bad("#c3c2b7")
    fig, axes = plt.subplots(len(CASES), 1, figsize=(12, 15), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.35})
    for ax, case in zip(axes, CASES):
        image = ax.imshow(np.ma.masked_invalid(wides[case].to_numpy()), aspect="auto",
                          cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest",
                          extent=(GRID_T[0] - DT / 2, GRID_T[-1] + DT / 2, len(MODELS), 0))
        ax.set_yticks(np.arange(len(MODELS)) + 0.5, MODELS, fontsize=7, color=INK_2)
        ax.axhline(len(BASELINES), color=SURFACE, linewidth=2)
        ax.axvline(0, color=INK, linewidth=0.8, linestyle=":")
        ax.set_title(SHORT[case], fontsize=10, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[-1].set_xlabel("time (s); 0 = closest approach (safe, false alarm: the collision moment)",
                        color=INK_2, fontsize=9)
    bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    bar.set_label("p_coll_gauss (cv); grey = pair not scored", color=INK_2, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Every model, every window", x=0.1, y=0.91, ha="left", fontsize=12, color=INK)
    _save(fig, "heatmap")


def fig_coverage(scores, cases):
    """Which windows each source can score for the case's own pair."""
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    _style(ax)
    labels = []
    for i, case in enumerate(CASES):
        for k, source in enumerate(("cv", "pred")):
            y = i * 3 + k
            pair = _pair(scores, case, cases[case], source)
            ts = np.sort(pair[pair.model == MODELS[0]].t.unique())
            ax.broken_barh([(t - DT / 2, DT) for t in ts], (y - 0.35, 0.7), color=BASE_C,
                           linewidth=0)
            labels.append((y, f"{SHORT[case]}  ·  {source}  ({len(ts) / len(GRID_T):.0%})"))
    ax.set_yticks([y for y, _ in labels], [t for _, t in labels], fontsize=8, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(GRID_T[0] - DT, GRID_T[-1] + DT)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s)", color=INK_2, fontsize=9)
    ax.set_title("Windows in which the case's own pair is scored (identical for all 15 models)",
                 fontsize=11, color=INK, loc="left")
    _save(fig, "coverage")


def fig_gauss_vs_mc(scores, cases):
    """The two probability measures, median over the 15 models, cv."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), facecolor=SURFACE, sharey=True,
                             gridspec_kw={"wspace": 0.1})
    for ax, case in zip(axes, CASES):
        _style(ax)
        pair = _pair(scores, case, cases[case])
        for metric, colour, style in (("p_coll_gauss", INK, "-"), ("p_coll_mc", RED, "--")):
            med = pair.groupby("t")[metric].median().reindex(GRID_T)
            ax.plot(GRID_T, med, color=colour, linestyle=style, linewidth=2)
        ax.set_title(SHORT[case], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("time (s)", color=INK_2, fontsize=9)
        ax.set_xlim(GRID_T[0], GRID_T[-1])
    axes[0].set_ylabel("median over 15 models (cv)", color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], color=INK, linewidth=2, label="p_coll_gauss: worst single step"),
               plt.Line2D([], [], color=RED, linewidth=2, linestyle="--",
                          label="p_coll_mc: any step (64 Monte Carlo draws)")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=2,
               bbox_to_anchor=(0.99, 1.08), labelcolor=INK_2)
    _save(fig, "gauss_vs_mc")


def _lane_edges(tracks):
    """Approximate lane boundaries: midpoints between per-lane median lateral positions."""
    centres = tracks.groupby("lane_id").local_x.median().sort_values()
    centres = centres[centres.index <= 6]          # 7+ are ramps on US-101
    mids = (centres.to_numpy()[1:] + centres.to_numpy()[:-1]) / 2
    half = np.median(np.diff(centres.to_numpy())) / 2
    return np.concatenate([[centres.iloc[0] - half], mids, [centres.iloc[-1] + half]])


def _box(ax, x, y, length, width, **kw):
    ax.add_patch(Rectangle((x - length / 2, y - width / 2), length, width, **kw))


def fig_birdseye(tracks, scores, truth, cases):
    """Top-down view of each case, anchored at the motorcycle's position at the
    snapshot time; boxes to scale, centred on the tracked point as the risk
    measures assume; paths over the whole -8 s .. +1 s window."""
    fig, axes = plt.subplots(len(CASES), 1, figsize=(13, 16), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.45})
    for ax, case in zip(axes, CASES):
        info = cases[case]
        window = info.get("from_window", case)
        t_ref = ref_time(case, info, scores, truth)
        tr = tracks[tracks.case == window]
        moto, other = info["target"], info["neighbour"]
        anchor = tr[(tr.vehicle_id == moto) & (tr.t == t_ref)].local_y.iloc[0]
        tr = tr.assign(x=tr.local_y - anchor)
        _style(ax, zero=False)
        ax.grid(False)
        for edge in _lane_edges(tr):
            ax.axhline(edge, color=GRID, linewidth=1, linestyle="--", zorder=0)
        snap = tr[tr.t == t_ref]
        for _, v in snap.iterrows():
            if v.vehicle_id in (moto, other):
                continue
            _box(ax, v.x, v.local_x, v.v_length, v.v_width, facecolor="#e1e0d9",
                 edgecolor=MUTED, linewidth=0.5)
        for vid, colour, name in ((other, RED, "neighbour"), (moto, INK, "motorcycle")):
            path = tr[tr.vehicle_id == vid].sort_values("t")
            ax.plot(path.x, path.local_x, color=colour, linewidth=1.4, alpha=0.9)
            for t in (-8.0, -6.0, -4.0, -2.0):
                p = path[path.t == t]
                if len(p):
                    ax.plot(p.x, p.local_x, "o", color=colour, markersize=3.5)
                    ax.annotate(f"{t:g} s", (p.x.iloc[0], p.local_x.iloc[0]),
                                textcoords="offset points", xytext=(0, 5), fontsize=7,
                                color=colour, ha="center")
            v = snap[snap.vehicle_id == vid].iloc[0]
            _box(ax, v.x, v.local_x, v.v_length, v.v_width, facecolor=colour,
                 edgecolor=SURFACE, linewidth=0.8, zorder=3)
        both = tr[tr.vehicle_id.isin([moto, other])]
        ax.set_xlim(min(both.x.min(), -10) - 5, max(both.x.max(), 10) + 5)
        ax.set_ylim(tr.local_x.min() - 2, tr.local_x.max() + 2)
        ax.set_aspect("equal")
        ax.set_ylabel("lateral (m)", color=INK_2, fontsize=8)
        ax.set_title(f"{SHORT[case]}: motorcycle {moto} (black) and vehicle {other} (red), "
                     f"snapshot at t = {t_ref:+.1f} s", fontsize=10, color=INK, loc="left")
    axes[-1].set_xlabel("longitudinal (m), relative to the motorcycle at the snapshot; "
                        "traffic flows left to right", color=INK_2, fontsize=9)
    _save(fig, "birdseye")


def fig_geometry(tracks, cases):
    """Why a < 2 m 'collision' scores ~0 box-overlap probability: the detected
    collision at t* to scale."""
    info = cases["collision"]
    snap = tracks[(tracks.case == "collision") & (tracks.t == 0.0)]
    moto = snap[snap.vehicle_id == info["target"]].iloc[0]
    car = snap[snap.vehicle_id == info["neighbour"]].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    _style(ax, zero=False)
    ox, oy = moto.local_y, moto.local_x
    _box(ax, 0, 0, moto.v_length, moto.v_width, facecolor=INK, edgecolor=SURFACE, zorder=3)
    _box(ax, car.local_y - ox, car.local_x - oy, car.v_length, car.v_width,
         facecolor=RED, edgecolor=RED, alpha=0.45, zorder=3)
    half_w = (moto.v_width + car.v_width) / 2
    half_l = (moto.v_length + car.v_length) / 2
    _box(ax, 0, 0, 2 * half_l, 2 * half_w, facecolor="none", edgecolor=BASE_C,
         linewidth=1.6, linestyle="--", zorder=2)
    ax.add_patch(Circle((0, 0), LABEL_DIST, facecolor="none", edgecolor=MUTED,
                        linewidth=1.2, linestyle=":"))
    ax.plot([0, car.local_y - ox], [0, car.local_x - oy], color=INK, linewidth=1.2, zorder=5)
    ax.plot([0, car.local_y - ox], [0, car.local_x - oy], "o", color=INK, markersize=5,
            markeredgecolor=SURFACE, zorder=6)
    gap = float(np.hypot(car.local_y - ox, car.local_x - oy))
    ax.annotate(f"tracked points {gap:.2f} m apart < 2 m\n-> labelled a collision",
                ((car.local_y - ox) / 2, (car.local_x - oy) / 2), xytext=(3.0, 2.4),
                textcoords="data", fontsize=8, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.annotate(f"dashed: where the car's tracked point must be for the\n"
                f"boxes to overlap, |dlat| < (w1 + w2)/2 = {half_w:.2f} m.\n"
                f"Actual dlat = {abs(car.local_x - oy):.2f} m: outside, so only the\n"
                f"spread of the predicted position reaches it -> p stays small",
                (half_l, -half_w), xytext=(3.0, -2.6), textcoords="data",
                fontsize=8, color=BASE_C, arrowprops={"arrowstyle": "-", "color": BASE_C})
    ax.text(0, -2.45, "2 m label radius", color=MUTED, fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 7.5)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("longitudinal (m), motorcycle at 0", color=INK_2, fontsize=9)
    ax.set_ylabel("lateral (m)", color=INK_2, fontsize=9)
    ax.set_title(f"Detected collision at t = 0, to scale: motorcycle "
                 f"{moto.v_length:.1f} x {moto.v_width:.1f} m (black), car "
                 f"{car.v_length:.1f} x {car.v_width:.1f} m (red)",
                 fontsize=10, color=INK, loc="left")
    _save(fig, "geometry")


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    truth = pd.read_csv(OUT / "truth.csv")
    tracks = pd.read_csv(OUT / "tracks.csv")
    cases = json.loads((OUT / "plotted_cases.json").read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = summary(scores, cases)
    fig_overview(scores, truth, cases)
    fig_heatmap(scores, cases)
    fig_coverage(scores, cases)
    fig_gauss_vs_mc(scores, cases)
    fig_birdseye(tracks, scores, truth, cases)
    fig_geometry(tracks, cases)
    for source in ("cv", "pred"):
        fig_metrics(scores, truth, cases, source)
    print(table.groupby(["case", "source"])[["coverage", "peak", "first_ge_0.1", "p_at_0"]]
          .median().round(3).to_string())
    return 0


if __name__ == "__main__":
    assert set(TITLES) == set(CASES)
    sys.exit(main())
