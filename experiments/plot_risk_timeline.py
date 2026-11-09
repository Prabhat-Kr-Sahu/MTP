#!/usr/bin/env python3
"""Collision-risk probability over time for one vehicle, every model, five cases.

The committed risk_v2 scores give one number per (target, neighbour) pair per
1 s window, and only their ROC placements were saved. A curve over time needs
the models re-run, so this re-predicts every 0.1 s (loader stride 1) over
[-8 s, +1 s] around each case's true closest approach, with the `pred` source:
the model predicts both the target and the neighbour, as in the headline study.

    --select   pick the cases from the committed risk_v2 placements (local, seconds)
    --score    rebuild the windows and score all 15 models     (lab PC: NGSIM + checkpoints)
    --plot     safe / false-alarm selection and the figures     (local, from the CSVs)

Cases:
    collision   < 2 m event the 15 models rank highest (mean p_coll_gauss placement)
    missed      < 2 m event they rank lowest
    near_miss   2-3 m event, median placement
    safe        a neighbour of the collision case's target that stays > 5 m away
                (but within 30 m, so it interacts), lowest peak median risk
    false_alarm same pool, highest peak median risk

Collision and missed share one vehicle when the data allows it; that vehicle is
"the particular vehicle", and the safe / false-alarm neighbours come from its
own timeline.

The y-axis is p_coll_gauss / p_coll_mc in [0, 1]: uncalibrated collision
probabilities (every Brier skill score in the study is negative), not
standardised values and not frequencies.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import DEVICE, DT, NGSIM_LOCATIONS, SEED, TF_STEPS, TH_STEPS  # noqa: E402
from src.data.loader import NGSIMDataLoader, create_sample  # noqa: E402
from src.evaluation.risk_metrics import METRICS  # noqa: E402
from src.evaluation.risk_scoring import _candidates, _score_batch  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
RISK_V2 = EXPERIMENTS / "results" / "risk_v2"
OUT = EXPERIMENTS / "results" / "risk_timeline"
FIGURES = EXPERIMENTS / "figures"

BASELINES = ("inatran", "cognitive", "dacr_amtp", "miat", "mha_lstm", "stdan",
             "x_track", "traj_mamba")
# traj_mamba's tuning was stopped at 20/30 trials and never retrained.
MODELS = BASELINES + tuple(f"{b}_tuned" for b in BASELINES[:7])
PLOTTED = ("p_coll_gauss", "p_coll_mc")
BEFORE, AFTER = 80, 10          # frames around the closest approach: -8 s .. +1 s
SAFE_MIN_DIST, INTERACT_DIST, ALARM_P = 5.0, 30.0, 0.1
LABEL_DIST = 2.0


# --------------------------------------------------------------------------- #
#  --select
# --------------------------------------------------------------------------- #
def _ids(key):
    """Event key (risk_scoring.event_keys) -> the two vehicle ids."""
    return int(key // 10**8), int(key % 10**8)


def event_placements(tag, metric="p_coll_gauss"):
    """(event keys, (models, events) mean placement) from the committed npz."""
    keys, rows = None, []
    for name in MODELS:
        z = np.load(RISK_V2 / f"{name}.pred.npz")
        unique, inverse = np.unique(z[f"events_{tag}"], return_inverse=True)
        if keys is not None and not np.array_equal(unique, keys):
            raise ValueError(f"{name}: event set differs from {MODELS[0]}")
        keys = unique
        placement = z[f"placement_{tag}__{metric}"]
        rows.append(np.bincount(inverse, placement) / np.bincount(inverse))
    return keys, np.array(rows)


def select_cases():
    k2, p2 = event_placements("2")
    k3, p3 = event_placements("3")
    m2, m3 = p2.mean(0), p3.mean(0)

    missed = int(m2.argmin())
    shares = np.array([bool(set(_ids(k)) & set(_ids(k2[missed]))) for k in k2])
    shares[missed] = False
    pool = shares if shares.any() else np.arange(len(k2)) != missed
    collision = int(np.where(pool, m2, -np.inf).argmax())
    common = set(_ids(k2[collision])) & set(_ids(k2[missed]))
    vehicle = common.pop() if common else _ids(k2[collision])[0]

    near = ~np.isin(k3, k2)
    with_vehicle = near & np.array([vehicle in _ids(k) for k in k3])
    index = np.flatnonzero(with_vehicle if with_vehicle.any() else near)
    near_miss = int(index[np.argsort(m3[index])[len(index) // 2]])

    def case(key, placements):
        a, b = _ids(key)
        target = vehicle if vehicle in (a, b) else a
        return {"target": target, "neighbour": b if target == a else a, "event": int(key),
                "mean_placement": float(placements.mean()),
                "placement_by_model": dict(zip(MODELS, map(float, placements)))}

    cases = {
        "vehicle": vehicle,
        "metric": "p_coll_gauss placement, source pred, mean over 15 models",
        "collision": case(k2[collision], p2[:, collision]),
        "missed": case(k2[missed], p2[:, missed]),
        "near_miss": case(k3[near_miss], p3[:, near_miss]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


# --------------------------------------------------------------------------- #
#  --score
# --------------------------------------------------------------------------- #
def closest_frame(df, a, b):
    """(frame, distance) of the pair's minimum true separation over shared frames."""
    cols = ["frame_id", "local_y", "local_x"]
    both = df.loc[df.vehicle_id == a, cols].merge(
        df.loc[df.vehicle_id == b, cols], on="frame_id", suffixes=("_a", "_b"))
    if both.empty:
        raise ValueError(f"vehicles {a} and {b} never share a frame")
    dist = np.hypot(both.local_y_a - both.local_y_b, both.local_x_a - both.local_x_b)
    i = int(np.argmin(dist.to_numpy()))
    return int(both.frame_id.iloc[i]), float(dist.iloc[i])


def window_scenes(df, target, t_star, loader_kwargs=None):
    """(loader, scenes): every stride-1 window of `target` whose "now" frame
    lies in [t_star - BEFORE, t_star + AFTER].

    Only the frames those windows touch are kept, but every vehicle in them is,
    because the scene's neighbour selection ranks all of them.
    """
    # ponytail: builds scenes for every vehicle in the slice and keeps the
    # target's -- ~20 s of traffic, so seconds; filter eligibility if it grows.
    lo, hi = t_star - BEFORE - TH_STEPS, t_star + AFTER + TF_STEPS
    loader = NGSIMDataLoader(stride=1, **(loader_kwargs or {}))
    loader.df = df[(df.frame_id >= lo) & (df.frame_id <= hi)].reset_index(drop=True)
    now = lambda s: int(s["frame_id_start"]) + TH_STEPS - 1  # noqa: E731
    scenes = [s for s in loader.build_scenes()
              if s["vehicle_ids"][0] == target and -BEFORE <= now(s) - t_star <= AFTER]
    return loader, scenes


def score_windows(model, norm_stats, loader, scenes, device, batch_size=64):
    """One row per (window, scene neighbour): pred-source risk for the window."""
    metrics = {name: METRICS[name] for name in PLOTTED}
    pending = []
    for scene in scenes:
        candidates = _candidates(loader, scene, "pred", loader.future_frames)
        if len(candidates[0]):
            sample = create_sample(scene, target_idx=0, T_h=TH_STEPS, T_f=loader.future_frames)
            pending.append((sample, scene, candidates))
    frames = []
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        out = _score_batch(model, loader, chunk, norm_stats, LABEL_DIST, "pred", device,
                           batch_size, loader.future_frames, metrics, ())
        frames.append(pd.DataFrame({
            "now_frame": np.concatenate([
                np.full(len(c[2][0]), int(c[1]["frame_id_start"]) + TH_STEPS - 1) for c in chunk]),
            "neighbour": out["pairs"][:, 1],
            "dist_now": np.concatenate([np.linalg.norm(c[2][1], axis=-1) for c in chunk]),
            "gt_min_dist_5s": out["gt_distance"],
            **{name: out["scores"][name] for name in PLOTTED},
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def true_distances(loader, target, neighbours, t_star):
    """True separation of target and each neighbour at every frame of the range."""
    data = loader._vehicle_data
    frames = np.arange(t_star - BEFORE, t_star + AFTER + 1)
    rows = []
    for nb in neighbours:
        a, b = data[int(target)], data[int(nb)]
        ia, ib = np.searchsorted(a["frames"], frames), np.searchsorted(b["frames"], frames)
        ok = (ia < len(a["frames"])) & (ib < len(b["frames"]))
        ok[ok] &= (a["frames"][ia[ok]] == frames[ok]) & (b["frames"][ib[ok]] == frames[ok])
        dist = np.hypot(a["local_y"][ia[ok]] - b["local_y"][ib[ok]],
                        a["local_x"][ia[ok]] - b["local_x"][ib[ok]])
        rows.append(pd.DataFrame({"neighbour": int(nb), "now_frame": frames[ok], "dist": dist}))
    return pd.concat(rows, ignore_index=True)


def score(args):
    from experiments.score_risk import load_notebook_model

    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    torch.manual_seed(SEED)
    full = NGSIMDataLoader(locations=NGSIM_LOCATIONS).fetch()
    stats = None
    if args.random_weights:
        from src.config import STATE_DIM
        stats = {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)}

    args.out.mkdir(parents=True, exist_ok=True)
    scores, truth, windows = [], [], {}
    for case in ("collision", "missed", "near_miss"):
        target, neighbour = cases[case]["target"], cases[case]["neighbour"]
        t_star, min_dist = closest_frame(full, target, neighbour)
        loader, scenes = window_scenes(full, target, t_star)
        row = full.loc[full.vehicle_id == target].iloc[0]
        windows[case] = {"target": target, "neighbour": neighbour, "t_star_frame": t_star,
                         "min_dist": min_dist, "n_windows": len(scenes),
                         "location": str(row["location"]),
                         "orig_vehicle_id": int(row.get("orig_vehicle_id", target))}
        print(f"{case}: {target} vs {neighbour}, t*={t_star}, min {min_dist:.2f} m, "
              f"{len(scenes)} windows", flush=True)
        seen = set()
        for name in args.models:
            model, namespace = load_notebook_model(
                name, device=args.device, load_weights=not args.random_weights, stats=stats)
            frame = score_windows(model, namespace["norm_stats"], loader, scenes, args.device)
            frame.insert(0, "model", name)
            frame.insert(0, "case", case)
            seen |= set(frame.neighbour.tolist())
            scores.append(frame)
            print(f"  {name}: {len(frame)} rows", flush=True)
        dist = true_distances(loader, target, sorted(seen | {neighbour}), t_star)
        dist.insert(0, "case", case)
        truth.append(dist)

    t0 = {c: w["t_star_frame"] for c, w in windows.items()}
    for frames in (scores, truth):
        for f in frames:
            f.insert(2, "t", ((f.now_frame - f.case.map(t0)) * DT).round(1))
    pd.concat(scores).to_csv(args.out / "scores.csv", index=False)
    pd.concat(truth).to_csv(args.out / "truth.csv", index=False)
    (args.out / "windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  --plot
# --------------------------------------------------------------------------- #
def derive_safe_and_alarm(scores, truth, windows):
    """Safe and false-alarm neighbours from the scored timelines.

    Pool: neighbours of the case targets other than the case's own neighbour,
    whose true distance stays > SAFE_MIN_DIST yet comes within INTERACT_DIST,
    and that are in the target's scene for at least half the windows. The
    collision case's window (the particular vehicle) is preferred for both.
    """
    n_models = scores.model.nunique()
    peak = (scores.groupby(["case", "neighbour", "t"])["p_coll_gauss"].median()
            .groupby(["case", "neighbour"]).max())
    coverage = (scores.groupby(["case", "neighbour"]).size()
                / (n_models * scores.groupby("case").t.nunique()))
    dist = truth.groupby(["case", "neighbour"]).dist.min()
    pool = pd.DataFrame({"peak": peak, "coverage": coverage, "min_dist": dist}).dropna()
    own = [(c, w["neighbour"]) for c, w in windows.items()]
    pool = pool[(pool.min_dist > SAFE_MIN_DIST) & (pool.min_dist < INTERACT_DIST)
                & (pool.coverage >= 0.5) & ~pool.index.isin(own)]
    if pool.empty:
        raise ValueError("no neighbour qualifies as safe / false alarm")
    home = pool[pool.index.get_level_values("case") == "collision"]
    safe = (home if len(home) else pool).peak.idxmin()
    alarm_pool = home if len(home) and home.peak.max() >= ALARM_P else pool
    alarm = alarm_pool.peak.idxmax()
    out = {}
    for name, (case, nb) in (("safe", safe), ("false_alarm", alarm)):
        out[name] = {**windows[case], "neighbour": int(nb), "from_window": case,
                     "min_dist": float(pool.loc[(case, nb), "min_dist"]),
                     "peak_median_p_coll_gauss": float(pool.loc[(case, nb), "peak"])}
    return out


TITLES = {
    "collision": "Collision (detected)",
    "missed": "Collision the models miss",
    "near_miss": "Near miss (2-3 m)",
    "safe": "No collision (safe neighbour)",
    "false_alarm": "False alarm (safe, but flagged risky)",
}
METRIC_LABEL = {
    "p_coll_gauss": "P(overlap at the worst step, next 5 s)  [p_coll_gauss]",
    "p_coll_mc": "P(overlap at any step, next 5 s)  [p_coll_mc, Monte Carlo]",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BASE_C, TUNED_C = "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")


def plot_case(case, info, scores, truth, metric, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = scores[(scores.case == info.get("from_window", case))
                  & (scores.neighbour == info["neighbour"])]
    windows = scores[scores.case == info.get("from_window", case)].t.nunique()
    top = max(float(rows[metric].max()) * 1.08 if len(rows) else 0.0, 0.05)
    x_range = (-BEFORE * DT, AFTER * DT)

    fig = plt.figure(figsize=(13, 8.2), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.85], hspace=0.45, wspace=0.12)
    for i, base in enumerate(BASELINES):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        _style(ax)
        for name, colour, style in ((base, BASE_C, "-"), (f"{base}_tuned", TUNED_C, "--")):
            series = rows[rows.model == name].set_index("t")[metric]
            if not len(series):
                continue
            # Reindex on the full grid so absent windows break the line, not bridge it.
            full = series.reindex(np.round(np.arange(-BEFORE, AFTER + 1) * DT, 1))
            ax.plot(full.index, full.values, color=colour, linestyle=style, linewidth=2)
        cover = rows[rows.model == base].t.nunique() / max(windows, 1)
        ax.set_title(f"{base}   (in scene {cover:.0%})", fontsize=9, color=INK, loc="left")
        ax.set_xlim(*x_range)
        ax.set_ylim(0, top)
        if i % 4:
            ax.tick_params(labelleft=False)
        if i // 4 == 0:
            ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(grid[2, :])
    _style(ax)
    dist = truth[(truth.case == info.get("from_window", case))
                 & (truth.neighbour == info["neighbour"])].sort_values("t")
    ax.plot(dist.t, dist.dist, color=INK, linewidth=2)
    ax.axhline(LABEL_DIST, color=INK_2, linewidth=1, linestyle="--")
    ax.text(x_range[0] + 0.1, LABEL_DIST, " 2 m = collision label", color=INK_2,
            fontsize=8, va="bottom")
    ax.set_xlim(*x_range)
    ax.set_ylim(0, None)
    ax.set_ylabel("true distance (m)", color=INK_2, fontsize=9)
    zero = ("closest approach" if "from_window" not in info
            else "the same vehicle's collision moment")
    ax.set_xlabel(f"time (s); 0 = {zero}", color=INK_2, fontsize=9)

    handles = [plt.Line2D([], [], color=BASE_C, linewidth=2, label="baseline"),
               plt.Line2D([], [], color=TUNED_C, linewidth=2, linestyle="--",
                          label="tuned (traj_mamba: none)")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               bbox_to_anchor=(0.98, 0.985), ncol=2, labelcolor=INK_2)
    fig.suptitle(
        f"{TITLES[case]}: vehicle {info['target']} vs {info['neighbour']}"
        f"   ({info['location']}, true min distance {info['min_dist']:.2f} m)",
        x=0.06, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.06, 0.935, f"y: {METRIC_LABEL[metric]}. Uncalibrated probability, "
             "source pred; a gap = neighbour not in the target's scene.",
             fontsize=9, color=INK_2)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot(args):
    scores = pd.read_csv(args.out / "scores.csv")
    truth = pd.read_csv(args.out / "truth.csv")
    windows = json.loads((args.out / "windows.json").read_text(encoding="utf-8"))
    cases = {**windows, **derive_safe_and_alarm(scores, truth, windows)}
    (args.out / "plotted_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for case, info in cases.items():
        for metric in PLOTTED:
            path = FIGURES / f"risk_timeline_{case}_{metric}.png"
            plot_case(case, info, scores, truth, metric, path)
            print(path.relative_to(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--plot", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--random-weights", action="store_true",
                        help="smoke test only: no checkpoints, MEANINGLESS scores")
    args = parser.parse_args(argv)
    args.device = DEVICE
    if args.select:
        print(json.dumps(select_cases(), indent=2))
    elif args.score:
        score(args)
    else:
        plot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
