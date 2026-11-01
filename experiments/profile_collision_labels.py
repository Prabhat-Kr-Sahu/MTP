"""Characterise the ground-truth collision labels on the full NGSIM test split.

Model-free: true futures only, through the same _candidates / _labels code the
scoring run uses (cv candidate set). Produces the tables in
RISK_PREDICTION_RESULTS.md section 2.1b. About 6 min on the GPU box, CPU only.

    python3 experiments/profile_collision_labels.py > experiments/runs/collision_profile.out
"""
import sys
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import DT, NGSIM_LOCATIONS, TF_STEPS, TH_STEPS  # noqa: E402
from src.data.loader import _LOCATION_NAMES, NGSIMDataLoader  # noqa: E402
from src.evaluation.risk_scoring import _candidates, _labels  # noqa: E402


def collect(loader, test):
    """One row per (target, candidate) pair, as a dict of flat arrays."""
    keys = ("target", "other", "dmin", "tmin", "d0", "dx", "dy",
            "vt", "vo", "lt", "wt", "lo", "wo")
    rec = {k: [] for k in keys}
    for i, sample in enumerate(test):
        scene = loader.scenes[loader._split_indices["test"][i]]
        ids, pos, vel, fut, dims, _ = _candidates(loader, scene, "cv", TF_STEPS)
        if not len(ids):
            continue
        tgt = sample[1][0].numpy()
        _, dmin, tmin = _labels(tgt, fut, 2.0)
        step = np.rint(tmin / DT).astype(int)
        rel = fut[step, np.arange(len(ids))] - tgt[step]
        n = len(ids)
        for key, value in (
            ("target", np.full(n, scene["vehicle_ids"][0])), ("other", ids),
            ("dmin", dmin), ("tmin", tmin), ("d0", np.linalg.norm(pos, axis=-1)),
            ("dx", np.abs(rel[:, 0])), ("dy", np.abs(rel[:, 1])),
            ("vt", np.full(n, scene["frames"][TH_STEPS - 1, 0, 2])), ("vo", vel[:, 0]),
            ("lt", np.full(n, scene["vehicle_lengths"][0])),
            ("wt", np.full(n, scene["vehicle_widths"][0])),
            ("lo", dims[:, 0]), ("wo", dims[:, 1]),
        ):
            rec[key].append(value)
    return {k: np.concatenate(v) for k, v in rec.items()}


def pct(values, qs):
    return " / ".join(f"{v:.2f}" for v in np.percentile(values, qs))


def report(r, threshold):
    p = r["dmin"] < threshold
    sub = {k: v[p] for k, v in r.items()}
    n_pos = int(p.sum())
    # One event = one unordered vehicle pair; windows and target/other roles repeat it.
    pair = np.minimum(sub["target"], sub["other"]) * 10**8 + np.maximum(sub["target"], sub["other"])
    events, inverse, per_event = np.unique(pair, return_inverse=True, return_counts=True)
    overlap = ((sub["dx"] < (sub["lt"] + sub["lo"]) / 2)
               & (sub["dy"] < (sub["wt"] + sub["wo"]) / 2))
    event_overlap = np.bincount(inverse, weights=overlap) > 0
    event_location = [_LOCATION_NAMES[int(e // 10**8) // 1_000_000] for e in events]

    print(f"\n===== d_min < {threshold} m =====")
    print(f"positive pairs: {n_pos:,} ({100 * n_pos / p.size:.4f} %)")
    print(f"distinct vehicle-pair events: {len(events):,}   "
          f"vehicles involved: {len(np.unique(np.r_[sub['target'], sub['other']])):,}")
    print(f"windows per event median/p90/max: {pct(per_event, [50, 90])} / {per_event.max()}")
    print(f"top-10 events hold {int(np.sort(per_event)[-10:].sum()):,} pairs")
    print("events by location:", dict(zip(*np.unique(event_location, return_counts=True))))
    print(f"boxes overlap at closest approach: {int(overlap.sum()):,} pairs "
          f"({100 * overlap.mean():.1f} %), {int(event_overlap.sum())} events "
          f"({100 * event_overlap.mean():.0f} %)")
    print(f"already < {threshold} m at last observed frame: {int((sub['d0'] < threshold).sum()):,} "
          f"({100 * (sub['d0'] < threshold).mean():.1f} %)")
    print(f"distance at last observed frame p10/p50/p90: {pct(sub['d0'], [10, 50, 90])}")
    edges = (-1, 0.05, 1, 2, 3, 5.01)
    print("closest approach in (0 | <=1 | 1-2 | 2-3 | 3-5 s):",
          [int(((sub["tmin"] > lo) & (sub["tmin"] <= hi)).sum()) for lo, hi in zip(edges, edges[1:])])
    print(f"|longitudinal| at closest p10/p50/p90: {pct(sub['dx'], [10, 50, 90])}")
    print(f"|lateral|      at closest p10/p50/p90: {pct(sub['dy'], [10, 50, 90])}")
    print(f"side-by-side (|lon| < |lat|): {int((sub['dx'] < sub['dy']).sum()):,}")
    print(f"target speed median: {np.median(sub['vt']):.2f} m/s   "
          f"|relative speed| p50/p90: {pct(np.abs(sub['vt'] - sub['vo']), [50, 90])}")
    print(f"heavy (>8 m) vehicle involved: {int(((sub['lt'] > 8) | (sub['lo'] > 8)).sum()):,}")


def main():
    loader = NGSIMDataLoader(locations=NGSIM_LOCATIONS)
    _, _, test = loader.get_splits()
    r = collect(loader, test)
    location = np.array([_LOCATION_NAMES[int(t) // 1_000_000] for t in r["target"]])
    print(f"samples: {len(test):,}   pairs: {r['dmin'].size:,}   "
          f"distinct targets: {len(np.unique(r['target'])):,}")
    print("pairs by location:", dict(zip(*np.unique(location, return_counts=True))))
    for threshold in (1.0, 2.0, 3.0):
        report(r, threshold)
    negative = r["dmin"] >= 2.0
    print(f"\nnegatives: distance at last observed frame p50 {np.median(r['d0'][negative]):.2f} m; "
          f"all-pairs target speed median {np.median(r['vt']):.2f} m/s")


if __name__ == "__main__":
    main()
