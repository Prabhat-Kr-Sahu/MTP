"""Batched pairwise risk scoring over a test split (doc Part D).

`score_split` walks the split once, predicts in batches, and evaluates every
registered risk measure on each (target, neighbour) pair. It replaces the
one-sample-at-a-time loop in `pipeline.collision_scores`, which needed one
model forward per sample -- 121k forwards per model per variant.

Four neighbour-future sources, because "risk from predicted trajectories" has
no single meaning and the contrast is the point:

    cv        15-nearest candidates, neighbours rolled forward at constant
              velocity. The original definition; kept for continuity.
    cv_scene  the scene's own neighbours, constant velocity. The matched
              control for `pred`.
    pred      the scene's own neighbours, each run through the model as its own
              target. Risk from genuinely predicted motion on both sides.
    oracle    ground-truth futures on both sides. The ceiling any predictor can
              reach under a surrogate label definition -- which is what NGSIM
              forces, since it contains essentially no real collisions.

`cv_scene` and `pred` share a pair set, so their difference isolates the
predictor's contribution over kinematic extrapolation. `cv` and `oracle` share
a pair set, so their difference separates "the predictor is bad" from "the
surrogate label definition is bad".

Why `pred` uses the scene's neighbours and not the 15 nearest: candidates from
`loader.collision_candidates` come from the global frame index, so most have no
column on the scene's vehicle axis and therefore no `target_idx` to build a
sample from. The scene's own neighbours do.
"""
import numpy as np
import torch

from src.config import DEVICE, DT, SEED, TF_STEPS, TH_STEPS
from src.data.loader import collate_fn, create_sample
from src.data.normalization import apply_normalization
from src.evaluation.risk_metrics import EARLINESS_METRICS, METRICS
from src.logger import get_logger

logger = get_logger()

NEIGHBOUR_SOURCES = ("cv", "cv_scene", "pred", "oracle")
ORACLE_SIGMA = 1e-3   # collapses the Gaussian box integral to exact box overlap
PREFIX_SECONDS = (1, 2, 3, 4, 5)


class RiskScores:
    """Per-pair risk scores and the ground truth they are judged against."""

    def __init__(self, scores, gt, gt_distance, gt_time, pairs, prefix_scores,
                 n_samples, n_filled, source, threshold):
        self.scores = scores                 # metric -> (P,) float32
        self.gt = gt                         # (P,) int, 1 = collision
        self.gt_distance = gt_distance       # (P,) m, true minimum separation
        self.gt_time = gt_time               # (P,) s, when that minimum occurs
        self.pairs = pairs                   # (P, 2) [target_id, neighbour_id]
        self.prefix_scores = prefix_scores   # metric -> (P, len(PREFIX_SECONDS))
        self.n_samples = n_samples
        self.n_filled = n_filled             # neighbour future steps gap-filled
        self.source = source
        self.threshold = threshold

    @property
    def n_pairs(self):
        return int(self.gt.size)

    @property
    def n_collisions(self):
        return int(self.gt.sum())

    def summary(self):
        return {
            "source": self.source, "threshold": self.threshold,
            "n_samples": self.n_samples, "n_pairs": self.n_pairs,
            "n_gt_collisions": self.n_collisions,
            "base_rate": float(self.gt.mean()) if self.n_pairs else 0.0,
            "n_future_steps_filled": self.n_filled,
        }


# --------------------------------------------------------------------------- #
#  Neighbour futures
# --------------------------------------------------------------------------- #
def _roll_forward(position, velocity, n_steps, dt=DT):
    """Constant-velocity rollout: (B,M,2) + (B,M,2) -> (B,n_steps,M,2).

    Batched counterpart of pipeline.constant_velocity, whose unsqueeze(0)
    hard-codes a batch of one.
    """
    steps = torch.arange(1, n_steps + 1, dtype=position.dtype, device=position.device)
    return position.unsqueeze(1) + steps.view(1, -1, 1, 1) * dt * velocity.unsqueeze(1)


def _fill_gaps(future):
    """Forward- then backward-fill absent steps of a (T, M, 2) GT future.

    A vehicle that leaves the section holds its last seen position. Not
    nan_to_num: zero is the target's own origin and would manufacture
    collisions. Forward-filling makes a departing vehicle look parked, which
    slightly inflates oracle risk; the alternative -- dropping any pair with a
    gap -- changes the pair set and breaks comparability with `cv`, so the
    count of filled steps is reported instead.
    """
    filled = future.copy()
    missing = ~np.isfinite(filled[..., 0])
    n_filled = int(missing.sum())
    if n_filled:
        index = np.where(~missing, np.arange(filled.shape[0])[:, None], 0)
        np.maximum.accumulate(index, axis=0, out=index)
        filled = np.take_along_axis(filled, index[..., None], axis=0)
        # Backward-fill a leading gap, where no earlier step exists to carry.
        still = ~np.isfinite(filled[..., 0])
        if still.any():
            rev = np.where(~still, np.arange(filled.shape[0])[:, None], filled.shape[0] - 1)
            rev = np.minimum.accumulate(rev[::-1], axis=0)[::-1]
            filled = np.take_along_axis(filled, rev[..., None], axis=0)
    return np.nan_to_num(filled, nan=0.0), n_filled


# --------------------------------------------------------------------------- #
#  Candidate extraction
# --------------------------------------------------------------------------- #
def _vehicle_dims(loader, vehicle_ids):
    """(M, 2) [length, width] per vehicle id, from the loader's vehicle index."""
    out = np.empty((len(vehicle_ids), 2), dtype=np.float32)
    for i, vid in enumerate(vehicle_ids):
        data = loader._vehicle_data[int(vid)]
        out[i] = (data["length"][0], data["width"][0])
    return out


def _candidates(loader, scene, source, n_future):
    """(ids, position, velocity, gt_future, dims, columns) for one scene.

    `columns` indexes the scene's vehicle axis, or None for the global
    15-nearest set, which has no such index -- that is exactly why the
    model-predicted variant uses the scene's own neighbours.
    """
    if source in ("cv", "oracle"):
        ids, position, velocity, future = loader.collision_candidates(scene)
        keep = np.isfinite(future[..., 0]).any(axis=0)
        ids = ids[keep]
        return (ids, position[keep], velocity[keep], future[:, keep],
                _vehicle_dims(loader, ids), None)

    frames = scene["frames"]
    columns = np.arange(1, frames.shape[1])
    origin = frames[TH_STEPS - 1, 0, :2]
    position = frames[TH_STEPS - 1, columns, :2] - origin
    velocity = np.stack(
        [frames[TH_STEPS - 1, columns, 2], np.zeros(len(columns))], axis=-1
    )
    future = frames[TH_STEPS:TH_STEPS + n_future, columns, :2] - origin
    keep = np.isfinite(position[:, 0]) & np.isfinite(future[..., 0]).any(axis=0)
    dims = np.stack(
        [scene["vehicle_lengths"][columns], scene["vehicle_widths"][columns]], axis=-1
    )
    return (scene["vehicle_ids"][columns][keep], position[keep], velocity[keep],
            future[:, keep], dims[keep].astype(np.float32), columns[keep])


def _labels(gt_target, gt_future, threshold, dt=DT):
    """(label, min distance, time of min distance) per candidate."""
    distance = np.linalg.norm(gt_future - gt_target[:, None, :], axis=-1)   # (T, M)
    absent = ~np.isfinite(distance)
    distance = np.where(absent, np.inf, distance)
    index = distance.argmin(axis=0)
    minimum = distance[index, np.arange(distance.shape[1])]
    return (minimum < threshold).astype(np.int8), minimum, index.astype(np.float32) * dt


# --------------------------------------------------------------------------- #
#  Batched prediction
# --------------------------------------------------------------------------- #
def _forward(model, samples, norm_stats, device):
    """One batched forward over raw samples -> (traj_dist, mask, dims) padded."""
    batch = collate_fn([
        (s[0].squeeze(0), s[1].squeeze(0), s[2].squeeze(0), s[5].squeeze(0),
         s[7].squeeze(0)) for s in samples
    ])
    states, gt_positions, _, mask, _, dims = batch
    if norm_stats:
        states = apply_normalization(states, norm_stats)
    with torch.no_grad():
        traj_dist, _, _ = model(states.to(device), mask.to(device), dims.to(device))
    return traj_dist, gt_positions.to(device), dims.to(device)


def _predicted_neighbours(model, loader, scenes, columns, positions, norm_stats,
                          device, batch_size, n_future, max_m):
    """Model-predicted neighbour futures, shifted into each target's frame.

    `positions[b][m]` is the candidate's observed position relative to the
    target, which is exactly the origin offset between the two samples -- so
    the shift is an addition of that position. A model returning zeros
    therefore parks every neighbour at its observed relative position, which is
    what the equivalence test checks.
    """
    requests, index = [], []
    for b, (scene, cols) in enumerate(zip(scenes, columns)):
        for m, column in enumerate(cols):
            requests.append(create_sample(scene, target_idx=int(column),
                                          T_h=TH_STEPS, T_f=n_future))
            index.append((b, m))

    out = torch.full((len(scenes), n_future, max(max_m, 1), 2),
                     float("nan"), device=device)
    for start in range(0, len(requests), batch_size):
        chunk = requests[start:start + batch_size]
        traj_dist, _, _ = _forward(model, chunk, norm_stats, device)
        for offset, (b, m) in enumerate(index[start:start + len(chunk)]):
            shift = torch.as_tensor(positions[b][m], dtype=traj_dist.dtype, device=device)
            out[b, :, m] = traj_dist[offset, :, :2] + shift
    return out


# --------------------------------------------------------------------------- #
#  The driver
# --------------------------------------------------------------------------- #
def score_split(model, loader, test_samples, norm_stats, threshold=2.0,
                source="cv", batch_size=128, sample_stride=1, device=DEVICE,
                metrics=None, prefixes=(), progress_every=200):
    """Score every (target, neighbour) pair of a split. Returns RiskScores.

    `prefixes` names the metrics also evaluated on 1..5 second horizon prefixes
    for the earliness axis; pass EARLINESS_METRICS to enable it.
    """
    if source not in NEIGHBOUR_SOURCES:
        raise ValueError(f"source must be one of {NEIGHBOUR_SOURCES}, got {source!r}")
    metrics = METRICS if metrics is None else metrics
    prefixes = tuple(p for p in prefixes if p in metrics)
    n_future = loader.future_frames
    scene_indices = loader._split_indices["test"]
    chosen = range(0, len(test_samples), sample_stride)
    torch.manual_seed(SEED)

    collected = {name: [] for name in metrics}
    prefix_collected = {name: [] for name in prefixes}
    gt, gt_distance, gt_time, pairs = [], [], [], []
    n_filled = n_scored = 0

    def flush(pending, n_scored, n_filled, batches):
        if not pending:
            return n_scored, n_filled, batches
        batch_out = _score_batch(model, loader, pending, norm_stats, threshold,
                                 source, device, batch_size, n_future,
                                 metrics, prefixes)
        for name, values in batch_out["scores"].items():
            collected[name].append(values)
        for name, values in batch_out["prefix"].items():
            prefix_collected[name].append(values)
        gt.append(batch_out["gt"])
        gt_distance.append(batch_out["gt_distance"])
        gt_time.append(batch_out["gt_time"])
        pairs.append(batch_out["pairs"])
        batches += 1
        n_scored += len(pending)
        if progress_every and batches % progress_every == 0:
            logger.info("risk scoring [%s]: %d/%d samples", source, n_scored, len(chosen))
        return n_scored, n_filled + batch_out["n_filled"], batches

    pending, batches = [], 0
    for sample_index in chosen:
        scene = loader.scenes[scene_indices[sample_index]]
        candidates = _candidates(loader, scene, source, n_future)
        if not len(candidates[0]):
            continue
        pending.append((test_samples[sample_index], scene, candidates))
        if len(pending) >= batch_size:
            n_scored, n_filled, batches = flush(pending, n_scored, n_filled, batches)
            pending = []
    n_scored, n_filled, batches = flush(pending, n_scored, n_filled, batches)

    if not gt:
        raise ValueError("no scorable pairs in the split")

    return RiskScores(
        scores={n: np.concatenate(v) for n, v in collected.items()},
        gt=np.concatenate(gt),
        gt_distance=np.concatenate(gt_distance),
        gt_time=np.concatenate(gt_time),
        pairs=np.concatenate(pairs),
        prefix_scores={n: np.concatenate(v) for n, v in prefix_collected.items()},
        n_samples=n_scored, n_filled=n_filled, source=source, threshold=threshold,
    )


def _score_batch(model, loader, pending, norm_stats, threshold, source, device,
                 batch_size, n_future, metrics, prefixes):
    """Score one batch of samples. Padding never reaches the output: every
    result is flattened through the validity mask."""
    samples = [p[0] for p in pending]
    scenes = [p[1] for p in pending]
    candidates = [p[2] for p in pending]
    batch = len(pending)
    max_m = max(len(c[0]) for c in candidates)

    traj_dist, gt_positions, dims = _forward(model, samples, norm_stats, device)
    if source == "oracle":
        target = gt_positions
        sigma = torch.full_like(target, ORACLE_SIGMA)
        rho = torch.zeros(batch, n_future, device=device)
    else:
        target = traj_dist[:, :, :2]
        sigma = traj_dist[:, :, 2:4]
        rho = traj_dist[:, :, 4]

    valid = torch.zeros(batch, max_m, dtype=torch.bool, device=device)
    position = torch.zeros(batch, max_m, 2, device=device)
    velocity = torch.zeros(batch, max_m, 2, device=device)
    other_dims = torch.ones(batch, max_m, 2, device=device)
    gt_future = np.full((batch, n_future, max_m, 2), np.nan, dtype=np.float32)
    labels, distances, times, pair_ids, filled = [], [], [], [], 0

    for b, (ids, pos, vel, future, cand_dims, _) in enumerate(candidates):
        m = len(ids)
        valid[b, :m] = True
        position[b, :m] = torch.as_tensor(pos, dtype=torch.float32, device=device)
        velocity[b, :m] = torch.as_tensor(vel, dtype=torch.float32, device=device)
        other_dims[b, :m] = torch.as_tensor(cand_dims, dtype=torch.float32, device=device)
        gt_future[b, :, :m] = future
        label, distance, time = _labels(
            samples[b][1][0].numpy(), future, threshold
        )
        labels.append(label)
        distances.append(distance)
        times.append(time)
        pair_ids.append(np.stack(
            [np.full(m, samples[b][4][0], dtype=np.int64), ids.astype(np.int64)], axis=-1
        ))

    if source in ("cv", "cv_scene"):
        neighbors = _roll_forward(position, velocity, n_future)
    elif source == "oracle":
        # (B,T,M,2) -> (T, B*M, 2) to gap-fill along time, then back.
        along_time = gt_future.transpose(1, 0, 2, 3).reshape(n_future, batch * max_m, 2)
        gap_filled, filled = _fill_gaps(along_time)
        neighbors = torch.as_tensor(
            gap_filled.reshape(n_future, batch, max_m, 2).transpose(1, 0, 2, 3).copy(),
            dtype=torch.float32, device=device,
        )
    else:
        neighbors = _predicted_neighbours(
            model, loader, scenes, [c[5] for c in candidates],
            [c[1] for c in candidates], norm_stats, device, batch_size,
            n_future, max_m,
        )

    # Padded columns sit far away so they can never register as an interaction;
    # they are dropped by the mask anyway (see the padding-invariance test).
    neighbors = torch.where(
        valid.view(batch, 1, max_m, 1), neighbors, torch.full_like(neighbors, 1e4)
    )
    ctx = {"sigma": sigma, "rho": rho, "ego_dims": dims[:, 0],
           "other_dims": other_dims}
    flat = valid.reshape(-1).cpu().numpy()

    scores, prefix = {}, {}
    for name, (fn, _, _) in metrics.items():
        out = fn(target, neighbors, dt=DT, threshold=threshold, **ctx)
        scores[name] = out.reshape(-1).float().cpu().numpy()[flat].astype(np.float32)
    for name in prefixes:
        fn = metrics[name][0]
        columns = []
        for seconds in PREFIX_SECONDS:
            steps = min(int(round(seconds / DT)), n_future)
            out = fn(target[:, :steps], neighbors[:, :steps],
                     dt=DT, threshold=threshold,
                     **{**ctx, "sigma": sigma[:, :steps], "rho": rho[:, :steps]})
            columns.append(out.reshape(-1).float().cpu().numpy()[flat])
        prefix[name] = np.stack(columns, axis=-1).astype(np.float32)

    return {
        "scores": scores, "prefix": prefix,
        "gt": np.concatenate(labels), "gt_distance": np.concatenate(distances),
        "gt_time": np.concatenate(times), "pairs": np.concatenate(pair_ids),
        "n_filled": filled,
    }


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def _safe(fn, *args, **kwargs):
    """AUCs raise on a degenerate label set; report nan rather than abort."""
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return float("nan")


def metric_report(scores, gt, name, higher_is_riskier, default_threshold):
    """One row: operating point under the shared ROC rule, plus both AUCs."""
    from src.evaluation import collision as C
    from src.evaluation.risk_metrics import is_collision

    chosen = _safe(C.select_threshold, scores, gt, higher_is_riskier)
    threshold = default_threshold if np.isnan(chosen) else chosen
    predicted = is_collision(name, scores, threshold)
    row = {
        "metric": name,
        "higher_is_riskier": bool(higher_is_riskier),
        "default_threshold": float(default_threshold),
        "threshold": float(threshold),
        "threshold_rule": "default" if np.isnan(chosen) else "roc_nearest_01",
        "non_finite_fraction": float(np.mean(~np.isfinite(scores))),
        **C.collision_metrics_from_labels(predicted.astype(int), gt),
        "recall_at_precision_50": _safe(
            C.recall_at_precision, scores, gt, higher_is_riskier, 0.5
        ),
        "pr_auc": _safe(C.compute_pr_auc, scores, gt, higher_is_riskier),
        "roc_auc": _safe(C.compute_roc_auc, scores, gt, higher_is_riskier),
    }
    return row


LABEL_THRESHOLDS = (1.0, 2.0, 3.0, 5.0)


def build_report(result, oracle=None, metrics=None, n_bins=10,
                 label_thresholds=LABEL_THRESHOLDS):
    """Assemble the JSON-ready report for one RiskScores: rows, the separation
    sanity check, calibration, earliness, stratification and oracle regression.

    `label_thresholds` re-derives the collision labels from the stored
    per-pair true minimum separation and reports both AUCs at each. The label
    rule is the one free parameter that decides whether this comparison says
    anything -- NGSIM has essentially no real collisions, so the positives are
    whatever a distance threshold declares them to be, and plan.md section 50
    says that threshold has to be settled experimentally. Sweeping it in the
    report means the scoring run does not have to be repeated to answer the
    question, and a base rate too small to support an AUC becomes visible
    rather than silently producing nan.
    """
    from src.evaluation import collision as C
    from src.evaluation.risk_metrics import METRICS, PROBABILITY_METRICS

    metrics = METRICS if metrics is None else metrics
    gt = result.gt
    positive, negative = gt == 1, gt == 0

    rows, sanity, calibration, earliness_out, regression = [], {}, {}, {}, {}
    for name, (_, higher_is_riskier, default_threshold) in metrics.items():
        scores = result.scores[name]
        rows.append(metric_report(scores, gt, name, higher_is_riskier,
                                  default_threshold))

        # Doc section C4: mean risk on collision vs non-collision pairs. Run
        # this first -- weak separation means something is wrong upstream and
        # no amount of AUC reporting will rescue the result.
        # float64: several measures reach 1e12+, which overflows a float32 sum.
        finite = np.isfinite(scores)
        wide = scores.astype(np.float64)
        sanity[name] = {
            "mean_collision": float(wide[positive & finite].mean())
            if (positive & finite).any() else None,
            "mean_non_collision": float(wide[negative & finite].mean())
            if (negative & finite).any() else None,
            "finite_fraction": float(finite.mean()),
        }

        if name in PROBABILITY_METRICS and finite.all():
            calibration[name] = {
                **C.murphy_decomposition(scores, gt, n_bins=n_bins),
                "log_loss": C.log_loss(scores, gt),
                "reliability_bins": C.reliability_bins(scores, gt, n_bins=n_bins),
            }

        if name in result.prefix_scores and positive.any():
            threshold = rows[-1]["threshold"]
            earliness_out[name] = {
                **C.earliness(result.prefix_scores[name], gt, result.gt_time,
                              threshold, higher_is_riskier, PREFIX_SECONDS),
                "lead_time_at_80_recall": _safe(
                    C.lead_time_at_recall, result.prefix_scores[name], gt,
                    result.gt_time, higher_is_riskier, 0.8, PREFIX_SECONDS,
                ),
            }

        if oracle is not None and name in oracle.scores:
            regression[name] = C.risk_regression(scores, oracle.scores[name])

    stratified = {}
    for bucket, mask in C.stratify(result.gt_time, gt).items():
        if not mask.any():
            continue
        # This bucket's positives against the full non-collision background:
        # "can the measure separate a collision this imminent from safe traffic".
        keep = mask | negative
        labels = mask[keep].astype(int)
        bucket_rows = [
            metric_report(result.scores[name][keep], labels, name, higher, default)
            for name, (_, higher, default) in metrics.items()
        ]
        stratified[bucket] = {"n_pairs": int(mask.sum()), "rows": bucket_rows}

    by_label_threshold = {}
    for candidate in label_thresholds:
        labels = (result.gt_distance < candidate).astype(int)
        by_label_threshold[f"{candidate:g}"] = {
            "n_gt_collisions": int(labels.sum()),
            "base_rate": float(labels.mean()) if labels.size else 0.0,
            "metrics": {
                name: {
                    "pr_auc": _safe(C.compute_pr_auc, result.scores[name], labels, higher),
                    "roc_auc": _safe(C.compute_roc_auc, result.scores[name], labels, higher),
                }
                for name, (_, higher, _) in metrics.items()
            },
        }

    return {
        **result.summary(),
        "rows": rows, "sanity": sanity, "calibration": calibration,
        "earliness": earliness_out, "stratified": stratified,
        "regression_vs_oracle": regression,
        "by_label_threshold": by_label_threshold,
        "gt_distance_percentiles": {
            f"p{q:g}": float(np.percentile(result.gt_distance, q))
            for q in (0.1, 0.5, 1, 2, 5, 10, 25, 50)
        },
    }
