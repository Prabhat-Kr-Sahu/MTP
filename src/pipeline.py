"""STRAP pipeline entry point: prepare | train | eval | collisions."""
import argparse
import os
import sys

# Allow `python src/pipeline.py` (script dir = src/) and `python -m src.pipeline`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import (
    BETA_LOSS, CHECKPOINT_DIR, DEBUG_MAX_ROWS, DEVICE, DT, INTENTIONS_FILE,
    LEARNING_RATE, LOG_DIR, NGSIM_LOCATIONS, NORMALIZATION_FILE, NUM_EPOCHS,
)
from src.data.loader import NGSIMDataLoader, create_dataloader, location_of
from src.data.normalization import apply_normalization, load_stats, normalize_samples
from src.evaluation.collision import (
    collision_metrics_from_labels, compute_pr_auc, compute_roc_auc,
)
from src.evaluation.evaluate import evaluate_trajectory, print_metrics
from src.evaluation.risk_metrics import METRICS, is_collision
from src.intentions import load_codebook
from src.logger import get_logger
from src.models.strap import STRAP
from src.train import train

logger = get_logger()


# ----------------------------------------------------------------- helpers

def make_loader(args):
    """Loader for --locations, or us-101 (debug) / all NGSIM_LOCATIONS."""
    if args.locations:
        locations = [loc.strip() for loc in args.locations.split(",") if loc.strip()]
    else:
        locations = ["us-101"] if args.debug else NGSIM_LOCATIONS
    return NGSIMDataLoader(
        locations=locations, max_rows=DEBUG_MAX_ROWS if args.debug else None
    )


def build_model():
    model = STRAP().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"STRAP on {DEVICE}: {n_params:,} parameters")
    return model


def load_trained(model):
    """Load best checkpoint; return normalization stats.

    The intention codebook is a buffer saved in the checkpoint. The .pkl is
    only checked against it, never loaded over it (SOLUTIONS problem 3).
    """
    checkpoint = f"{CHECKPOINT_DIR}/best_model.pt"
    if not os.path.exists(checkpoint):
        raise SystemExit(f"No checkpoint at {checkpoint}. Run --mode train first.")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    intentions = f"{CHECKPOINT_DIR}/{INTENTIONS_FILE}"
    if os.path.exists(intentions):
        saved = torch.from_numpy(load_codebook(intentions)).to(DEVICE)
        drift = (model.risk_decoder.intention_means - saved).norm(dim=-1).max().item()
        if drift > 1e-4:
            logger.warning(
                f"{INTENTIONS_FILE} differs from the checkpoint's codebook "
                f"(max {drift:.3f}); using the checkpoint's"
            )
    norm_path = f"{CHECKPOINT_DIR}/{NORMALIZATION_FILE}"
    return load_stats(norm_path) if os.path.exists(norm_path) else None


def predict_target(model, sample, norm_stats):
    """Predicted target future (1, T_f, 2), relative to its last observed
    position (the frame create_sample uses)."""
    raw_states, _, _, mask = sample[:4]
    dims = sample[7].to(DEVICE) if len(sample) > 7 else None
    states = apply_normalization(raw_states, norm_stats) if norm_stats else raw_states
    with torch.no_grad():
        traj_dist, _, _ = model(states.to(DEVICE), mask.to(DEVICE), dims)
    return traj_dist[:, :, :2]


def constant_velocity(position, velocity, T_f):
    """Roll (M, 2) positions forward at constant (M, 2) velocity -> (1, T_f, M, 2)."""
    t = torch.arange(1, T_f + 1, device=DEVICE).view(T_f, 1, 1).float() * DT
    position = torch.as_tensor(position, dtype=torch.float32, device=DEVICE)
    velocity = torch.as_tensor(velocity, dtype=torch.float32, device=DEVICE)
    return (position + velocity * t).unsqueeze(0)


def _auc(fn, scores, gt, higher_is_riskier):
    try:
        return fn(scores, gt, higher_is_riskier=higher_is_riskier)
    except ValueError:  # no positive or no negative labels
        return float("nan")


def collision_report(metric_name, threshold, m, n_pairs):
    return [
        "Collision Prediction Metrics",
        f"Metric: {metric_name}, Threshold: {threshold}",
        "=" * 40,
        f"Total pairs evaluated: {n_pairs}",
        f"True Positives:  {int(m['tp'])}",
        f"False Positives: {int(m['fp'])}",
        f"False Negatives: {int(m['fn'])}",
        f"True Negatives:  {int(m['tn'])}",
        f"Accuracy:   {m['accuracy']:.4f}",
        f"Precision:  {m['precision']:.4f}",
        f"Recall:     {m['recall']:.4f}",
        f"F1 Score:   {m['f1']:.4f}",
    ]


def comparison_report(scores, gt, n_samples):
    header = (f"{'Metric':<15} {'Threshold':>10} {'Dir':>6} {'TP':>5} {'FP':>5} "
              f"{'FN':>5} {'TN':>5} {'Acc':>8} {'Prec':>8} {'Recall':>8} "
              f"{'F1':>8} {'FPR':>8} {'PR-AUC':>8} {'ROC-AUC':>8}")
    sep = "=" * len(header)
    lines = [
        "Risk Metric Comparison Report",
        f"Test samples: {n_samples}, Total pairs: {len(gt)}",
        f"GT collisions: {int(gt.sum())}, GT non-collisions: {int((1 - gt).sum())}",
        "", sep, header, sep,
    ]
    for name, (_, higher, default) in METRICS.items():
        pred = np.array([is_collision(name, v) for v in scores[name]], dtype=int)
        m = collision_metrics_from_labels(pred, gt)
        pr = _auc(compute_pr_auc, scores[name], gt, higher)
        roc = _auc(compute_roc_auc, scores[name], gt, higher)
        lines.append(
            f"{name:<15} {default:>10.3f} {'>' if higher else '<':>6} "
            f"{int(m['tp']):>5} {int(m['fp']):>5} {int(m['fn']):>5} {int(m['tn']):>5} "
            f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['false_alarm_rate']:>8.4f} {pr:>8.4f} {roc:>8.4f}"
        )
    lines.append(sep)
    return lines


def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved {path}")


# ------------------------------------------------------------------- modes

def cmd_prepare(args):
    loader = make_loader(args)
    train_s, val_s, test_s = loader.get_splits()
    stats = loader.get_dataset_stats()
    logger.info(
        "Dataset prepared: rows=%s, vehicles=%s, frames=%s, frame_range=%s, "
        "samples(train/val/test)=%s/%s/%s",
        stats.get("total_rows", 0), stats.get("unique_vehicles", 0),
        stats.get("num_frames", 0), stats.get("frame_range", "N/A"),
        len(train_s), len(val_s), len(test_s),
    )


def cmd_train(args):
    use_risk_loss = not args.basic_loss
    logger.info("Mode: " + ("STRAP-R (risk-scaled loss)" if use_risk_loss else "STRAP-B (basic loss)"))
    train(
        build_model(), make_loader(args),
        num_epochs=args.epochs, use_risk_loss=use_risk_loss, beta=BETA_LOSS,
        resume_from=args.resume, checkpoint_every=args.checkpoint_every,
        learning_rate=args.lr,
    )


def cmd_eval(args):
    model = build_model()
    norm_stats = load_trained(model)
    _, _, test_samples = make_loader(args).get_splits()
    if norm_stats:
        test_samples = normalize_samples(test_samples, norm_stats)
    if not test_samples:
        raise SystemExit("No test samples available.")
    # Overall, per site (problem D) and target-only scenes (problem 9).
    groups = {"NGSIM Test Split": test_samples}
    for sample in test_samples:
        groups.setdefault(f"site {location_of(sample[4][0])}", []).append(sample)
        if sample[0].shape[2] == 1:
            groups.setdefault("zero-neighbour samples", []).append(sample)
    for label, samples in groups.items():
        test_dl = create_dataloader(samples, batch_size=32, shuffle=False)
        print_metrics(
            evaluate_trajectory(model, test_dl, DEVICE),
            label=f"{label} (n={len(samples)})",
        )


def cmd_collisions(args):
    """Score every test target-candidate pair with all METRICS in one pass.

    Candidates are the nearest vehicles present at the last history frame,
    not the model's risk-selected neighbours, so conflicts the risk field
    missed are evaluated too (SOLUTIONS problem 9).
    """
    model = build_model()
    norm_stats = load_trained(model)
    loader = make_loader(args)
    _, _, test_samples = loader.get_splits()
    if not test_samples:
        raise SystemExit("No test samples available.")
    scene_indices = loader._split_indices["test"]
    T_h, T_f = loader.history_frames, loader.future_frames
    threshold, chosen = args.collision_threshold, args.collision_metric

    scores = {name: [] for name in METRICS}
    gt_labels, pairs = [], []
    for i, sample in enumerate(test_samples):
        ids, position, velocity, future = loader.collision_candidates(
            loader.scenes[scene_indices[i]]
        )
        # Pairs whose candidate leaves the road before the target's horizon
        # starts have no ground truth; drop them.
        seen = np.isfinite(future[..., 0]).any(axis=0)
        if not seen.any():
            continue
        ids, position, velocity, future = ids[seen], position[seen], velocity[seen], future[:, seen]
        target = predict_target(model, sample, norm_stats)
        neighbors = constant_velocity(position, velocity, T_f)
        gt_target = sample[1][0].numpy()[:, None, :]  # (T_f, 1, 2), same origin
        gt_d = np.nanmin(np.linalg.norm(future - gt_target, axis=-1), axis=0)
        for name, (fn, _, _) in METRICS.items():
            scores[name].extend(fn(target, neighbors, dt=DT, threshold=threshold)[0].tolist())
        gt_labels.extend((gt_d < threshold).astype(int).tolist())
        pairs.extend((sample[4][0], vehicle_id) for vehicle_id in ids)
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(test_samples)} samples...")
    gt = np.array(gt_labels)

    # Per-pair log and confusion matrix for the chosen metric.
    pred = [int(is_collision(chosen, v, threshold)) for v in scores[chosen]]
    rows = [f"Target_Vehicle_ID,Neighbor_Vehicle_ID,{chosen.upper()}_Value,GT_Collision"]
    rows += [
        f"{t},{n},{v:.3f},{g}"
        for (t, n), v, p, g in zip(pairs, scores[chosen], pred, gt_labels) if p or g
    ]
    _write(f"{LOG_DIR}/collisions.txt", rows)
    report = collision_report(chosen, threshold, collision_metrics_from_labels(pred, gt), len(gt))
    for line in report:
        logger.info(line)
    _write(f"{LOG_DIR}/collision_metrics.txt", report)

    comparison = comparison_report(scores, gt, len(test_samples))
    for line in comparison:
        logger.info(line)
    _write(f"{LOG_DIR}/metric_comparison.txt", comparison)


MODES = {"prepare": cmd_prepare, "train": cmd_train, "eval": cmd_eval, "collisions": cmd_collisions}


def main():
    parser = argparse.ArgumentParser(description="STRAP Trajectory Prediction Pipeline")
    parser.add_argument("--mode", default="train", choices=list(MODES))
    parser.add_argument("--debug", action="store_true", help="Small us-101 subset")
    parser.add_argument("--basic_loss", action="store_true", help="STRAP-B (no risk-scaled loss)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume training from")
    parser.add_argument("--checkpoint-every", type=int, default=2, help="Save epoch_N.pt every N epochs")
    parser.add_argument("--collision_metric", default="min_dist", choices=list(METRICS))
    parser.add_argument("--collision_threshold", type=float, default=2.0,
                        help="Distance (m) for GT collision labels and the chosen metric")
    parser.add_argument("--locations", default=None, help="Comma-separated NGSIM locations")
    args = parser.parse_args()
    MODES[args.mode](args)


if __name__ == "__main__":
    main()
