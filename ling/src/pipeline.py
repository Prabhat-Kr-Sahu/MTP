"""Main pipeline entry point for STRAP."""
import argparse
import os
import sys

# Allow `python src/pipeline.py` (script dir = src/) and `python -m src.pipeline`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
logger = get_logger()

import torch
import json

from src.config import DEVICE, CHECKPOINT_DIR, NUM_EPOCHS, LEARNING_RATE, DROPOUT, NORMALIZATION_FILE, INTENTIONS_FILE, LOG_DIR
from src.models.strap import STRAP
from src.training.train import train
from src.evaluation.evaluate import evaluate_trajectory, print_metrics
from src.data.loader import NGSIMDataLoader, create_dataloader
from src.data.normalization import apply_normalization, load_stats
from src.intentions.kmeans import load_codebook


def main():
    parser = argparse.ArgumentParser(description="STRAP Trajectory Prediction Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict", "log_collisions"])
    parser.add_argument("--debug", action="store_true", help="Use small dataset for debugging")
    parser.add_argument("--use_risk_loss", action="store_true", default=True, help="Use risk-scaled loss (STRAP-R)")
    parser.add_argument("--basic_loss", action="store_true", help="Use basic loss (STRAP-B)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--risk_metrics", type=str, default="ttc,min_dist",
                        help="Comma-separated list of risk metrics to evaluate in predict mode")
    parser.add_argument("--collision_metric", type=str, default="min_dist",
                        help="Risk metric to use for logging collisions (default: min_dist)")
    parser.add_argument("--collision_threshold", type=float, default=2.0,
                        help="Threshold for the collision metric (default: 2.0)")
    args = parser.parse_args()

    use_risk_loss = not args.basic_loss and args.use_risk_loss
    beta = 0.0  # risk-scaled loss bias

    # Initialize model
    # State dimension: relative_pos(2) + vel(2) + type(1) + lane(1)
    #   + length(1) + width(1) = 8 base + risk(2) = 10 (see generate_synthetic.py)
    input_dim = 10
    model = STRAP(
        input_dim=input_dim,
        d_model=64,
        num_heads=4,
        k_intentions=100,
        num_encoder_layers=3,
        num_decoder_layers=2,
        dropout=DROPOUT,
    ).to(DEVICE)

    logger.info(f"STRAP model initialized on {DEVICE}")
    logger.info(f"  Input dim: {input_dim}, Hidden dim: 64, Intention modes: 100")
    logger.info(f"  Encoder layers: 3, Decoder layers: 2")
    logger.info(f"  Mode: {'STRAP-R (risk-scaled loss)' if use_risk_loss else 'STRAP-B (basic loss)'}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    if args.mode == "train":
        num_epochs = args.epochs

        training_log = train(
            model,
            num_epochs=num_epochs,
            use_risk_loss=use_risk_loss,
            beta=beta,
            debug=args.debug,
        )

        # Print final metrics
        logger.info("\nTraining Summary:")
        logger.info(f"  Best validation loss seen during training")
        logger.info(f"  Final learning rate: {training_log[-1]['lr']:.6f}")

        # Verify model works
        model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pt", map_location=DEVICE, weights_only=True))
        model.eval()

        # Quick sanity check with real data subset
        loader = NGSIMDataLoader(location='us-101', max_rows=50000)
        loader.fetch()
        samples = loader.build_samples(max_samples=1)
        if len(samples) > 0:
            norm_path = f"{CHECKPOINT_DIR}/{NORMALIZATION_FILE}"
            if os.path.exists(norm_path):
                norm_stats = load_stats(norm_path)
                states = apply_normalization(samples[0][0], norm_stats).to(DEVICE)
            else:
                states = samples[0][0].to(DEVICE)
            mask = samples[0][3].to(DEVICE)
            with torch.no_grad():
                traj_dist, goals, risk = model(states, mask)
            logger.info(f"\nSanity check output shapes:")
            logger.info(f"  Traj dist: {traj_dist.shape}  (expected: 1x50x5)")
            logger.info(f"  Goals: {goals.shape}  (expected: 1x{states.shape[2]-1}x4)")
            logger.info(f"  Risk field: {risk.shape}  (expected: 1x100x2)")

    elif args.mode == "eval":
        checkpoint_path = f"{CHECKPOINT_DIR}/best_model.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
            logger.info(f"Loaded model from {checkpoint_path}")
        else:
            logger.info("No checkpoint found. Run training first.")
            return

        model.eval()

        # Load normalization and intentions
        norm_path = f"{CHECKPOINT_DIR}/{NORMALIZATION_FILE}"
        intent_path = f"{CHECKPOINT_DIR}/{INTENTIONS_FILE}"
        norm_stats = load_stats(norm_path) if os.path.exists(norm_path) else None
        if os.path.exists(intent_path):
            centers = load_codebook(intent_path)
            model.risk_decoder.set_intentions(torch.from_numpy(centers))

        loader = NGSIMDataLoader(location='us-101', max_rows=50000)
        loader.fetch()
        _, _, test_samples = loader.get_splits()
        if norm_stats:
            test_samples = [
                (apply_normalization(s, norm_stats), gt, g, m, vids)
                for s, gt, g, m, vids in test_samples
            ]
        test_dl = create_dataloader(test_samples, batch_size=32, shuffle=False)

        metrics = evaluate_trajectory(model, test_dl, DEVICE)
        print_metrics(metrics, label="NGSIM Test Split")

    elif args.mode == "predict":
        checkpoint_path = f"{CHECKPOINT_DIR}/best_model.pt"
        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint found. Run training first.")
            return

        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # Load normalization and intentions
        norm_path = f"{CHECKPOINT_DIR}/{NORMALIZATION_FILE}"
        intent_path = f"{CHECKPOINT_DIR}/{INTENTIONS_FILE}"
        norm_stats = load_stats(norm_path) if os.path.exists(norm_path) else None
        if os.path.exists(intent_path):
            centers = load_codebook(intent_path)
            model.risk_decoder.set_intentions(torch.from_numpy(centers))

        # Load a scene and predict
        loader = NGSIMDataLoader(location='us-101', max_rows=50000)
        loader.fetch()
        samples = loader.build_samples(max_samples=1)
        if len(samples) > 0:
            states, gt_pos, gt_goals, mask, vehicle_ids = samples[0]
            if norm_stats:
                states = apply_normalization(states, norm_stats)
            states = states.to(DEVICE)
            mask = mask.to(DEVICE)

            with torch.no_grad():
                traj_dist, goals, risk = model(states, mask)

        logger.info("Prediction complete!")
        logger.info(f"  Predicted future positions shape: {traj_dist.shape}")
        logger.info(f"  First timestep mean: ({traj_dist[0, 0, 0].item():.2f}, {traj_dist[0, 0, 1].item():.2f})")
        logger.info(f"  First timestep uncertainty: sigma_x={traj_dist[0, 0, 2].item():.3f}, sigma_y={traj_dist[0, 0, 3].item():.3f}")
        logger.info(f"  Correlation: {traj_dist[0, 0, 4].item():.3f}")
        logger.info(f"  Risk field shape: {risk.shape}")

        # Evaluate extensible risk metrics
        from src.evaluation.risk_metrics import RiskMetricRegistry

        target_pred_mu = traj_dist[:, :, :2]  # (1, T_f, 2)
        # gt_pos represents ground-truth future positions (B, T_f, 2) for target
        # Wait, the current create_sample returns gt_pos for the target only.
        # We need the neighbor futures to evaluate collisions.
        # Actually, neighbor_goals contains (N_v, 4) [end_x, end_y, end_vx, end_vy].
        # For full trajectory evaluation, we'd need neighbor_trajs. Since our pipeline
        # currently doesn't return full neighbor GT trajectories from create_sample,
        # we will approximate them using constant velocity from current state for demonstration,
        # or we need to update create_sample to return them.
        # Let's do a simple constant-velocity rollout for neighbors from their current state.
        
        # Neighbor current states: relative pos + vel
        # states: (1, T_h, N_v+1, 10). T_h-1 is the current timestep.
        current_neighbor_pos = states[:, -1, 1:, :2]  # (1, N_v, 2)
        current_neighbor_vel = states[:, -1, 1:, 2:4] * 30.0  # (1, N_v, 2) (un-normalize velocity)
        
        T_f = traj_dist.shape[1]
        dt = 0.1
        time_steps = torch.arange(1, T_f + 1, device=DEVICE).view(1, T_f, 1, 1).float() * dt
        
        # Approximate neighbor future: pos + vel * t
        neighbors_future = current_neighbor_pos.unsqueeze(1) + current_neighbor_vel.unsqueeze(1) * time_steps
        # shape: (1, T_f, N_v, 2)

        metrics_to_run = args.risk_metrics.split(",")
        logger.info("\nRisk Metric Evaluation:")
        for metric_name in metrics_to_run:
            metric_name = metric_name.strip()
            if not metric_name:
                continue
            try:
                metric = RiskMetricRegistry.get(metric_name)
                risk_values = metric.compute(target_pred_mu, neighbors_future, dt=dt)
                logger.info(f"  {metric_name.upper()}:")
                # print values for the first 5 neighbors (if available)
                nv_print = min(5, risk_values.shape[1])
                for i in range(nv_print):
                    logger.info(f"    Neighbor {i+1}: {risk_values[0, i].item():.3f}")
            except Exception as e:
                logger.info(f"  {metric_name.upper()}: Error computing metric - {e}")

    elif args.mode == "log_collisions":
        checkpoint_path = f"{CHECKPOINT_DIR}/best_model.pt"
        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint found. Run training first.")
            return

        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # Load normalization and intentions
        norm_path = f"{CHECKPOINT_DIR}/{NORMALIZATION_FILE}"
        intent_path = f"{CHECKPOINT_DIR}/{INTENTIONS_FILE}"
        norm_stats = load_stats(norm_path) if os.path.exists(norm_path) else None
        if os.path.exists(intent_path):
            centers = load_codebook(intent_path)
            model.risk_decoder.set_intentions(torch.from_numpy(centers))

        # We will use the specified collision metric
        from src.evaluation.risk_metrics import RiskMetricRegistry
        from src.evaluation.collision import collision_metrics_from_labels
        import numpy as np
        metric_name = args.collision_metric.strip()
        threshold = args.collision_threshold
        metric = RiskMetricRegistry.get(metric_name)
        
        # Determine subset size based on debug flag
        max_rows = 50000 if args.debug else None
        
        logger.info(f"Loading data (debug={args.debug})...")
        loader = NGSIMDataLoader(location='us-101', max_rows=max_rows)
        loader.fetch()
        _, _, test_samples = loader.get_splits()
        
        if not test_samples:
            logger.info("No test samples available.")
            return

        # Get the scene indices for test samples so we can access GT futures
        test_scene_indices = loader._split_indices['test']
            
        collision_log_path = f"{LOG_DIR}/collisions.txt"
        metrics_log_path = f"{LOG_DIR}/collision_metrics.txt"
        logger.info(f"Evaluating {len(test_samples)} test samples for collisions...")
        logger.info(f"Using metric '{metric_name}' with threshold {threshold}")
        logger.info(f"Writing collisions to {collision_log_path}")
        
        num_collisions = 0
        all_pred_labels = []  # binary: 1 = predicted collision
        all_gt_labels = []    # binary: 1 = GT collision
        
        T_h = loader.history_frames  # 30
        T_f = loader.future_frames   # 50
        
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(collision_log_path, "w") as f:
            f.write(f"Target_Vehicle_ID,Neighbor_Vehicle_ID,{metric_name.upper()}_Value,GT_Collision\n")
            
            for i, sample in enumerate(test_samples):
                states, gt_pos, gt_goals, mask, vehicle_ids = sample
                target_id = vehicle_ids[0]
                
                # Get the corresponding scene for GT neighbor futures
                scene_idx = test_scene_indices[i]
                scene = loader.scenes[scene_idx]
                scene_frames = scene["frames"]  # (80, N, 4)
                
                # GT neighbor future trajectories: frames[T_h:T_h+T_f, 1:, :2]
                n_scene_vehicles = scene_frames.shape[1]
                n_neighbors = n_scene_vehicles - 1
                gt_target_future = torch.from_numpy(
                    scene_frames[T_h:T_h + T_f, 0, :2]
                ).float().unsqueeze(0)  # (1, T_f, 2)
                gt_neighbor_futures = torch.from_numpy(
                    scene_frames[T_h:T_h + T_f, 1:, :2]
                ).float().unsqueeze(0)  # (1, T_f, N_v, 2)
                
                if norm_stats:
                    states = apply_normalization(states, norm_stats)
                states = states.to(DEVICE)
                mask = mask.to(DEVICE)

                with torch.no_grad():
                    traj_dist, goals, risk = model(states, mask)

                target_pred_mu = traj_dist[:, :, :2]
                
                # Approximate neighbor future: pos + vel * t (for predictions)
                current_neighbor_pos = states[:, -1, 1:, :2]  # (1, N_v, 2)
                current_neighbor_vel = states[:, -1, 1:, 2:4] * 30.0  # (1, N_v, 2)
                
                dt = 0.1
                time_steps = torch.arange(1, T_f + 1, device=DEVICE).view(1, T_f, 1, 1).float() * dt
                neighbors_future_pred = current_neighbor_pos.unsqueeze(1) + current_neighbor_vel.unsqueeze(1) * time_steps

                # Compute predicted risk metric
                risk_values = metric.compute(target_pred_mu, neighbors_future_pred, dt=dt, collision_threshold=threshold)
                
                # Compute GT collision labels using actual scene trajectories
                # GT min distance between target and each neighbor
                gt_dist = torch.norm(
                    gt_target_future.unsqueeze(2) - gt_neighbor_futures, dim=-1
                )  # (1, T_f, N_v)
                gt_min_dist = gt_dist.min(dim=1).values  # (1, N_v)
                
                # For each neighbor, check predicted vs GT collision
                n_eval = min(risk_values.shape[1], n_neighbors)
                for nv_idx in range(n_eval):
                    pred_val = risk_values[0, nv_idx].item()
                    gt_d = gt_min_dist[0, nv_idx].item()
                    
                    pred_collision = 1 if pred_val < threshold else 0
                    gt_collision = 1 if gt_d < threshold else 0
                    
                    all_pred_labels.append(pred_collision)
                    all_gt_labels.append(gt_collision)
                    
                    if pred_collision or gt_collision:
                        neighbor_id = vehicle_ids[nv_idx + 1]
                        f.write(f"{target_id},{neighbor_id},{pred_val:.3f},{gt_collision}\n")
                        if pred_collision:
                            num_collisions += 1
                        
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i+1}/{len(test_samples)} samples...")
                    
        logger.info(f"Finished evaluating collisions! Found {num_collisions} predicted colliding pairs.")
        
        # Compute and log classification metrics
        metrics = collision_metrics_from_labels(
            np.array(all_pred_labels), np.array(all_gt_labels)
        )
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Collision Prediction Metrics (metric={metric_name}, threshold={threshold})")
        logger.info(f"{'='*50}")
        logger.info(f"  Total pairs evaluated: {len(all_pred_labels)}")
        logger.info(f"  True Positives:  {int(metrics['tp'])}")
        logger.info(f"  False Positives: {int(metrics['fp'])}")
        logger.info(f"  False Negatives: {int(metrics['fn'])}")
        logger.info(f"  True Negatives:  {int(metrics['tn'])}")
        logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
        logger.info(f"{'='*50}")
        
        # Save metrics to file
        with open(metrics_log_path, "w") as f:
            f.write(f"Collision Prediction Metrics\n")
            f.write(f"Metric: {metric_name}, Threshold: {threshold}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Total pairs evaluated: {len(all_pred_labels)}\n")
            f.write(f"True Positives:  {int(metrics['tp'])}\n")
            f.write(f"False Positives: {int(metrics['fp'])}\n")
            f.write(f"False Negatives: {int(metrics['fn'])}\n")
            f.write(f"True Negatives:  {int(metrics['tn'])}\n")
            f.write(f"Accuracy:   {metrics['accuracy']:.4f}\n")
            f.write(f"Precision:  {metrics['precision']:.4f}\n")
            f.write(f"Recall:     {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:   {metrics['f1']:.4f}\n")
        logger.info(f"Metrics saved to {metrics_log_path}")

if __name__ == "__main__":
    main()

