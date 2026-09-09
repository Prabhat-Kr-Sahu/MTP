"""Main pipeline entry point for STRAP."""
import argparse
import os
import sys

# Allow `python src/pipeline.py` (script dir = src/) and `python -m src.pipeline`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json

from src.config import DEVICE, CHECKPOINT_DIR, NUM_EPOCHS, LEARNING_RATE, DROPOUT
from src.models.strap import STRAP
from src.training.train import train
from src.evaluation.evaluate import evaluate_trajectory, print_metrics
from src.data.generate_synthetic import generate_synthetic_trajectory, create_sample


def main():
    parser = argparse.ArgumentParser(description="STRAP Trajectory Prediction Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"])
    parser.add_argument("--debug", action="store_true", help="Use small dataset for debugging")
    parser.add_argument("--use_risk_loss", action="store_true", default=True, help="Use risk-scaled loss (STRAP-R)")
    parser.add_argument("--basic_loss", action="store_true", help="Use basic loss (STRAP-B)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
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

    print(f"STRAP model initialized on {DEVICE}")
    print(f"  Input dim: {input_dim}, Hidden dim: 64, Intention modes: 100")
    print(f"  Encoder layers: 3, Decoder layers: 2")
    print(f"  Mode: {'STRAP-R (risk-scaled loss)' if use_risk_loss else 'STRAP-B (basic loss)'}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    if args.mode == "train":
        # For debugging, use fewer epochs
        num_epochs = 2 if args.debug else args.epochs

        training_log = train(
            model,
            num_epochs=num_epochs,
            use_risk_loss=use_risk_loss,
            beta=beta,
        )

        # Print final metrics
        print("\nTraining Summary:")
        print(f"  Best validation loss seen during training")
        print(f"  Final learning rate: {training_log[-1]['lr']:.6f}")

        # Verify model works
        model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pt", map_location=DEVICE, weights_only=True))
        model.eval()

        # Quick sanity check with synthetic data
        test_scene = generate_synthetic_trajectory(num_vehicles=5, seed=999)
        states, gt_pos, _, _ = create_sample(test_scene, T_h=30, T_f=50)
        states = states.to(DEVICE)
        with torch.no_grad():
            traj_dist, goals, risk = model(states)
        print(f"\nSanity check output shapes:")
        print(f"  Traj dist: {traj_dist.shape}  (expected: 1x50x5)")
        print(f"  Goals: {goals.shape}  (expected: 1x{states.shape[2]-1}x4)")
        print(f"  Risk field: {risk.shape}  (expected: 1x100x2)")
        print(f"  Trajectory means: {traj_dist[0, 0, :2].detach().cpu().numpy()}")
        print(f"  Sigma: {traj_dist[0, 0, 2:4].detach().cpu().numpy()}")
        print(f"  Rho: {traj_dist[0, 0, 4].detach().cpu().numpy()}")

    elif args.mode == "eval":
        checkpoint_path = f"{CHECKPOINT_DIR}/best_model.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
            print(f"Loaded model from {checkpoint_path}")
        else:
            print("No checkpoint found. Run training first.")
            return

        model.eval()
        test_scene = generate_synthetic_trajectory(num_vehicles=10, seed=42)
        states, gt_pos, gt_goals, _ = create_sample(test_scene, T_h=30, T_f=50)
        states = states.to(DEVICE)
        gt_pos = gt_pos.to(DEVICE)
        with torch.no_grad():
            traj_dist, _, _ = model(states)

        metrics = evaluate_trajectory(model, [(states, gt_pos, gt_goals)], DEVICE)
        print_metrics(metrics, label="Synthetic Test")

    elif args.mode == "predict":
        checkpoint_path = f"{CHECKPOINT_DIR}/best_model.pt"
        if not os.path.exists(checkpoint_path):
            print("No checkpoint found. Run training first.")
            return

        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # Load a scene and predict
        test_scene = generate_synthetic_trajectory(num_vehicles=10, seed=42)
        states, _, _, mask = create_sample(test_scene, T_h=30, T_f=50)
        states = states.to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.no_grad():
            traj_dist, goals, risk = model(states, mask)

        print("Prediction complete!")
        print(f"  Predicted future positions shape: {traj_dist.shape}")
        print(f"  First timestep mean: ({traj_dist[0, 0, 0].item():.2f}, {traj_dist[0, 0, 1].item():.2f})")
        print(f"  First timestep uncertainty: sigma_x={traj_dist[0, 0, 2].item():.3f}, sigma_y={traj_dist[0, 0, 3].item():.3f}")
        print(f"  Correlation: {traj_dist[0, 0, 4].item():.3f}")
        print(f"  Risk field shape: {risk.shape}")


if __name__ == "__main__":
    main()
