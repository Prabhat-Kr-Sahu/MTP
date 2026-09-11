"""Training pipeline for STRAP."""
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import os

from src.config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, LR_DECAY,
    WEIGHT_DECAY, CHECKPOINT_DIR, LOG_DIR, SEED,
)
from src.losses.loss import risk_scaled_loss, basic_loss, gaussian_nll
from src.data.loader import NGSIMDataLoader, create_dataloader


def train_one_epoch(model, optimizer, dataloader, device, use_risk_loss=True, beta=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_goal_loss = 0.0
    total_traj_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 4:
            states, gt_positions, gt_goals, mask = batch
            mask = mask.to(device)
        else:
            states, gt_positions, gt_goals = batch
            mask = None

        states = states.to(device)
        gt_positions = gt_positions.to(device)
        gt_goals = gt_goals.to(device)

        # Forward pass
        traj_dist, pred_goals, risk_field = model(states, mask)

        # Compute loss
        if use_risk_loss:
            loss, gamma, gl, tl = risk_scaled_loss(
                pred_goals, gt_goals, traj_dist, gt_positions,
                risk_field, beta
            )
        else:
            loss, gl, tl = basic_loss(pred_goals, gt_goals, traj_dist, gt_positions)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_goal_loss += gl
        total_traj_loss += tl
        num_batches += 1

    return total_loss / num_batches, total_goal_loss / num_batches, total_traj_loss / num_batches


def validate(model, dataloader, device, use_risk_loss=True, beta=0.0):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                states, gt_positions, gt_goals, mask = batch
                mask = mask.to(device)
            else:
                states, gt_positions, gt_goals = batch
                mask = None

            states = states.to(device)
            gt_positions = gt_positions.to(device)
            gt_goals = gt_goals.to(device)

            traj_dist, pred_goals, risk_field = model(states, mask)

            # Compute loss
            if use_risk_loss:
                loss, _, _, _ = risk_scaled_loss(
                    pred_goals, gt_goals, traj_dist, gt_positions, risk_field, beta
                )
            else:
                loss, _, _ = basic_loss(pred_goals, gt_goals, traj_dist, gt_positions)

            # RMSE at each horizon
            pred_mu = traj_dist[:, :, :2]  # (B, T_f, 2)
            rmse = torch.norm(pred_mu - gt_positions, dim=-1).mean().item()

            total_loss += loss.item()
            total_rmse += rmse
            num_samples += 1

    return total_loss / num_samples, total_rmse / num_samples


def train(
    model,
    num_epochs: int = NUM_EPOCHS,
    use_risk_loss: bool = True,
    beta: float = 0.0,
    debug: bool = False,
):
    """Full training loop."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY)

    best_val_loss = float("inf")
    training_log = []

    print(f"Training STRAP on {DEVICE}")
    print(f"Epochs: {num_epochs}, Risk Loss: {use_risk_loss}, Beta: {beta}, Debug: {debug}")

    # Load dataset
    max_rows = 50000 if debug else None
    loader = NGSIMDataLoader(location='us-101', max_rows=max_rows)
    loader.fetch()
    train_samples, val_samples, _ = loader.get_splits(seed=SEED)

    train_dl = create_dataloader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = create_dataloader(val_samples, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Number of training batches: {len(train_dl)}")
    print(f"Number of validation batches: {len(val_dl)}")

    for epoch in range(num_epochs):
        train_loss, gl, tl = train_one_epoch(
            model, optimizer, train_dl, DEVICE, use_risk_loss, beta
        )
        val_loss, val_rmse = validate(
            model, val_dl, DEVICE, use_risk_loss, beta
        )

        scheduler.step()

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "goal_loss": gl,
            "traj_loss": tl,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_log.append(log_entry)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f} (goal={gl:.4f}, traj={tl:.4f}), "
              f"Val Loss={val_loss:.4f}, Val RMSE={val_rmse:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save training log
    with open(f"{LOG_DIR}/training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/final_model.pt")

    print("Training complete!")
    return training_log
