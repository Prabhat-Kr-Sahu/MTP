# STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory Prediction

Implementation of the STRAP framework for autonomous driving trajectory prediction, as described in the paper *"STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory Prediction for Autonomous Driving"* by Xinyi Ning, Zilin Bian, Dachuan Zuo, and Semiha Ergan.

## Overview

STRAP is a trajectory prediction framework that incorporates a **risk potential field** to assess perceived risks from surrounding vehicles and embed them into spatial-temporal feature representations. The framework uses a **risk-scaled loss function** to improve prediction accuracy in high-risk scenarios.

Key contributions:
- **Spatial-temporal encoder** with multi-head attention for vehicle interactions
- **Risk-attentive feature fusion decoder** with 100 intention modes
- **Risk-scaled loss** that prioritizes safety-critical scenarios

## Architecture

```
Input (vehicle states + risk features)
    ↓
Motion Encoder (FC + ELU + LSTM)
    ↓
Spatial Encoder (MultiHeadAttn + GLU + LayerNorm) × 3 layers
    ↓
Temporal Encoder (PE + MultiHeadAttn + GLU + LayerNorm) × 3 layers
    ↓
Neighbor Goal Predictor (MLP)
    ↓
100 Intention Modes (k-means codebook)
    ↓
Future Risk Field (S-field + O-field)
    ↓
Risk-Attentive Cross-Attention
    ↓
Trajectory Generator (LSTM + MLP)
    ↓
Output: Bivariate Gaussian (μx, μy, σx, σy, ρ)
```

## Project Structure

```
MTP/
├── src/
│   ├── config.py              # Hyperparameters, paths and configuration
│   ├── pipeline.py            # Entry point: prepare | train | eval | collisions
│   ├── train.py               # Training loop and validation
│   ├── losses.py              # Goal loss, trajectory loss, risk-scaled loss
│   ├── risk.py                # S-field, O-field, closest approach, pairwise distance
│   ├── intentions.py          # k-means intention modes
│   ├── logger.py
│   ├── data/
│   │   ├── loader.py          # NGSIM loader, scenes, samples, splits, DataLoader
│   │   └── normalization.py   # Train-only feature normalization
│   ├── evaluation/
│   │   ├── evaluate.py        # Per-horizon RMSE
│   │   ├── collision.py       # Collision classification metrics, PR/ROC-AUC
│   │   └── risk_metrics.py    # METRICS dict of 11 trajectory risk metrics
│   └── models/                # motion/spatial/temporal encoders, goal predictor,
│                              # risk decoder, full STRAP model
├── tests/
├── conftest.py                # Puts the repo root on sys.path for pytest
├── checkpoints/ data/ logs/ results/   # git-ignored, generated
├── plan.md  STRAP_REPRODUCTION_PLAN.md
├── README.md
└── strap.pdf                  # Original STRAP paper
```

## Installation

```bash
# Navigate to the project directory
cd C:\Users\prabh\OneDrive\Desktop\MTP

# Install dependencies (PyTorch 2.5.1+ already installed)
pip install torch numpy pandas scikit-learn matplotlib pytest
```

## Quick Start

### Prepare the NGSIM dataset
```bash
python src/pipeline.py --mode prepare
```

This downloads the default `us-101` NGSIM data from the Socrata API,
converts measurements to metric units, caches the CSV under
`data/raw/ngsim/`, and logs dataset and train/validation/test sample counts.
Use `--debug` for a 50,000-row preparation run. Full training uses all rows.

### Train
```bash
python src/pipeline.py --mode train
```

Training saves `best_model.pt`, `final_model.pt`, `last_checkpoint.pt`, and
periodic `epoch_<N>.pt` checkpoints. Resume with:
```bash
python src/pipeline.py --mode train --resume checkpoints/strap_reproduction_v3/last_checkpoint.pt
```

### Train with basic loss (STRAP-B)
```bash
python src/pipeline.py --mode train --basic_loss
```

### Debug training
```bash
python src/pipeline.py --mode train --debug --epochs 2
```

### Evaluate
```bash
python src/pipeline.py --mode eval
```

### Collision evaluation
```bash
python src/pipeline.py --mode collisions [--collision_metric min_dist] [--collision_threshold 2.0]
```

Scores every test target-neighbour pair with all 11 risk metrics in one pass and writes
`collisions.txt`, `collision_metrics.txt` and `metric_comparison.txt` to the log directory.

### Run tests
```bash
python -m pytest tests -v
```

## Configuration

All hyperparameters are defined in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D` | 64 | Hidden feature dimension |
| `K` | 100 | Number of spatial intention modes |
| `N_V` | 15 | Maximum number of neighbor vehicles |
| `T_H` | 3 | History length (seconds) |
| `T_F` | 5 | Future prediction horizon (seconds) |
| `NUM_ENCODER_LAYERS` | 3 | Number of spatial-temporal encoder layers |
| `NUM_DECODER_LAYERS` | 2 | Number of risk fusion decoder layers |
| `BATCH_SIZE` | 128 | Training batch size |
| `LEARNING_RATE` | 0.0005 | Initial learning rate |
| `NUM_EPOCHS` | 12 | Number of training epochs |
| `LR_DECAY` | 0.6 | Per-epoch learning rate decay |
| `RISK_THRESHOLD` | 0.005 | Risk-aware neighborhood selection threshold |

## Dataset

The implementation is designed for **NGSIM** (Next Generation Simulation) dataset. The paper also evaluates on **HighD**.

NGSIM data is downloaded from the Socrata API by `--mode prepare` and cached in `data/raw/ngsim/`.

## Key Components

### Risk Potential Field

**S-field (Subjective):** Gaussian-based spatial proximity risk
$$r_{ij}^s = \exp\left(-\left|\frac{\Delta x_{ij}}{\gamma_x}\right|^{\alpha_x} - \left|\frac{\Delta y_{ij}}{\gamma_y}\right|^{\alpha_y}\right)$$

**O-field (Objective):** Future collision probability
$$r_{ij}^o = \exp\left[-\left(\frac{\hat{d}_{m,ij}}{d^*}\right)^{\beta_1}\right] \exp\left[-\left(\frac{\hat{t}_{m,ij}}{t^*}\right)^{\beta_2}\right]$$

### Loss Function

$$\gamma_{risk} = \max\left[\exp(R^s + R^o) - \beta, 1\right]$$
$$\mathcal{L}_{total} = \gamma_{risk} (\mathcal{L}_{goal} + \mathcal{L}_{traj})$$

Where:
- $\mathcal{L}_{goal}$: MSE loss for surrounding vehicle goal predictions
- $\mathcal{L}_{traj}$: MSE + Negative Log-Likelihood for trajectory prediction

## Implementation Notes

- **Parameter verification needed**: Several risk-field parameters (γx, γy, αx, αy, d*, t*, β1, β2, β) are marked as configurable hyperparameters that need verification against the authors' implementation.
- **k-means intentions**: Intention codebook (K=100) should be fitted on training data only.
- **Normalization**: Train-only normalization statistics should be saved and reused.

## MTP Extension

The implementation is designed to support the MTP project's final objective of **pairwise vehicle collision-risk prediction**:

1. STRAP provides the trajectory prediction engine
2. Pairwise trajectory analysis computes predicted minimum distances
3. Collision risk is estimated from predicted trajectories
4. Ground-truth collision labels are used only for evaluation

## References

1. Ning, X., Bian, Z., Zuo, D., & Ergan, S. "STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory Prediction for Autonomous Driving."
2. NGSIM Dataset — [ITS DataHub](https://data.transportation.gov/)
3. HighD Dataset — Krajewski et al. (2018)

## License

This implementation is for research and educational purposes.
