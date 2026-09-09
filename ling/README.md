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
ling/
├── src/
│   ├── config.py              # Hyperparameters and configuration
│   ├── pipeline.py            # Main entry point (train/eval/predict)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # NGSIM data loader utilities
│   │   ├── generate_synthetic.py  # Synthetic data generator for testing
│   │   └── __init__.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── s_field.py         # Subjective spatial proximity risk
│   │   ├── o_field.py         # Objective future collision risk
│   │   └── risk_features.py   # Combined risk computation + neighborhood selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── motion_encoder.py  # FC + ELU + LSTM
│   │   ├── spatial_encoder.py # MultiHeadAttn + GLU + LayerNorm
│   │   ├── temporal_encoder.py # PE + MultiHeadAttn + GLU + LayerNorm
│   │   ├── goal_predictor.py  # MLP for neighbor goal prediction
│   │   ├── risk_decoder.py    # Risk-attentive feature fusion
│   │   └── strap.py           # Full STRAP model
│   ├── losses/
│   │   ├── __init__.py
│   │   └── loss.py            # Goal loss, trajectory loss, risk-scaled loss
│   ├── training/
│   │   ├── train.py           # Training loop and validation
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── evaluate.py        # RMSE, collision metrics
│   │   └── __init__.py
│   └── intentions/
│       └── kmeans.py          # k-means intention mode generation
├── tests/
│   ├── __init__.py
│   ├── test_risk.py           # Risk field unit tests
│   └── test_model.py          # Model architecture unit tests
├── checkpoints/               # Saved model weights
├── logs/                      # Training logs
├── results/                   # Evaluation results
├── plan.md                    # Detailed implementation plan
├── README.md                  # This file
└── strap.pdf                  # Original STRAP paper
```

## Installation

```bash
# Navigate to the project directory
cd C:\Users\prabh\OneDrive\Desktop\MTP\ling

# Install dependencies (PyTorch 2.5.1+ already installed)
pip install torch numpy pandas scikit-learn matplotlib pytest
```

## Quick Start

### Train
```bash
python src/pipeline.py --mode train
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

### Predict
```bash
python src/pipeline.py --mode predict
```

### Run tests
```bash
python -m pytest tests/ -v
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

Since NGSIM data is not included in this repository, a synthetic data generator (`src/data/generate_synthetic.py`) is provided for testing the full pipeline. To use real NGSIM data:

1. Download NGSIM data from [ITS DataHub](https://data.transportation.gov/)
2. Place the data in `data/raw/ngsim/`
3. Update `src/data/loader.py` with the appropriate parsing logic
4. The data loader expects CSV files with columns: `vehicle_id, frame_id, local_x, local_y, velocity, acceleration, vehicle_type, lane_id`

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
- **NGSIM data**: The current pipeline uses synthetic data. NGSIM data loader requires actual data files.
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
