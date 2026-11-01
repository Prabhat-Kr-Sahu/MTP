"""STRAP configuration and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root

# --- Dataset ---
DATA_DIR = str(ROOT / "data" / "raw" / "ngsim")
DEBUG_MAX_ROWS = 250_000   # --debug row cap per location

# NGSIM locations available via Socrata API (8ect-6jqj)
# The STRAP paper uses NGSIM (plural locations) to reach 7.7M trajectories.
# We load us-101 + i-80 (freeway datasets) to better match the paper's scope.
NGSIM_LOCATIONS = ["us-101", "i-80"]

# --- Temporal ---
T_H = 3          # history length (seconds)
T_F = 5          # future prediction horizon (seconds)
DT = 0.1         # sampling interval (seconds), NGSIM is typically 10 Hz
TH_STEPS = int(T_H / DT)   # 30
TF_STEPS = int(T_F / DT)   # 50

# --- Model ---
D = 64              # hidden feature dimension
K = 100             # number of spatial intention modes
N_V = 15            # maximum number of neighbor vehicles
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1

# --- Attention ---
NUM_HEADS = 4       # multi-head attention heads

# --- Risk Field ---
# The paper gives no values. REPRODUCTION ASSUMPTION (SOLUTIONS problem 4):
# full calibration of reference [22] (Zuo et al. 2025, arXiv 2504.21158).
# S-field acts on bumper-to-bumper gaps; gamma_x and alpha_x are cubics in the
# ego (target) speed v [m/s], coefficients highest power first.
RISK_THRESHOLD = 0.005
GAMMA_X_POLY = (5.1053e-4, -3.7051e-2, 1.0621, 1.2925)   # 5.7 m @5 m/s -> 12.7 m @25 m/s
ALPHA_X_POLY = (2.2214e-5, -1.4834e-3, 9.6673e-3, 3.2589)  # ~3.3 -> 2.8
GAMMA_Y = 1.4310    # S-field lateral scale (m)
ALPHA_Y = 4.9956    # S-field lateral shape
# O-field: perpendicular miss distance, d* = 0.5 (w_i + w_j); 0 when moving apart.
BETA_1 = 10.0       # O-field distance shape
T_STAR = 7.5        # O-field time scale (s)
BETA_2 = 2.0        # O-field time shape
BETA_LOSS = 0.0     # risk-scaled loss bias term (paper Eq.13; value unspecified)
# Fallback vehicle size (m) when a caller gives no dimensions.
DEFAULT_LENGTH = 4.5
DEFAULT_WIDTH = 1.8

# --- Training ---
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 12
LR_DECAY = 0.6      # per-epoch decay factor
WEIGHT_DECAY = 1e-5
SEED = 42

# --- Splits ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# --- Feature dimensions ---
# rel_pos(2) + vel(2) + longitudinal_acceleration(1) + vehicle_type(1)
# + relative_lane(1) + length(1) + width(1) = 9 base + 2 risk features
STATE_DIM = 11

# --- Paths ---
# v3 = after the SOLUTIONS fixes (problems 1-6, 9); v2 checkpoints are not comparable.
CHECKPOINT_DIR = str(ROOT / "checkpoints" / "strap_reproduction_v3")
LOG_DIR = str(ROOT / "logs" / "strap_reproduction_v3")
NORMALIZATION_FILE = "normalization.json"
INTENTIONS_FILE = "intention_clusters.pkl"

# --- Device ---
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

