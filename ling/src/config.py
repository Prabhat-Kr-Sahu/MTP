"""STRAP configuration and hyperparameters."""

# --- Dataset ---
DATASET_NAME = "NGSIM"
DATA_DIR = "data/raw/ngsim"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"

# NGSIM locations available via Socrata API (8ect-6jqj)
# The STRAP paper uses NGSIM (plural locations) to reach 7.7M trajectories.
# We load us-101 + i-80 (freeway datasets) to better match the paper's scope.
NGSIM_LOCATIONS = ["us-101", "i-80"]

# --- Temporal ---
T_H = 3          # history length (seconds)
T_F = 5          # future prediction horizon (seconds)
WINDOW_LENGTH = 8  # total window (seconds)
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
# Paper Eq.1 requires gamma_x > 1 and gamma_y > 1.
# The paper cites [21] (Wang et al. 2022) for risk-field calibration but
# does not provide exact numeric values in the PDF. We use 1.5 as a
# documented placeholder that satisfies the constraint.
# REPRODUCTION ASSUMPTION: gamma values are not recovered from [21];
# label results accordingly.
RISK_THRESHOLD = 0.005
GAMMA_X = 1.5       # S-field longitudinal scaling — paper requires > 1
GAMMA_Y = 1.5       # S-field lateral scaling — paper requires > 1
ALPHA_X = 2.0       # S-field longitudinal shape factor (paper: >= 2)
ALPHA_Y = 2.0       # S-field lateral shape factor (paper: >= 2)
D_STAR = 5.0        # O-field distance scaling factor (reproduction assumption)
T_STAR = 2.0        # O-field time scaling factor (reproduction assumption)
BETA_1 = 1.0        # O-field distance shape factor (paper: >= 1)
BETA_2 = 1.0        # O-field time shape factor (paper: >= 1)
BETA_LOSS = 0.0     # risk-scaled loss bias term (paper Eq.13; value unspecified)

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
# + lane_id(1) + length(1) + width(1) = 9 base + 2 risk features
STATE_DIM = 11
BASE_STATE_DIM = 9

# --- Paths ---
CHECKPOINT_DIR = "checkpoints/strap_reproduction_v2"
LOG_DIR = "logs/strap_reproduction_v2"
RESULT_DIR = "results/strap_reproduction_v2"
NORMALIZATION_FILE = "normalization.json"
INTENTIONS_FILE = "intention_clusters.pkl"

# --- Device ---
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config():
    """Return all config as a dict."""
    import inspect
    config = {}
    for name, value in list(globals().items()):
        if not name.startswith("_") and not inspect.ismodule(value):
            config[name] = value
    return config
