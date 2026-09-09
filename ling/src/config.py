"""STRAP configuration and hyperparameters."""

# --- Dataset ---
DATASET_NAME = "NGSIM"
DATA_DIR = "data/raw/ngsim"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"

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
RISK_THRESHOLD = 0.005
GAMMA_X = 1.0       # S-field longitudinal scaling (TO VERIFY)
GAMMA_Y = 1.0       # S-field lateral scaling (TO VERIFY)
ALPHA_X = 2.0       # S-field longitudinal shape factor
ALPHA_Y = 2.0       # S-field lateral shape factor
D_STAR = 5.0        # O-field distance scaling factor (TO VERIFY)
T_STAR = 2.0        # O-field time scaling factor (TO VERIFY)
BETA_1 = 1.0        # O-field distance shape factor (TO VERIFY)
BETA_2 = 1.0        # O-field time shape factor (TO VERIFY)
BETA_LOSS = 0.0     # risk-scaled loss bias term (TO VERIFY)

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
# rel_pos(2) + vel(2) + vehicle_type(1) + lane_id(1) + length(1) + width(1)
# = 8 base + S-field (1) + O-field (1) = 10 total model inputs
STATE_DIM = 10  # total input features per vehicle per timestep
BASE_STATE_DIM = 8  # without risk features

# --- Paths ---
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
RESULT_DIR = "results"
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
