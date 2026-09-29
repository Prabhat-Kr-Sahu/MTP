"""One-off script: regenerate normalization.json with all 11 features.

The old normalization.json had only 10 features — O-field risk (index 10)
was missing. This rebuilds it from the us-101 cached data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import NGSIMDataLoader
from src.data.normalization import compute_normalization_stats, save_stats
from src.config import NORMALIZATION_FILE

loader = NGSIMDataLoader(location="us-101", max_rows=None)
loader.fetch()
train_samples, _, _ = loader.get_splits()
print(f"Train samples: {len(train_samples)}")

train_states_list = [s[0] for s in train_samples]
feature_dim = train_states_list[0].shape[-1]
print(f"Feature dimension per sample: {feature_dim}")

norm_stats = compute_normalization_stats(train_states_list)
print(f"Computed normalization for {len(norm_stats['mean'])} features")

# Save to checkpoint dir
checkpoint_dir = "checkpoints/strap_reproduction_v2"
os.makedirs(checkpoint_dir, exist_ok=True)
save_stats(norm_stats, f"{checkpoint_dir}/{NORMALIZATION_FILE}")
print(f"Saved to {checkpoint_dir}/{NORMALIZATION_FILE}")

# Verify
for i, (m, s) in enumerate(zip(norm_stats["mean"], norm_stats["std"])):
    print(f"  feat {i}: mean={m:.4f}, std={s:.4f}")
