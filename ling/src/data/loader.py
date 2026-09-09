"""NGSIM data loader utilities."""
import torch
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class NGSIMDataLoader:
    """Loader for NGSIM trajectory data."""

    def __init__(self, data_dir: str = "data/raw/ngsim"):
        self.data_dir = data_dir
        self.scenes = []

    def load_scenes(self) -> list:
        """Load and validate NGSIM scene data."""
        raise NotImplementedError(
            "NGSIM data not present. Use generate_synthetic.py for testing."
        )

    def get_dataset_stats(self) -> dict:
        """Return dataset statistics."""
        return {"num_scenes": len(self.scenes)}


class TrafficDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for traffic trajectory data."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloader(samples: list, batch_size: int = 128, shuffle: bool = True):
    """Create a DataLoader from sample list."""
    dataset = TrafficDataset(samples)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """Custom collate function for variable-sized scenes."""
    states = torch.stack([b[0] for b in batch])
    gt_positions = torch.stack([b[1] for b in batch])
    gt_goals = torch.stack([b[2] for b in batch])
    return states, gt_positions, gt_goals
