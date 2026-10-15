"""Data loading and preprocessing utilities."""
from src.data.loader import NGSIMDataLoader, TrafficDataset, create_dataloader, collate_fn
from src.data.generate_synthetic import generate_synthetic_trajectory, create_sample
