"""Tests for MTP extension modules (intentions, collision, normalization)."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, 'C:/Users/prabh/OneDrive/Desktop/MTP/ling')

from src.intentions.kmeans import fit_intention_codebook
from src.evaluation.collision import (
    pairwise_min_distance,
    collision_label_from_trajectories,
    predicted_collision,
    collision_metrics_from_labels,
)
from src.data.normalization import compute_normalization_stats, apply_normalization


def test_kmeans_codebook_shape():
    endpoints = np.random.randn(500, 4).astype(np.float32) * 10
    centers = fit_intention_codebook(endpoints, k=100, seed=0)
    assert centers.shape == (100, 4)


def test_pairwise_distance():
    a = torch.zeros(2, 50, 2)
    b = torch.ones(2, 50, 2) * 5.0
    d_min, _ = pairwise_min_distance(a, b)
    assert d_min.shape == (2,)
    assert (d_min > 0).all()


def test_collision_label():
    a = torch.zeros(1, 10, 2)
    b = torch.zeros(1, 10, 2)  # same position -> collision
    label = collision_label_from_trajectories(a, b, threshold=2.0)
    assert label.item() == 1.0
    c = torch.ones(1, 10, 2) * 50.0
    label2 = collision_label_from_trajectories(a, c, threshold=2.0)
    assert label2.item() == 0.0


def test_collision_metrics():
    m = collision_metrics_from_labels([1, 0, 1, 0], [1, 0, 0, 0])
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"] <= 1
    assert 0 <= m["f1"] <= 1


def test_normalization_roundtrip():
    states = [torch.randn(1, 30, 5, 12) * 5 + 2 for _ in range(4)]
    stats = compute_normalization_stats(states)
    normed = apply_normalization(states[0], stats)
    assert normed.shape == states[0].shape
    assert torch.isfinite(normed).all()
