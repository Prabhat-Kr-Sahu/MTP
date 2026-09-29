import json
import urllib.error

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import loader as loader_module
from src.data.loader import NGSIMDataLoader, create_dataloader
from src.data.generate_synthetic import create_sample


def _record(vehicle_id, frame_id, local_x):
    return {
        "vehicle_id": str(vehicle_id),
        "frame_id": str(frame_id),
        "local_x": str(local_x),
        "local_y": "20",
        "v_length": "15",
        "v_width": "6",
        "v_class": "2",
        "v_vel": "10",
        "v_acc": "1",
        "lane_id": "1",
    }


def test_fetch_ngsim_paginates_converts_and_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_page(location, limit, offset, app_token=None):
        calls.append(offset)
        pages = {
            0: [_record(1, 1, 10), _record(1, 2, 20)],
            2: [_record(1, 3, 30)],
        }
        return pages.get(offset, [])

    monkeypatch.setattr(loader_module, "_fetch_page", fake_fetch_page)

    data = loader_module.fetch_ngsim(
        cache_dir=str(tmp_path), page_size=2, max_rows=3
    )

    assert len(data) == 3
    assert data["local_x"].iloc[0] == 10 * loader_module._FT_TO_M
    assert calls == [0, 2]
    metadata_path = tmp_path / "ngsim_us-101.csv.meta.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["requested_max_rows"] == 3

    cached = loader_module.fetch_ngsim(
        cache_dir=str(tmp_path), page_size=2, max_rows=3
    )
    pd.testing.assert_frame_equal(data, cached)
    assert calls == [0, 2]

    loader_module.fetch_ngsim(cache_dir=str(tmp_path), page_size=2)
    assert calls == [0, 2, 0, 2]


def test_fetch_ngsim_retries_temporary_network_error(tmp_path, monkeypatch):
    calls = []

    def flaky_fetch_page(location, limit, offset, app_token=None):
        calls.append(offset)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary timeout")
        return [_record(1, 1, 10)]

    monkeypatch.setattr(loader_module, "_fetch_page", flaky_fetch_page)
    monkeypatch.setattr(loader_module.time, "sleep", lambda _: None)

    data = loader_module.fetch_ngsim(cache_dir=str(tmp_path), page_size=2)

    assert len(data) == 1
    assert calls == [0, 0]


def test_fetch_ngsim_resumes_saved_pages_after_interruption(tmp_path, monkeypatch):
    calls = []

    def interrupted_fetch_page(location, limit, offset, app_token=None):
        calls.append(offset)
        if offset == 0:
            return [_record(1, 1, 10), _record(1, 2, 20)]
        raise KeyboardInterrupt

    monkeypatch.setattr(loader_module, "_fetch_page", interrupted_fetch_page)
    with pytest.raises(KeyboardInterrupt):
        loader_module.fetch_ngsim(cache_dir=str(tmp_path), page_size=2)

    page_dir = tmp_path / ".ngsim_us-101_full_p2.pages"
    assert (page_dir / "page_000000000000.csv").exists()

    def resumed_fetch_page(location, limit, offset, app_token=None):
        calls.append(offset)
        return [_record(1, 3, 30)] if offset == 2 else []

    monkeypatch.setattr(loader_module, "_fetch_page", resumed_fetch_page)
    data = loader_module.fetch_ngsim(cache_dir=str(tmp_path), page_size=2)

    assert len(data) == 3
    assert calls == [0, 2, 2]


def test_build_scenes_uses_indexed_complete_windows():
    rows = []
    for vehicle_id in (1, 2, 3):
        for frame_id in range(5):
            rows.append({
                "vehicle_id": vehicle_id,
                "frame_id": frame_id,
                "local_x": vehicle_id * 0.1 + frame_id * 0.01,
                "local_y": frame_id * 0.1,
                "v_vel": 2.0,
                "v_acc": 0.2,
                "v_class": 2,
                "v_length": 4.5,
                "v_width": 1.8,
                "lane_id": 1,
            })

    loader = NGSIMDataLoader(
        window_frames=4,
        history_frames=2,
        future_frames=2,
        stride=1,
        max_neighbors=2,
        min_vehicle_frames=4,
    )
    loader.df = pd.DataFrame(rows)

    scenes = loader.build_scenes()

    assert len(scenes) == 6
    assert [scene["vehicle_ids"][0] for scene in scenes] == [1, 1, 2, 2, 3, 3]
    assert [scene["frame_id_start"] for scene in scenes] == [0, 1, 0, 1, 0, 1]
    for scene in scenes:
        target_id = scene["vehicle_ids"][0]
        frame_start = scene["frame_id_start"]
        window_frames = np.arange(frame_start, frame_start + 4)
        expected_longitudinal = window_frames * 0.1
        expected_lateral = target_id * 0.1 + window_frames * 0.01
        np.testing.assert_allclose(scene["frames"][:, 0, 0], expected_longitudinal)
        np.testing.assert_allclose(scene["frames"][:, 0, 1], expected_lateral)


def test_create_sample_uses_fixed_target_origin():
    frames = np.array([
        [[10, 100, 5, 0, 1], [15, 108, 4, 0, 2]],
        [[11, 102, 5, 0, 2], [16, 110, 4, 0, 3]],
        [[13, 105, 5, 0, 3], [18, 113, 4, 0, 4]],
        [[16, 109, 5, 0, 4], [21, 118, 4, 0, 5]],
        [[20, 114, 5, 0, 5], [25, 124, 4, 0, 6]],
    ], dtype=np.float32)
    scene = {
        "vehicle_ids": np.array([10, 20]),
        "frames": frames,
        "vehicle_types": np.array([1, 1]),
        "vehicle_lengths": np.array([4.5, 4.5]),
        "vehicle_widths": np.array([1.8, 1.8]),
        "lane_ids": np.array([1, 1]),
        "lane_history": np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]),
    }
    translated_scene = {**scene, "frames": frames.copy()}
    translated_scene["frames"][:, :, :2] += np.array([1000, -500])

    sample = create_sample(scene, T_h=3, T_f=2)
    translated_sample = create_sample(translated_scene, T_h=3, T_f=2)

    states, target_future, neighbor_goals = sample[:3]
    shifted_states, shifted_target_future, shifted_neighbor_goals = translated_sample[:3]
    expected_history = torch.tensor([[5, 8], [5, 8], [5, 8]], dtype=torch.float32)
    torch.testing.assert_close(states[0, :, 1, :2], expected_history)
    assert states.shape[-1] == 11
    torch.testing.assert_close(
        states[0, :, 0, 4], torch.tensor([0.1, 0.2, 0.3])
    )
    torch.testing.assert_close(
        states[0, :, 0, 6], torch.tensor([0.5, 0.5, 1.0])
    )
    torch.testing.assert_close(target_future[0], torch.tensor([[3, 4], [7, 9]], dtype=torch.float32))
    torch.testing.assert_close(neighbor_goals[0, 0, :2], torch.tensor([12, 19], dtype=torch.float32))
    torch.testing.assert_close(states, shifted_states)
    torch.testing.assert_close(target_future, shifted_target_future)
    torch.testing.assert_close(neighbor_goals, shifted_neighbor_goals)
    torch.testing.assert_close(sample[5], torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(sample[6], torch.tensor([[13, 105]], dtype=torch.float32))
    torch.testing.assert_close(
        translated_sample[6], torch.tensor([[1013, -395]], dtype=torch.float32)
    )


def test_build_samples_keep_history_neighbors_without_future_goal():
    rows = []
    for frame_id in range(4):
        rows.append({
            "vehicle_id": 1,
            "frame_id": frame_id,
            "local_x": 0.0,
            "local_y": float(frame_id),
            "v_vel": 1.0,
            "v_acc": 0.0,
            "v_class": 2,
            "v_length": 4.5,
            "v_width": 1.8,
            "lane_id": 1,
        })
    for frame_id in range(2):
        rows.append({
            "vehicle_id": 2,
            "frame_id": frame_id,
            "local_x": 0.2,
            "local_y": float(frame_id) + 0.1,
            "v_vel": 1.0,
            "v_acc": 0.0,
            "v_class": 2,
            "v_length": 4.5,
            "v_width": 1.8,
            "lane_id": 1,
        })

    loader = NGSIMDataLoader(
        window_frames=4,
        history_frames=2,
        future_frames=2,
        stride=1,
        max_neighbors=2,
        min_vehicle_frames=4,
    )
    loader.df = pd.DataFrame(rows)
    samples = loader.build_samples()

    assert len(samples) == 1
    states, gt_pos, gt_goals, vehicle_mask, vehicle_ids, goal_mask, origin = samples[0]
    assert vehicle_ids.tolist() == [1, 2]
    assert vehicle_mask.all()
    assert not goal_mask[0, 0]
    assert torch.count_nonzero(gt_goals) == 0
    assert torch.isfinite(states).all()

    batch = next(iter(create_dataloader(samples, batch_size=1, shuffle=False)))
    assert len(batch) == 5
    assert batch[0].shape == (1, 2, 2, 11)
    assert batch[3].tolist() == [[True, True]]
    assert batch[4].tolist() == [[False]]


def test_temporal_splits_have_no_overlapping_windows():
    loader = NGSIMDataLoader(window_frames=4, history_frames=2, future_frames=2)
    loader.df = pd.DataFrame({"frame_id": np.arange(40)})
    loader.scenes = [{"frame_id_start": frame} for frame in range(40)]
    loader._samples_cache = [frame for frame in range(40)]

    train, val, test = loader.get_splits(
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    splits = [loader._split_indices[name] for name in ("train", "val", "test")]

    assert train and val and test
    for earlier, later in zip(splits, splits[1:]):
        earlier_end = max(loader.scenes[index]["frame_id_start"] + 3 for index in earlier)
        later_start = min(loader.scenes[index]["frame_id_start"] for index in later)
        assert later_start > earlier_end


def test_build_scenes_uses_frame_start_for_input_o_field():
    """O-field input risk must use frame_start (observation-time) positions,
    not midpoint_frame (future) positions. Verify by checking that the
    input_o_risk in each scene is consistent with frame_start data."""
    rows = []
    # Vehicle 1 (target) moves forward; vehicle 2 (neighbor) is stationary.
    # At frame_start (frame 0): target at local_y=0, neighbor at local_y=5 (ahead).
    # At midpoint (frame 4): target at local_y=20, neighbor at local_y=5 (behind).
    for frame_id in range(10):
        rows.append({
            "vehicle_id": 1,
            "frame_id": frame_id,
            "local_x": 10.0,    # lateral: constant
            "local_y": float(frame_id * 2),  # longitudinal: moving
            "v_vel": 2.0,
            "v_acc": 0.0,
            "v_class": 2,
            "v_length": 4.5,
            "v_width": 1.8,
            "lane_id": 1,
        })
    for frame_id in range(10):
        rows.append({
            "vehicle_id": 2,
            "frame_id": frame_id,
            "local_x": 20.0,    # lateral: constant (5m lateral offset)
            "local_y": 5.0,     # longitudinal: stationary
            "v_vel": 0.0,
            "v_acc": 0.0,
            "v_class": 2,
            "v_length": 4.5,
            "v_width": 1.8,
            "lane_id": 1,
        })

    loader = NGSIMDataLoader(
        window_frames=8, history_frames=4, future_frames=4,
        stride=4, max_neighbors=5, min_vehicle_frames=8,
    )
    loader.df = pd.DataFrame(rows)
    scenes = loader.build_scenes()

    assert len(scenes) > 0
    for scene in scenes:
        input_o_risk = scene["input_o_risk"]
        assert isinstance(input_o_risk, float)
        assert 0.0 <= input_o_risk <= 1.0
        # With the fix, input_o_risk must be based on frame_start positions:
        # target at (local_y=0, local_x=10) and neighbor at (local_y=5, local_x=20).
        # The O-field should be non-zero since vehicles are within interaction range.
        assert input_o_risk > 0.0, (
            f"input_o_risk={input_o_risk} should be >0 for nearby vehicles "
            f"at frame_start"
        )


def test_config_gamma_x_y_greater_than_one():
    """Paper Eq.1 requires gamma_x > 1 and gamma_y > 1."""
    from src.config import GAMMA_X, GAMMA_Y
    assert GAMMA_X > 1.0, f"GAMMA_X={GAMMA_X} must be > 1 per paper Eq.1"
    assert GAMMA_Y > 1.0, f"GAMMA_Y={GAMMA_Y} must be > 1 per paper Eq.1"


def test_normalization_has_11_features():
    """Normalization stats must have 11 features (9 base + 2 risk) matching STATE_DIM."""
    import os
    from src.config import STATE_DIM, NORMALIZATION_FILE
    from src.data.normalization import load_stats

    norm_path = f"checkpoints/strap_reproduction_v2/{NORMALIZATION_FILE}"
    if not os.path.exists(norm_path):
        pytest.skip(f"Normalization file not found at {norm_path}")

    stats = load_stats(norm_path)
    assert stats["feature_dim"] == STATE_DIM, (
        f"Normalization feature_dim={stats['feature_dim']} != STATE_DIM={STATE_DIM}"
    )
    assert len(stats["mean"]) == STATE_DIM
    assert len(stats["std"]) == STATE_DIM
    # No NaN values in stats
    for i, (m, s) in enumerate(zip(stats["mean"], stats["std"])):
        assert not (m != m), f"Feature {i} mean is NaN"
        assert not (s != s), f"Feature {i} std is NaN"
        assert s > 0, f"Feature {i} std={s} must be > 0"


def test_multi_location_loader():
    """NGSIMDataLoader should support multiple locations via `locations` parameter."""
    loader = NGSIMDataLoader(locations=["us-101", "i-80"], max_rows=100)
    assert loader.locations == ["us-101", "i-80"]
    assert loader.location == "us-101"  # primary for backward compat

    # Single location still works
    loader2 = NGSIMDataLoader(location="us-101", max_rows=100)
    assert loader2.locations == ["us-101"]


def test_fetch_ngsim_multi_concatenates():
    """fetch_ngsim_multi should concatenate locations with frame_id offset."""
    from src.data.loader import fetch_ngsim_multi
    import pandas as pd

    # Create fake cached data for two locations
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Location 1: frames 0-9, vehicle 1
        df1 = pd.DataFrame({
            "vehicle_id": [1] * 10,
            "frame_id": list(range(10)),
            "local_x": [0.0] * 10,
            "local_y": list(range(10)),
            "v_vel": [1.0] * 10,
            "v_acc": [0.0] * 10,
            "v_class": [2] * 10,
            "v_length": [4.5] * 10,
            "v_width": [1.8] * 10,
            "lane_id": [1] * 10,
            "location": ["us-101"] * 10,
        })
        df1.to_csv(f"{tmpdir}/ngsim_us-101.csv", index=False)

        # Location 2: frames 0-9, vehicle 2
        df2 = pd.DataFrame({
            "vehicle_id": [2] * 10,
            "frame_id": list(range(10)),
            "local_x": [0.0] * 10,
            "local_y": list(range(10)),
            "v_vel": [1.0] * 10,
            "v_acc": [0.0] * 10,
            "v_class": [2] * 10,
            "v_length": [4.5] * 10,
            "v_width": [1.8] * 10,
            "lane_id": [1] * 10,
            "location": ["i-80"] * 10,
        })
        df2.to_csv(f"{tmpdir}/ngsim_i-80.csv", index=False)

        combined = fetch_ngsim_multi(
            locations=["us-101", "i-80"],
            cache_dir=tmpdir,
            max_rows_per_location=10,
        )

        assert len(combined) == 20
        # frame_ids should be offset: first location 0-9, second 10-19
        assert combined["frame_id"].min() == 0
        assert combined["frame_id"].max() == 19
        assert set(combined["vehicle_id"].unique()) == {1, 2}
        # location column should be preserved
        assert "location" in combined.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
