"""One check per solution in SOLUTIONS_PAPER_VS_IMPLEMENTATION.md."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ALPHA_X_POLY, GAMMA_X_POLY
from src.data.loader import (
    NGSIMDataLoader, collate_fn, create_dataloader, fetch_ngsim_multi, location_of,
)
from src.intentions import build_codebook_from_samples
from src.losses import risk_scaled_loss
from src.models.risk_decoder import RiskAttentiveDecoder
from src.models.strap import STRAP
from src.risk import _polyval, end_velocity, ground_truth_risk, pair_risk


def _row(vehicle_id, frame_id, local_y, local_x=0.0, lane=1, speed=1.0, global_time=None):
    return {
        "vehicle_id": vehicle_id, "frame_id": frame_id,
        "global_time": global_time if global_time is not None else frame_id * 100,
        "local_x": local_x, "local_y": local_y, "v_vel": speed, "v_acc": 0.0,
        "v_class": 2, "v_length": 4.5, "v_width": 1.8, "lane_id": lane,
    }


# ---- Problem 1: unique keys per (location, recording period) ----------------

def test_recordings_get_unique_vehicle_ids_and_frame_ranges(tmp_path):
    rows = []
    for frame in range(10):   # period 1 and 2 reuse vehicle 7 and frames 0-9
        rows.append(_row(7, frame, frame, global_time=(1000 + frame) * 100))
        rows.append(_row(7, frame, 500 + frame, global_time=(5000 + frame) * 100))
    rows.append(dict(rows[0]))  # an exact duplicate row
    pd.DataFrame(rows).to_csv(tmp_path / "ngsim_us-101.csv", index=False)

    df = fetch_ngsim_multi(["us-101"], cache_dir=str(tmp_path), max_rows_per_location=100)

    assert len(df) == 20                                   # duplicate dropped
    assert set(df["vehicle_id"]) == {1_100_007, 1_200_007}  # loc 1, periods 1 and 2
    assert set(df["orig_vehicle_id"]) == {7}
    by_vehicle = df.groupby("vehicle_id")["frame_id"].agg(["min", "max"])
    assert by_vehicle.loc[1_100_007].tolist() == [0, 9]
    assert by_vehicle.loc[1_200_007].tolist() == [90, 99]   # 80-frame gap
    # every (vehicle_id, frame_id) key is one car
    assert not df.duplicated(["vehicle_id", "frame_id"]).any()


# ---- Problem D: per-site reporting ------------------------------------------

def test_location_is_recoverable_from_vehicle_id():
    assert location_of(1_200_434) == "us-101"
    assert location_of(2_300_001) == "i-80"
    assert location_of(5) == "unknown"


# ---- Problem 2: gamma_risk from ground truth, no gradient path --------------

def _straight_track(speed, T=50, dt=0.1, y=0.0):
    t = torch.arange(1, T + 1, dtype=torch.float32) * dt
    return torch.stack([speed * t, torch.full_like(t, y)], dim=-1)


def test_ground_truth_risk_sees_true_conflicts_only():
    gt_positions = _straight_track(10.0).unsqueeze(0).repeat(2, 1, 1)  # ends at x=50, 10 m/s
    goals = torch.tensor([[[55.0, 0.0, 5.0, 0.0]]]).repeat(2, 1, 1)    # slower car 5 m ahead
    goal_mask = torch.tensor([[True], [False]])
    dims = torch.tensor([[4.5, 1.8], [4.5, 1.8]]).expand(2, 2, 2)

    risk = ground_truth_risk(gt_positions, goals, goal_mask, dims)

    assert risk.shape == (2, 2)
    assert risk[0, 0] > 0.1 and risk[0, 1] > 0.1   # close and closing: S and O fire
    torch.testing.assert_close(risk[1], torch.zeros(2))  # masked neighbour


def test_risk_scaled_loss_has_no_gradient_path_through_risk():
    pred_goals = torch.zeros(1, 1, 4, requires_grad=True)
    pred_dist = torch.zeros(1, 50, 5)
    pred_dist[:, :, 2:4] = 1.0
    risk = torch.tensor([[math.log(3.0), 0.0]], requires_grad=True)

    loss, gamma, _, _ = risk_scaled_loss(
        pred_goals, torch.ones(1, 1, 4), pred_dist, torch.zeros(1, 50, 2), risk, beta=1.0
    )
    loss.backward()

    assert gamma == pytest.approx(2.0)  # paper form: max(exp(R) - beta, 1)
    assert risk.grad is None
    assert pred_goals.grad is not None


# ---- Problem 3: frozen codebook, 1 s end velocity ---------------------------

def test_intention_codebook_is_a_frozen_buffer():
    decoder = RiskAttentiveDecoder(k_intentions=4)
    assert "intention_means" not in dict(decoder.named_parameters())
    assert "intention_means" in decoder.state_dict()
    decoder.set_intentions(torch.ones(4, 4))
    torch.testing.assert_close(decoder.intention_means, torch.ones(4, 4))


def test_codebook_velocity_spans_one_second():
    track = _straight_track(2.0)  # 2 m/s, so 1 s displacement = 2 m
    torch.testing.assert_close(end_velocity(track), torch.tensor([2.0, 0.0]))
    samples = [(None, track.unsqueeze(0), None)] * 3
    centers = build_codebook_from_samples(samples, k=1)
    np.testing.assert_allclose(centers[0], [10.0, 0.0, 2.0, 0.0], atol=1e-5)


# ---- Problem 4: [22] calibration --------------------------------------------

def test_s_field_scale_follows_the_22_polynomials():
    v = torch.tensor([5.0, 25.0])
    torch.testing.assert_close(_polyval(GAMMA_X_POLY, v), torch.tensor([5.74, 12.66]), atol=0.01, rtol=0)
    torch.testing.assert_close(_polyval(ALPHA_X_POLY, v), torch.tensor([3.27, 2.92]), atol=0.01, rtol=0)


def test_pair_risk_uses_gaps_and_only_closing_conflicts():
    car = torch.tensor([4.5, 1.8])
    rel_pos = torch.tensor([
        [4.0, 0.0],    # bumpers touching, same lane, same speed
        [20.0, 0.0],   # same lane, closing at 5 m/s
        [20.0, 0.0],   # same lane, moving apart
        [20.0, 3.6],   # next lane, closing
    ])
    rel_vel = torch.tensor([[0.0, 0.0], [-5.0, 0.0], [5.0, 0.0], [-5.0, 0.0]])

    s, o = pair_risk(rel_pos, rel_vel, 20.0, car, car.expand(4, 2))

    assert s[0] == pytest.approx(1.0)                       # gap 0 -> full S
    assert s[1] < 0.5
    assert o[0] == 0.0                                      # no relative motion
    assert o[1] == pytest.approx(math.exp(-(4.0 / 7.5) ** 2), rel=1e-5)  # t_m = 4 s, d_m = 0
    assert o[2] == 0.0                                      # moving apart
    assert o[3] < 1e-3                                      # miss distance 3.6 m >> d* = 1.8 m


# ---- Problem 5: split each recording by time, then pool ---------------------

def test_each_recording_is_split_70_10_20_with_guard_gaps():
    loader = NGSIMDataLoader(window_frames=4, history_frames=2, future_frames=2)
    segments = [("us-101", 1, np.arange(0, 100)), ("i-80", 1, np.arange(200, 300))]
    loader.df = pd.DataFrame([
        {"location": loc, "period": period, "frame_id": frame}
        for loc, period, frames in segments for frame in frames
    ])
    starts = [frame for _, _, frames in segments for frame in frames[:-3]]
    loader.scenes = [{"frame_id_start": int(frame)} for frame in starts]
    loader._samples_cache = list(range(len(starts)))

    loader.get_splits()

    for _, _, frames in segments:
        windows = {
            name: [loader.scenes[i]["frame_id_start"] for i in loader._split_indices[name]
                   if frames[0] <= loader.scenes[i]["frame_id_start"] <= frames[-1]]
            for name in ("train", "val", "test")
        }
        assert all(windows.values()), "every recording contributes to every split"
        assert max(windows["train"]) + 3 < min(windows["val"])
        assert max(windows["val"]) + 3 < min(windows["test"])


# ---- Problem 6: interleaved encoder blocks ----------------------------------

def test_encoder_runs_spatial_temporal_interleaved():
    model = STRAP(input_dim=11)
    calls = []
    for name, module in model.named_modules():
        if name.startswith(("spatial_encoders.", "temporal_encoders.")) and name.count(".") == 1:
            module.register_forward_hook(lambda m, i, o, name=name: calls.append(name[0].upper()))

    model(torch.randn(1, 30, 3, 11), torch.ones(1, 3, dtype=torch.bool))

    assert calls == ["S", "T", "S", "T", "S", "T"]


# ---- Problem 9: no nearest-15 fallback, honest collision candidates ---------

def test_no_fallback_neighbours_but_far_car_stays_a_collision_candidate():
    rows = [_row(1, f, 0.2 * f, speed=2.0) for f in range(4)]
    rows += [_row(2, f, 100.0 + 0.2 * f, speed=2.0) for f in range(4)]  # 100 m ahead, same speed
    loader = NGSIMDataLoader(window_frames=4, history_frames=2, future_frames=2,
                             stride=1, max_neighbors=15, min_vehicle_frames=4)
    loader.df = pd.DataFrame(rows)

    scenes = loader.build_scenes()
    target_scene = next(scene for scene in scenes if scene["vehicle_ids"][0] == 1)

    assert target_scene["vehicle_ids"].tolist() == [1]      # target only
    ids, position, velocity, future = loader.collision_candidates(target_scene)
    assert ids.tolist() == [2]
    np.testing.assert_allclose(position, [[100.0, 0.0]])
    np.testing.assert_allclose(velocity, [[2.0, 0.0]])
    # relative to the target's last observed position (0.2 m), like the labels
    np.testing.assert_allclose(future[:, 0, 0], [100.2, 100.4])


def test_all_zero_neighbour_batch_runs_through_model_and_losses():
    loader = NGSIMDataLoader(window_frames=4, history_frames=2, future_frames=2,
                             stride=1, min_vehicle_frames=4)
    rows = [_row(1, f, 0.2 * f, speed=2.0) for f in range(4)]
    loader.df = pd.DataFrame(rows)
    samples = loader.build_samples()
    assert len(samples) == 1 and samples[0][0].shape[2] == 1

    states = torch.randn(2, 30, 1, 11)  # model needs T_h = 30, so use full-length tensors
    batch = collate_fn([
        (states[i], torch.zeros(50, 2), torch.zeros(0, 4),
         torch.zeros(0, dtype=torch.bool), torch.tensor([[4.5, 1.8]]))
        for i in range(2)
    ])
    states, gt_positions, gt_goals, mask, goal_mask, dims = batch
    assert gt_goals.shape == (2, 0, 4)

    traj_dist, pred_goals, risk_field = STRAP(input_dim=11)(states, mask, dims)
    loss, gamma, _, _ = risk_scaled_loss(
        pred_goals, gt_goals, traj_dist, gt_positions,
        ground_truth_risk(gt_positions, gt_goals, goal_mask, dims), goal_mask=goal_mask,
    )
    assert torch.isfinite(traj_dist).all() and torch.isfinite(loss)
    assert gamma == 1.0
    torch.testing.assert_close(risk_field, torch.zeros_like(risk_field))
    next(iter(create_dataloader(samples, batch_size=1)))  # real target-only sample collates
