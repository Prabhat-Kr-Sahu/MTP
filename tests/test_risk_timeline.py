"""Risk-over-time windows: one row per stride-1 window, true geometry, risk peaks at t*."""
import numpy as np
import pandas as pd
import torch

from experiments.plot_risk_timeline import (
    AFTER, BEFORE, closest_frame, score_windows, true_distances, window_scenes,
)
from src.config import TF_STEPS, TH_STEPS
from tests.test_risk_scoring import _ZeroPredictor


def test_timeline_follows_a_closing_pair():
    # Target (1) at 10 m/s closes on a leader (2) at 8 m/s, 0.5 m offset:
    # gap = 30 - 2 t, so the closest approach is frame 150.
    rows = [{"vehicle_id": v, "frame_id": f,
             "local_x": 3.6 + 0.5 * (v == 2),
             "local_y": (1.0 * f) if v == 1 else 30.0 + 0.8 * f,
             "v_vel": 10.0 if v == 1 else 8.0, "v_acc": 0.0, "v_class": 2,
             "v_length": 4.5, "v_width": 1.8, "lane_id": 1}
            for v in (1, 2) for f in range(320)]
    df = pd.DataFrame(rows)
    t_star, min_dist = closest_frame(df, 1, 2)
    assert t_star == 150 and np.isclose(min_dist, 0.5)

    loader, scenes = window_scenes(df, 1, t_star, dict(
        window_frames=TH_STEPS + TF_STEPS, history_frames=TH_STEPS, future_frames=TF_STEPS))
    now = [int(s["frame_id_start"]) + TH_STEPS - 1 for s in scenes]
    assert now == list(range(t_star - BEFORE, t_star + AFTER + 1))   # every 0.1 s

    out = score_windows(_ZeroPredictor(), None, loader, scenes, torch.device("cpu"))
    out = out[out.neighbour == 2].set_index("now_frame")
    gap = np.hypot(30.0 - 0.2 * out.index.to_numpy(), 0.5)
    assert np.allclose(out.dist_now, gap, atol=1e-4)
    truth = true_distances(loader, 1, [2], t_star).set_index("now_frame").dist
    assert np.allclose(truth.loc[out.index], gap, atol=1e-4)
    # Zero predictor: both stay put, so risk is highest when they are closest.
    assert abs(out.p_coll_gauss.idxmax() - t_star) <= 10
    assert out.p_coll_gauss.between(0, 1).all()
