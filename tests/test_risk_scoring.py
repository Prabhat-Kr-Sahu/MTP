"""Batched risk scoring: equivalence, the four neighbour sources, ragged counts."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import TF_STEPS, TH_STEPS
from src.data.loader import NGSIMDataLoader
from src.evaluation.risk_scoring import NEIGHBOUR_SOURCES, _fill_gaps, score_split


class _ZeroPredictor(torch.nn.Module):
    """Predicts the target stays at its origin, with unit isotropic spread.

    A zero mean makes the pred-source origin shift directly observable: every
    neighbour prediction must land exactly on its observed relative position.
    """

    def forward(self, states, mask=None, dims=None):
        batch = states.shape[0]
        n_neighbours = states.shape[2] - 1
        traj = torch.zeros(batch, TF_STEPS, 5)
        traj[..., 2:4] = 1.0          # sigma
        return traj, torch.zeros(batch, max(n_neighbours, 0), 4), None


def _loader(n_vehicles=4, n_frames=900, missing=()):
    """A loader over synthetic straight-line traffic, with real window sizes.

    900 frames is not arbitrary: get_splits leaves a full-window guard gap
    after the validation cut, so the test split needs window starts beyond
    0.8 * n_frames + window_frames, which a short recording never reaches.

    Vehicles sit 12 m apart across two lanes so the bumper-to-bumper gaps put
    pair_risk above RISK_THRESHOLD -- otherwise build_scenes selects no
    neighbours at all and the scene-based sources have nothing to score.

    `missing` lists (vehicle_id, frame_id) rows to drop, which is how a
    neighbour's future acquires the gaps the oracle source has to fill.
    """
    rows = []
    for vehicle_id in range(1, n_vehicles + 1):
        lane = 1 + vehicle_id % 2
        for frame_id in range(n_frames):
            if (vehicle_id, frame_id) in missing:
                continue
            rows.append({
                "vehicle_id": vehicle_id,
                "frame_id": frame_id,
                "local_x": 3.6 * lane,
                "local_y": 12.0 * vehicle_id + frame_id * 0.8,   # 8 m/s
                "v_vel": 8.0,
                "v_acc": 0.0,
                "v_class": 2,
                "v_length": 4.5,
                "v_width": 1.8,
                "lane_id": lane,
            })
    loader = NGSIMDataLoader(
        window_frames=TH_STEPS + TF_STEPS, history_frames=TH_STEPS,
        future_frames=TF_STEPS, stride=10, max_neighbors=15,
        min_vehicle_frames=TH_STEPS + TF_STEPS,
    )
    loader.df = pd.DataFrame(rows)
    return loader


def _scored(source="cv", batch_size=8, loader=None, **kwargs):
    loader = loader or _loader()
    _, _, test_samples = loader.get_splits()
    if not test_samples:
        pytest.skip("synthetic split produced no test samples")
    return score_split(
        _ZeroPredictor(), loader, test_samples, norm_stats=None, threshold=2.0,
        source=source, batch_size=batch_size, device=torch.device("cpu"),
        progress_every=0, **kwargs
    ), loader, test_samples


def test_every_source_scores_every_registered_metric():
    from src.evaluation.risk_metrics import METRICS
    for source in NEIGHBOUR_SOURCES:
        result, _, _ = _scored(source=source)
        assert set(result.scores) == set(METRICS), source
        assert result.n_pairs > 0, source
        for name, values in result.scores.items():
            assert values.shape == (result.n_pairs,), (source, name)
            assert not np.isnan(values).any(), (source, name)


@pytest.mark.parametrize("source", ["cv", "cv_scene", "oracle"])
def test_batched_matches_unbatched(source):
    """Batch size must not change a single score, label or pair ordering.

    This is what makes the rewrite of the per-sample loop safe.
    """
    one, _, _ = _scored(source=source, batch_size=1)
    seven, _, _ = _scored(source=source, batch_size=7)
    big, _, _ = _scored(source=source, batch_size=256)
    assert one.n_pairs == seven.n_pairs == big.n_pairs
    for reference in (seven, big):
        np.testing.assert_array_equal(one.gt, reference.gt)
        np.testing.assert_array_equal(one.pairs, reference.pairs)
        np.testing.assert_allclose(one.gt_distance, reference.gt_distance, atol=1e-5)
        for name in one.scores:
            np.testing.assert_allclose(
                np.nan_to_num(one.scores[name], posinf=1e30, neginf=-1e30),
                np.nan_to_num(reference.scores[name], posinf=1e30, neginf=-1e30),
                atol=1e-5, rtol=1e-5, err_msg=f"{source}/{name}",
            )


def test_pred_source_shifts_predictions_into_the_target_frame():
    """With a zero-mean predictor every neighbour sits at its observed relative
    position, so min_dist must equal the norm of that position. This pins the
    one coordinate transform that would otherwise be silently wrong."""
    result, loader, test_samples = _scored(source="pred", batch_size=4)
    expected = []
    for sample_index in range(len(test_samples)):
        scene = loader.scenes[loader._split_indices["test"][sample_index]]
        frames = scene["frames"]
        columns = np.arange(1, frames.shape[1])
        origin = frames[TH_STEPS - 1, 0, :2]
        position = frames[TH_STEPS - 1, columns, :2] - origin
        future = frames[TH_STEPS:TH_STEPS + TF_STEPS, columns, :2] - origin
        keep = np.isfinite(position[:, 0]) & np.isfinite(future[..., 0]).any(axis=0)
        expected.extend(np.linalg.norm(position[keep], axis=-1).tolist())
    np.testing.assert_allclose(result.scores["min_dist"], expected, atol=1e-4)


def test_cv_scene_and_pred_share_a_pair_set():
    """The matched control: their difference isolates the predictor's
    contribution over kinematic extrapolation, which only means anything if
    the pairs are identical."""
    control, _, _ = _scored(source="cv_scene")
    predicted, _, _ = _scored(source="pred")
    np.testing.assert_array_equal(control.pairs, predicted.pairs)
    np.testing.assert_array_equal(control.gt, predicted.gt)


def test_cv_and_oracle_share_a_pair_set():
    cv, _, _ = _scored(source="cv")
    oracle, _, _ = _scored(source="oracle")
    np.testing.assert_array_equal(cv.pairs, oracle.pairs)
    np.testing.assert_array_equal(cv.gt, oracle.gt)


def test_oracle_reports_and_fills_absent_future_steps():
    gaps = {(3, frame) for frame in range(150, 200)}
    result, _, _ = _scored(source="oracle", loader=_loader(missing=gaps))
    for name, values in result.scores.items():
        assert not np.isnan(values).any(), name


def test_fill_gaps_carries_the_last_seen_position():
    future = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[np.nan, np.nan]]])
    filled, n_filled = _fill_gaps(future)
    assert n_filled == 1
    np.testing.assert_allclose(filled[2, 0], [3.0, 4.0])


def test_fill_gaps_backfills_a_leading_gap():
    future = np.array([[[np.nan, np.nan]], [[3.0, 4.0]]])
    filled, n_filled = _fill_gaps(future)
    assert n_filled == 1
    np.testing.assert_allclose(filled[0, 0], [3.0, 4.0])


def test_ground_truth_is_independent_of_the_model():
    """Labels come from true futures only, so two different predictors must
    produce identical labels, distances and times."""
    loader = _loader()
    _, _, test_samples = loader.get_splits()
    if not test_samples:
        pytest.skip("synthetic split produced no test samples")

    class _Offset(_ZeroPredictor):
        def forward(self, states, mask=None, dims=None):
            traj, goals, extra = super().forward(states, mask, dims)
            traj[..., :2] += 7.0
            return traj, goals, extra

    common = dict(loader=loader, test_samples=test_samples, norm_stats=None,
                  threshold=2.0, source="cv", batch_size=8,
                  device=torch.device("cpu"), progress_every=0)
    a = score_split(_ZeroPredictor(), **common)
    b = score_split(_Offset(), **common)
    np.testing.assert_array_equal(a.gt, b.gt)
    np.testing.assert_allclose(a.gt_distance, b.gt_distance)
    np.testing.assert_allclose(a.gt_time, b.gt_time)
    assert not np.allclose(a.scores["min_dist"], b.scores["min_dist"])


def test_prefix_scores_have_one_column_per_horizon():
    from src.evaluation.risk_metrics import EARLINESS_METRICS
    result, _, _ = _scored(source="cv", prefixes=EARLINESS_METRICS)
    for name in EARLINESS_METRICS:
        assert result.prefix_scores[name].shape == (result.n_pairs, 5), name


def test_sample_stride_reduces_the_work():
    full, _, _ = _scored(source="cv")
    strided, _, _ = _scored(source="cv", sample_stride=3)
    assert strided.n_samples < full.n_samples
    assert strided.n_pairs < full.n_pairs


def test_unknown_source_is_rejected():
    loader = _loader()
    _, _, test_samples = loader.get_splits()
    with pytest.raises(ValueError, match="source must be one of"):
        score_split(_ZeroPredictor(), loader, test_samples, None, source="nope")
