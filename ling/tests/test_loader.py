import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from src.data import loader as loader_module
from src.data.loader import NGSIMDataLoader


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
        expected_x = target_id * 0.1 + np.arange(frame_start, frame_start + 4) * 0.01
        np.testing.assert_allclose(scene["frames"][:, 0, 0], expected_x)