"""NGSIM data loader — fetches from the Socrata SODA API and caches locally.

API endpoint : https://data.transportation.gov/resource/8ect-6jqj.json
Documentation: https://dev.socrata.com/foundry/data.transportation.gov/8ect-6jqj

Available locations: us-101, i-80, lankershim, peachtree
STRAP-relevant   : us-101, i-80  (freeway datasets)

Units in the raw API response are **feet / ft·s⁻¹ / ft·s⁻²**.
This loader converts everything to **metres / m·s⁻¹ / m·s⁻²** on load.
"""
from src.logger import get_logger
logger = get_logger()

import os
import shutil
import time
import json
import http.client
import urllib.request
import urllib.parse
import urllib.error

import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "https://data.transportation.gov/resource/8ect-6jqj.json"
_FT_TO_M = 0.3048  # 1 foot = 0.3048 metres

from src.risk.s_field import compute_s_field
from src.risk.o_field import compute_o_field, closest_approach
from src.config import (
    ALPHA_X, ALPHA_Y, BETA_1, BETA_2, D_STAR, DATA_DIR, GAMMA_X, GAMMA_Y,
    N_V, RISK_THRESHOLD, T_STAR, TEST_RATIO, TF_STEPS, TH_STEPS, TRAIN_RATIO, VAL_RATIO,
)

# Columns to request from the API (skip arterial-only fields)
_SELECT_COLS = [
    "vehicle_id", "frame_id", "total_frames", "global_time",
    "local_x", "local_y", "global_x", "global_y",
    "v_length", "v_width", "v_class", "v_vel", "v_acc",
    "lane_id", "preceding", "following",
    "space_headway", "time_headway", "location",
]

# Numeric columns that need string→float conversion
_NUMERIC_COLS = [
    "vehicle_id", "frame_id", "total_frames", "global_time",
    "local_x", "local_y", "global_x", "global_y",
    "v_length", "v_width", "v_class", "v_vel", "v_acc",
    "lane_id", "preceding", "following",
    "space_headway", "time_headway",
]

# Columns whose units are in feet (need ×0.3048 to get metres)
_FT_COLS = ["local_x", "local_y", "global_x", "global_y",
            "v_length", "v_width", "space_headway"]
# ft/s → m/s
_FT_PER_S_COLS = ["v_vel"]
# ft/s² → m/s²
_FT_PER_S2_COLS = ["v_acc"]

# Maximum rows the SODA API will return per GET request
_API_PAGE_SIZE = 50_000


# ---------------------------------------------------------------------------
# API fetching helpers
# ---------------------------------------------------------------------------

def _fetch_page(location: str, limit: int, offset: int,
                order: str = "vehicle_id,frame_id",
                app_token: Optional[str] = None) -> list:
    """Fetch one page of JSON records from the SODA API.

    Args:
        location: NGSIM location filter (e.g. 'us-101').
        limit:    Max rows per request (≤50 000).
        offset:   Row offset for pagination.
        order:    $order clause.
        app_token: Optional Socrata app token for higher rate limits.

    Returns:
        List of dicts (one per record).
    """
    params = {
        "$select": ",".join(_SELECT_COLS),
        "$where":  f"location='{location}'",
        "$order":  order,
        "$limit":  str(limit),
        "$offset": str(offset),
    }
    url = f"{_API_BASE}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if app_token:
        req.add_header("X-App-Token", app_token)

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data


def fetch_ngsim(
    location: str = "us-101",
    cache_dir: str = DATA_DIR,
    page_size: int = _API_PAGE_SIZE,
    app_token: Optional[str] = None,
    max_rows: Optional[int] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download NGSIM trajectory data from the SODA API and cache as CSV.

    If a cached CSV already exists for the given location, it is loaded
    directly (unless *force_download* is True).

    Args:
        location:       One of 'us-101', 'i-80', 'lankershim', 'peachtree'.
        cache_dir:      Directory to store cached CSV files.
        page_size:      Rows per API request (max 50 000).
        app_token:      Optional Socrata application token.
        max_rows:       Cap the total number of rows fetched (None = all).
        force_download: Re-download even if a cached file exists.

    Returns:
        pandas DataFrame with numeric columns and metric units.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"ngsim_{location}.csv")
    metadata_path = f"{cache_path}.meta.json"
    mode = "full" if max_rows is None else f"rows_{max_rows}"
    page_cache_dir = os.path.join(
        cache_dir, f".ngsim_{location}_{mode}_p{page_size}.pages"
    )

    # ---- Use cache if available ----
    if os.path.exists(cache_path) and not force_download:
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        if max_rows is None and (
            metadata is None or metadata.get("requested_max_rows") is not None
        ):
            logger.info("[NGSIMLoader] Cached data is not verified as complete; re-downloading")
        else:
            logger.info(f"[NGSIMLoader] Loading cached data from {cache_path}")
            df = pd.read_csv(cache_path)
            if max_rows is not None:
                df = df.head(max_rows)
            return df

    if force_download and os.path.isdir(page_cache_dir):
        shutil.rmtree(page_cache_dir)
    os.makedirs(page_cache_dir, exist_ok=True)

    # ---- Paginated download ----
    logger.info(f"[NGSIMLoader] Downloading '{location}' data from SODA API …")
    offset = 0
    page_num = 0
    total_rows = 0

    while True:
        remaining = None
        if max_rows is not None:
            remaining = max_rows - total_rows
            if remaining <= 0:
                break
        fetch_limit = min(page_size, remaining) if remaining else page_size

        page_num += 1
        page_path = os.path.join(page_cache_dir, f"page_{offset:012d}.csv")
        if os.path.exists(page_path):
            page_df = pd.read_csv(page_path)
            elapsed = 0.0
            logger.info(f"  Resuming from saved page at offset {offset:,}")
        else:
            attempt = 0
            while True:
                t0 = time.time()
                try:
                    page = _fetch_page(
                        location, limit=fetch_limit, offset=offset,
                        app_token=app_token,
                    )
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 429 and not 500 <= exc.code < 600:
                        raise
                    reason = f"HTTP {exc.code}"
                except (
                    urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, json.JSONDecodeError,
                ) as exc:
                    reason = str(exc)

                wait = min(5 * (2 ** min(attempt, 4)), 60)
                logger.warning(
                    f"  Page at offset {offset:,} failed ({reason}); "
                    f"retrying in {wait}s"
                )
                time.sleep(wait)
                attempt += 1

            elapsed = time.time() - t0
            if not page:
                break

            page_df = pd.DataFrame(page)
            for col in _NUMERIC_COLS:
                if col in page_df.columns:
                    page_df[col] = pd.to_numeric(page_df[col], errors="coerce")
            for col in _FT_COLS + _FT_PER_S_COLS + _FT_PER_S2_COLS:
                if col in page_df.columns:
                    page_df[col] = page_df[col] * _FT_TO_M
            page_df.sort_values(["vehicle_id", "frame_id"], inplace=True)

            temp_page_path = f"{page_path}.tmp"
            page_df.to_csv(temp_page_path, index=False)
            os.replace(temp_page_path, page_path)

        page_rows = len(page_df)
        if not page_rows:
            break
        total_rows += page_rows
        logger.info(f"  Page {page_num}: fetched {page_rows:,} rows "
                    f"(total {total_rows:,}) in {elapsed:.1f}s")

        if page_rows < fetch_limit:
            # Last page
            break

        offset += page_rows

        # Polite pause between requests to avoid rate-limiting
        time.sleep(0.5)

    if not total_rows:
        raise RuntimeError(
            f"No records returned for location='{location}'. "
            f"Valid options: us-101, i-80, lankershim, peachtree."
        )

    logger.info(f"[NGSIMLoader] Downloaded {total_rows:,} rows for '{location}'")

    # Assemble the final CSV by streaming saved pages, so another timeout or
    # process restart does not discard already downloaded rows.
    temp_cache_path = f"{cache_path}.tmp"
    with open(temp_cache_path, "wb") as output_file:
        for page_offset in range(0, total_rows, page_size):
            page_path = os.path.join(
                page_cache_dir, f"page_{page_offset:012d}.csv"
            )
            with open(page_path, "rb") as page_file:
                if page_offset:
                    page_file.readline()
                shutil.copyfileobj(page_file, output_file)
    os.replace(temp_cache_path, cache_path)

    df = pd.read_csv(cache_path)
    df.sort_values(["vehicle_id", "frame_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "location": location,
            "requested_max_rows": max_rows,
            "row_count": len(df),
            "complete": max_rows is None,
        }, metadata_file, indent=2)
    shutil.rmtree(page_cache_dir)
    logger.info(f"[NGSIMLoader] Cached to {cache_path}")

    return df


def fetch_ngsim_multi(
    locations: List[str],
    cache_dir: str = DATA_DIR,
    page_size: int = _API_PAGE_SIZE,
    app_token: Optional[str] = None,
    max_rows_per_location: Optional[int] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch NGSIM data for multiple locations and concatenate.

    Each location's ``frame_id`` is offset so that combined data has a
    monotonic, non-overlapping frame timeline — this lets the temporal
    split in :class:`NGSIMDataLoader.get_splits` work correctly across
    locations.

    Args:
        locations:            List of NGSIM locations, e.g. ['us-101', 'i-80'].
        cache_dir:            Directory for cached CSV files.
        page_size:            Rows per API request.
        app_token:            Optional Socrata application token.
        max_rows_per_location: Max rows per location (None = all).
        force_download:       Re-download even if cached.

    Returns:
        Concatenated DataFrame with offset frame_ids.
    """
    frames = []
    frame_offset = 0
    for loc in locations:
        df = fetch_ngsim(
            location=loc, cache_dir=cache_dir, page_size=page_size,
            app_token=app_token, max_rows=max_rows_per_location,
            force_download=force_download,
        )
        # Offset frame_ids to prevent overlap across locations
        df["frame_id"] = df["frame_id"].astype(np.int64) + frame_offset
        frames.append(df)
        frame_offset = int(df["frame_id"].max()) + 1
        logger.info(f"[NGSIMLoader] Concatenated '{loc}': {len(df):,} rows, frame_id offset={frame_offset}")
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"[NGSIMLoader] Combined {len(locations)} locations: {len(combined):,} total rows")
    return combined


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def create_sample(scene: dict, target_idx: int = 0, T_h: int = 30, T_f: int = 50):
    """
    Create a training sample from a synthetic scene.

    Returns:
        states: (T_h, N_v+1, F) state features
        target_traj: (T_f, 2) ground truth future positions
        neighbor_goals: (N_v, 4) ground truth neighbor goals
        mask: (N_v+1,) boolean mask
    """
    num_vehicles = len(scene["vehicle_ids"])
    frames = scene["frames"]

    # Target vehicle history and future
    target_hist = frames[:T_h, target_idx]  # (T_h, 4)
    target_future = frames[T_h:T_h + T_f, target_idx]  # (T_f, 4)
    target_origin = torch.from_numpy(target_hist[-1, :2].copy()).float().unsqueeze(0)

    # Relative coordinates: subtract target position from all vehicles
    all_hist = frames[:T_h]  # (T_h, num_vehicles, 4)
    target_pos = all_hist[:, target_idx:target_idx + 1, :2]  # (T_h, 1, 2)

    # Relative positions
    rel_pos = all_hist[:, :, :2] - target_pos  # (T_h, N, 2)

    # Build state features
    states = []
    for v in range(num_vehicles):
        v_rel_pos = torch.from_numpy(rel_pos[:, v]).float()  # (T_h, 2)
        v_vel = torch.from_numpy(all_hist[:, v, 2:4]).float()  # (T_h, 2)
        v_type_f = float(scene["vehicle_types"][v]) / 2.0
        lane_history = scene.get("lane_history")
        if lane_history is None:
            v_lane = np.full(T_h, scene["lane_ids"][v], dtype=np.float32)
        else:
            v_lane = np.asarray(lane_history[:T_h, v], dtype=np.float32)
        v_acc = all_hist[:, v, 4:5] if all_hist.shape[-1] > 4 else np.zeros((T_h, 1), dtype=np.float32)
        v_length_f = float(scene["vehicle_lengths"][v]) / 10.0
        v_width_f = float(scene["vehicle_widths"][v]) / 3.0

        # Simple state: relative pos + abs vel + type + lane + dimensions
        # Normalize by scene length
        feat = torch.cat([
            v_rel_pos,
            v_vel / 30.0,  # normalize velocity
            torch.from_numpy(v_acc / 10.0).float(),
            torch.full((T_h, 1), v_type_f),
            torch.from_numpy((v_lane / 2.0).reshape(T_h, 1)).float(),
            torch.full((T_h, 1), v_length_f),
            torch.full((T_h, 1), v_width_f),
        ], dim=-1)  # (T_h, 11) including the two risk features
        states.append(feat)

    # Stack: (T_h, N_v+1, F)
    states = torch.stack(states, dim=1)  # (T_h, N_v+1, F)

    # ------ Compute real S-field and O-field risk features ------
    # S-field: spatial proximity risk at each timestep
    rel_pos_t = torch.from_numpy(rel_pos).float()  # (T_h, N, 2)
    delta_x = rel_pos_t[:, :, 0]  # (T_h, N)
    delta_y = rel_pos_t[:, :, 1]  # (T_h, N)
    s_risk = compute_s_field(delta_x, delta_y, GAMMA_X, GAMMA_Y, ALPHA_X, ALPHA_Y)  # (T_h, N)
    s_risk[:, target_idx] = 0.0  # self-risk = 0

    # O-field: future collision risk approximated via velocity-based TTC
    vel_all = torch.from_numpy(all_hist[:, :, 2:4]).float()  # (T_h, N, 2)
    target_vel = vel_all[:, target_idx:target_idx + 1, :]  # (T_h, 1, 2)
    rel_vel = vel_all - target_vel  # (T_h, N, 2)
    min_distance, time_to_closest = closest_approach(rel_pos_t, rel_vel)
    o_risk = compute_o_field(
        min_distance, time_to_closest.clamp(min=0.1),
        D_STAR, T_STAR, BETA_1, BETA_2,
    )
    o_risk[:, target_idx] = 0.0  # self-risk = 0

    risk_features = torch.stack([s_risk, o_risk], dim=-1)  # (T_h, N, 2)
    states = torch.cat([states, risk_features], dim=-1)  # (T_h, N_v+1, F+2)

    # Add batch dimension
    states = states.unsqueeze(0)  # (1, T_h, N_v+1, F)

    # Future labels share a fixed origin at the last observed target position.
    target_traj = torch.tensor(target_future[:, :2], dtype=torch.float32) - target_origin
    target_traj = target_traj.unsqueeze(0)  # (1, T_f, 2)

    # Neighbor goals (position + velocity at end of prediction horizon)
    neighbor_goals = []
    goal_mask = []
    future_goal_mask = scene.get("future_goal_mask")
    for v in range(num_vehicles):
        if v == target_idx:
            continue
        has_future_goal = (
            bool(future_goal_mask[v]) if future_goal_mask is not None
            else np.isfinite(frames[T_h + T_f - 1, v, :4]).all()
        )
        if has_future_goal:
            future_pos = frames[T_h + T_f - 1, v, :2]
            future_vel = frames[T_h + T_f - 1, v, 2:4]
            goal = torch.cat([
                torch.from_numpy(future_pos.copy()).float() - target_origin[0],
                torch.from_numpy(future_vel.copy()).float(),
            ])
        else:
            goal = torch.zeros(4, dtype=torch.float32)
        neighbor_goals.append(goal)
        goal_mask.append(has_future_goal)
    if neighbor_goals:
        neighbor_goals = torch.stack(neighbor_goals).unsqueeze(0)
        goal_mask = torch.tensor(
            [bool(valid) for valid in goal_mask], dtype=torch.bool
        ).unsqueeze(0)
    else:
        neighbor_goals = torch.zeros(1, 0, 4, dtype=torch.float32)
        goal_mask = torch.zeros(1, 0, dtype=torch.bool)

    # Mask: (1, num_vehicles) — target + neighbors present in this sample
    mask = torch.ones(1, num_vehicles, dtype=torch.bool)
    
    # Keep track of vehicle IDs for logging
    vehicle_ids = scene["vehicle_ids"]

    return states, target_traj, neighbor_goals, mask, vehicle_ids, goal_mask, target_origin


# ---------------------------------------------------------------------------
# NGSIMDataLoader
# ---------------------------------------------------------------------------

class NGSIMDataLoader:
    """Load NGSIM data from the Socrata SODA API and produce scene dicts
    compatible with ``create_sample()``.

    Usage::

        loader = NGSIMDataLoader(location="us-101", max_rows=500_000)
        loader.fetch()                    # download (or load cache)
        scenes = loader.build_scenes()    # list of scene dicts
        samples = loader.build_samples()  # list of (states, gt_pos, gt_goals, mask)

    Multi-location usage::

        loader = NGSIMDataLoader(locations=["us-101", "i-80"], max_rows=500_000)
    """

    def __init__(
        self,
        location: str = "us-101",
        data_dir: str = DATA_DIR,
        max_rows: Optional[int] = None,
        app_token: Optional[str] = None,
        window_frames: int = TH_STEPS + TF_STEPS,     # 8 s at 10 Hz
        history_frames: int = TH_STEPS,    # 3 s
        future_frames: int = TF_STEPS,     # 5 s
        stride: int = 10,            # sliding-window stride (1 s)
        max_neighbors: int = N_V,
        min_vehicle_frames: int = TH_STEPS + TF_STEPS,  # skip vehicles with < 8 s of data
        locations: Optional[List[str]] = None,
    ):
        # Support multi-location; 'locations' takes priority over 'location'
        self.locations = locations if locations is not None else [location]
        self.location = self.locations[0]  # primary location for backward compat
        self.data_dir = data_dir
        self.max_rows = max_rows
        self.app_token = app_token
        self.window_frames = window_frames
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.stride = stride
        self.max_neighbors = max_neighbors
        self.min_vehicle_frames = min_vehicle_frames

        self.df: Optional[pd.DataFrame] = None
        self.scenes: List[dict] = []
        self._sample_scene_indices: List[int] = []

    # ------------------------------------------------------------------ #
    #  Step 1: Fetch / load data                                          #
    # ------------------------------------------------------------------ #

    def fetch(self, force_download: bool = False) -> pd.DataFrame:
        """Download data from the SODA API (or load from cache).

        Returns:
            The full DataFrame for the selected location(s).
        """
        if len(self.locations) > 1:
            self.df = fetch_ngsim_multi(
                locations=self.locations,
                cache_dir=self.data_dir,
                app_token=self.app_token,
                max_rows_per_location=self.max_rows,
                force_download=force_download,
            )
        else:
            self.df = fetch_ngsim(
                location=self.locations[0],
                cache_dir=self.data_dir,
                app_token=self.app_token,
                max_rows=self.max_rows,
                force_download=force_download,
            )
        return self.df

    # ------------------------------------------------------------------ #
    #  Step 2: Build scene dicts                                          #
    # ------------------------------------------------------------------ #

    def build_scenes(
        self,
        max_scenes: Optional[int] = None,
    ) -> List[dict]:
        """Build target-vehicle windows using indexed frame and vehicle data."""
        if self.df is None:
            self.fetch()

        df = self.df
        window_length = self.window_frames
        scenes: List[dict] = []
        all_vehicle_ids = np.sort(df["vehicle_id"].unique().astype(int))
        vehicle_data = {}
        eligible_vehicle_ids = []

        for vehicle_id, vehicle_df in df.groupby("vehicle_id", sort=True):
            vehicle_df = vehicle_df.sort_values("frame_id")
            vehicle_id = int(vehicle_id)
            vehicle_data[vehicle_id] = {
                "frames": vehicle_df["frame_id"].to_numpy(dtype=np.int64),
                "local_x": vehicle_df["local_x"].to_numpy(),
                "local_y": vehicle_df["local_y"].to_numpy(),
                "velocity": vehicle_df["v_vel"].to_numpy(),
                "acceleration": vehicle_df["v_acc"].to_numpy(),
                "lane_history": vehicle_df["lane_id"].to_numpy(),
                "type": vehicle_df["v_class"].to_numpy(),
                "length": vehicle_df["v_length"].to_numpy(),
                "width": vehicle_df["v_width"].to_numpy(),
                "lane": vehicle_df["lane_id"].to_numpy(),
            }
            if len(vehicle_df) >= self.min_vehicle_frames:
                eligible_vehicle_ids.append(vehicle_id)

        logger.info(
            f"[NGSIMLoader] {len(eligible_vehicle_ids)} vehicles with "
            f">={self.min_vehicle_frames} frames"
        )

        frame_min = int(df["frame_id"].min())
        frame_max = int(df["frame_id"].max())
        frame_masks = [0] * (frame_max - frame_min + 1)
        frame_data = {}
        for frame_id, frame_df in df.groupby("frame_id", sort=False):
            frame_vehicle_ids = frame_df["vehicle_id"].to_numpy(dtype=np.int64)
            frame_order = np.argsort(frame_vehicle_ids, kind="stable")
            frame_vehicle_ids = frame_vehicle_ids[frame_order]
            longitudinal = frame_df["local_y"].to_numpy()[frame_order]
            lateral = frame_df["local_x"].to_numpy()[frame_order]
            velocity = frame_df["v_vel"].to_numpy()[frame_order]
            vehicle_indices = np.searchsorted(all_vehicle_ids, frame_vehicle_ids)
            frame_mask = 0
            for vehicle_index in np.unique(vehicle_indices):
                frame_mask |= 1 << int(vehicle_index)
            frame_masks[int(frame_id) - frame_min] = frame_mask
            frame_data[int(frame_id)] = (
                frame_vehicle_ids, longitudinal, lateral, velocity
            )

        prefix_masks = [0] * len(frame_masks)
        suffix_masks = [0] * len(frame_masks)
        all_vehicles_mask = (1 << len(all_vehicle_ids)) - 1
        history_length = self.history_frames
        for block_start in range(0, len(frame_masks), history_length):
            block_end = min(block_start + history_length, len(frame_masks))
            running_mask = all_vehicles_mask
            for frame_index in range(block_start, block_end):
                running_mask &= frame_masks[frame_index]
                prefix_masks[frame_index] = running_mask
            running_mask = all_vehicles_mask
            for frame_index in range(block_end - 1, block_start - 1, -1):
                running_mask &= frame_masks[frame_index]
                suffix_masks[frame_index] = running_mask

        window_cache = {}
        for target_vehicle_id in eligible_vehicle_ids:
            target_data = vehicle_data[target_vehicle_id]
            target_frames = target_data["frames"]
            for start_position in range(
                0, len(target_frames) - window_length + 1, self.stride
            ):
                window_frame_ids = target_frames[
                    start_position:start_position + window_length
                ]
                if window_frame_ids[-1] - window_frame_ids[0] != window_length - 1:
                    continue

                frame_start = int(window_frame_ids[0])
                cache_entry = window_cache.get(frame_start)
                if cache_entry is None:
                    start_index = frame_start - frame_min
                    end_index = start_index + history_length - 1
                    full_window_mask = suffix_masks[start_index] & prefix_masks[end_index]
                    present_indices = []
                    while full_window_mask:
                        lowest_bit = full_window_mask & -full_window_mask
                        present_indices.append(lowest_bit.bit_length() - 1)
                        full_window_mask ^= lowest_bit
                    present_vehicle_ids = all_vehicle_ids[present_indices]
                    midpoint_frame = frame_start + self.history_frames - 1
                    midpoint_data = frame_data.get(midpoint_frame)
                    cache_entry = (present_vehicle_ids, midpoint_data)
                    window_cache[frame_start] = cache_entry

                present_vehicle_ids, midpoint_data = cache_entry
                if midpoint_data is None or target_vehicle_id not in present_vehicle_ids:
                    continue

                neighbor_ids = present_vehicle_ids[
                    present_vehicle_ids != target_vehicle_id
                ]
                midpoint_vehicle_ids, midpoint_longitudinal, midpoint_lateral, midpoint_velocity = midpoint_data
                target_position = np.searchsorted(midpoint_vehicle_ids, target_vehicle_id)
                if len(neighbor_ids):
                    midpoint_positions = np.searchsorted(midpoint_vehicle_ids, neighbor_ids)
                    delta_long = (
                        midpoint_longitudinal[midpoint_positions]
                        - midpoint_longitudinal[target_position]
                    )
                    delta_lat = (
                        midpoint_lateral[midpoint_positions]
                        - midpoint_lateral[target_position]
                    )
                    risk_s = np.exp(-(
                        np.abs(delta_long / GAMMA_X) ** ALPHA_X
                        + np.abs(delta_lat / GAMMA_Y) ** ALPHA_Y
                    ))
                    relative_velocity = (
                        midpoint_velocity[midpoint_positions]
                        - midpoint_velocity[target_position]
                    )
                    time_to_closest = np.maximum(
                        -(delta_long * relative_velocity)
                        / np.maximum(relative_velocity ** 2, 1e-12),
                        0.0,
                    )
                    min_distance = np.sqrt(
                        (delta_long + time_to_closest * relative_velocity) ** 2
                        + delta_lat ** 2
                    )
                    risk_o = np.exp(-((min_distance / D_STAR) ** BETA_1)) * np.exp(
                        -((time_to_closest / T_STAR) ** BETA_2)
                    )
                    combined_risk = risk_s + risk_o
                    interacting = (risk_s > RISK_THRESHOLD) | (risk_o > RISK_THRESHOLD)
                    if np.any(interacting):
                        ranked_indices = np.argsort(-combined_risk, kind="stable")
                        ranked_indices = ranked_indices[interacting[ranked_indices]]
                    else:
                        geometric_distance = np.sqrt(delta_long ** 2 + delta_lat ** 2)
                        ranked_indices = np.argsort(geometric_distance, kind="stable")
                    selected_ids = neighbor_ids[
                        ranked_indices[:self.max_neighbors]
                    ].tolist()
                else:
                    selected_ids = []
                scene_vehicle_ids = [target_vehicle_id] + selected_ids
                num_scene_vehicles = len(scene_vehicle_ids)

                frames_array = np.zeros(
                    (window_length, num_scene_vehicles, 5), dtype=np.float32
                )
                frames_array[:] = np.nan
                vehicle_types = np.zeros(num_scene_vehicles, dtype=np.float32)
                vehicle_lengths = np.zeros(num_scene_vehicles, dtype=np.float32)
                vehicle_widths = np.zeros(num_scene_vehicles, dtype=np.float32)
                lane_ids = np.zeros(num_scene_vehicles, dtype=np.float32)
                lane_history = np.full(
                    (window_length, num_scene_vehicles), np.nan, dtype=np.float32
                )
                future_goal_mask = np.ones(num_scene_vehicles, dtype=bool)
                frame_ids_window = np.arange(frame_start, frame_start + window_length)
                for scene_index, vehicle_id in enumerate(scene_vehicle_ids):
                    data = vehicle_data[vehicle_id]
                    row_start = np.searchsorted(data["frames"], frame_start)
                    row_indices = np.minimum(
                        row_start + np.arange(window_length), len(data["frames"]) - 1
                    )
                    valid_rows = data["frames"][row_indices] == frame_ids_window
                    valid_positions = np.flatnonzero(valid_rows)
                    valid_row_indices = row_indices[valid_rows]
                    frames_array[valid_positions, scene_index, 0] = data["local_y"][valid_row_indices]
                    frames_array[valid_positions, scene_index, 1] = data["local_x"][valid_row_indices]
                    frames_array[valid_positions, scene_index, 2] = data["velocity"][valid_row_indices]
                    frames_array[valid_positions, scene_index, 3] = 0.0
                    frames_array[valid_positions, scene_index, 4] = data["acceleration"][valid_row_indices]
                    lane_history[valid_positions, scene_index] = data["lane_history"][valid_row_indices]
                    vehicle_types[scene_index] = data["type"][row_start]
                    vehicle_lengths[scene_index] = data["length"][row_start]
                    vehicle_widths[scene_index] = data["width"][row_start]
                    lane_ids[scene_index] = data["lane_history"][row_start]
                    if scene_index > 0 and not valid_rows[-1]:
                        future_goal_mask[scene_index] = False

                scenes.append({
                    "vehicle_ids": np.asarray(scene_vehicle_ids),
                    "frames": frames_array,
                    "vehicle_types": vehicle_types,
                    "vehicle_lengths": vehicle_lengths,
                    "vehicle_widths": vehicle_widths,
                    "lane_ids": lane_ids,
                    "lane_history": lane_history,
                    "future_goal_mask": future_goal_mask,
                    "frame_id_start": frame_start,
                })

                if max_scenes is not None and len(scenes) >= max_scenes:
                    break

            if max_scenes is not None and len(scenes) >= max_scenes:
                break

        self.scenes = scenes
        logger.info(f"[NGSIMLoader] Built {len(scenes)} scenes")
        return scenes

    # ------------------------------------------------------------------ #
    #  Step 3: Build training samples                                     #
    # ------------------------------------------------------------------ #

    def build_samples(
        self,
        max_samples: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert scenes to training-ready (states, gt_pos, gt_goals, mask)
        tuples, built with ``create_sample()``.

        Each returned tuple has:
            states      : (1, T_h, N, F)   float32
            gt_pos      : (1, T_f, 2)      float32
            gt_goals    : (1, N-1, 4)       float32
            mask        : (1, N)            bool
        """
        if not self.scenes:
            self.build_scenes(max_scenes=max_samples)

        samples = []
        self._sample_scene_indices = []
        for scene_index, scene in enumerate(self.scenes):
            try:
                sample = create_sample(
                    scene,
                    target_idx=0,
                    T_h=self.history_frames,
                    T_f=self.future_frames,
                )
                if (
                    not torch.isfinite(sample[0][:, :, 0]).all()
                    or not torch.isfinite(sample[1]).all()
                    or not torch.isfinite(sample[2][sample[5]]).all()
                ):
                    continue
                samples.append(sample)
                self._sample_scene_indices.append(scene_index)
            except Exception as exc:
                # Some scenes may fail (e.g. NaN in data) — skip them
                continue

            if max_samples is not None and len(samples) >= max_samples:
                break

        logger.info(f"[NGSIMLoader] Built {len(samples)} training samples")
        return samples

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def get_dataset_stats(self) -> dict:
        """Return basic dataset statistics."""
        stats = {"locations": self.locations, "num_scenes": len(self.scenes)}
        if self.df is not None:
            stats.update({
                "total_rows": len(self.df),
                "unique_vehicles": int(self.df["vehicle_id"].nunique()),
                "num_frames": int(self.df["frame_id"].nunique()),
                "frame_range": (
                    int(self.df["frame_id"].min()),
                    int(self.df["frame_id"].max()),
                ),
                "vehicle_classes": self.df["v_class"].value_counts().to_dict(),
            })
        return stats

    def get_splits(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
    ) -> Tuple[list, list, list]:
        """Split samples into train / val / test sets using TEMPORAL splitting.

        Time boundaries are chosen on raw frames. A full-window gap is left
        between splits so no source frame can occur in more than one split.

        Returns:
            (train_samples, val_samples, test_samples)
        """
        samples = self.build_samples() if not hasattr(self, "_samples_cache") else self._samples_cache
        self._samples_cache = samples

        if not samples:
            logger.info("[NGSIMLoader] No samples to split.")
            return [], [], []

        # Each sample also carries goal-validity, vehicle IDs, and target origin.
        # The corresponding scene's frame_id_start is stored in self.scenes
        # Pair each sample with its scene's start frame for sorting
        scene_starts = []
        sample_scene_indices = getattr(self, "_sample_scene_indices", [])
        if len(sample_scene_indices) != len(samples):
            sample_scene_indices = list(range(min(len(samples), len(self.scenes))))
            self._sample_scene_indices = sample_scene_indices
        scene_starts = [
            int(self.scenes[scene_index].get("frame_id_start", 0))
            for scene_index in sample_scene_indices
        ]

        n = min(len(samples), len(scene_starts))
        if (
            not 0 < train_ratio < 1
            or not 0 < val_ratio < 1
            or not 0 < test_ratio < 1
            or not np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
        ):
            raise ValueError("train_ratio, val_ratio, and test_ratio must be positive and sum to 1")

        if self.df is not None and "frame_id" in self.df:
            timeline = np.sort(self.df["frame_id"].unique())
        else:
            timeline = np.sort(np.unique(scene_starts))
        if len(timeline) < 3:
            raise ValueError("Not enough unique frames to create temporal splits")

        train_cut = int(timeline[min(int(len(timeline) * train_ratio), len(timeline) - 1)])
        validation_cut = int(
            timeline[min(int(len(timeline) * (train_ratio + val_ratio)), len(timeline) - 1)]
        )
        frame_ends = [start + self.window_frames - 1 for start in scene_starts[:n]]
        train_sample_idx = [i for i in range(n) if frame_ends[i] < train_cut]
        val_sample_idx = [
            i for i in range(n)
            if scene_starts[i] >= train_cut + self.window_frames
            and frame_ends[i] < validation_cut
        ]
        test_sample_idx = [
            i for i in range(n)
            if scene_starts[i] >= validation_cut + self.window_frames
            and frame_ends[i] <= int(timeline[-1])
        ]

        train = [samples[i] for i in train_sample_idx]
        val = [samples[i] for i in val_sample_idx]
        test = [samples[i] for i in test_sample_idx]

        # Log the temporal boundaries
        if train_sample_idx:
            train_frames = [scene_starts[i] for i in train_sample_idx]
            val_frames = [scene_starts[i] for i in val_sample_idx] if val_sample_idx else []
            test_frames = [scene_starts[i] for i in test_sample_idx] if test_sample_idx else []
            logger.info(f"[NGSIMLoader] Temporal split: train={len(train)} "
                  f"(frames {min(train_frames)}-{max(train_frames)}), "
                  f"val={len(val)} "
                  f"(frames {min(val_frames) if val_frames else 'N/A'}-{max(val_frames) if val_frames else 'N/A'}), "
                  f"test={len(test)} "
                  f"(frames {min(test_frames) if test_frames else 'N/A'}-{max(test_frames) if test_frames else 'N/A'})")
        else:
            logger.info(f"[NGSIMLoader] Split: train={len(train)}, "
                  f"val={len(val)}, test={len(test)}")

        # Store indices so callers can access corresponding scenes
        self._split_indices = {
            'train': [sample_scene_indices[i] for i in train_sample_idx],
            'val': [sample_scene_indices[i] for i in val_sample_idx],
            'test': [sample_scene_indices[i] for i in test_sample_idx],
        }
        return train, val, test


# ---------------------------------------------------------------------------
# TrafficDataset & DataLoader helpers
# ---------------------------------------------------------------------------

class TrafficDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for traffic trajectory data."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        states, gt_pos, gt_goals, mask, vehicle_ids, goal_mask, target_origin = self.samples[idx]
        # Remove the batch dim that create_sample adds (DataLoader re-batches)
        return states.squeeze(0), gt_pos.squeeze(0), gt_goals.squeeze(0), goal_mask.squeeze(0)


def create_dataloader(
    samples: list,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader from a list of (states, gt_pos, gt_goals, mask)."""
    dataset = TrafficDataset(samples)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """Custom collate function for variable-sized scenes.

    Each item in *batch* is ``(states, gt_pos, gt_goals, goal_mask)`` **without** a
    leading batch dimension (TrafficDataset squeezes it).

    Because different scenes may have different numbers of vehicles, we
    pad to the maximum N in the batch and return a mask.
    """
    states_list, gt_pos_list, gt_goals_list, goal_masks = zip(*batch)

    # gt_pos is always (T_f, 2) — stack directly
    gt_positions = torch.stack(gt_pos_list)  # (B, T_f, 2)

    # states: (T_h, N_i, F) — N varies across samples → pad
    T_h = states_list[0].shape[0]
    F = states_list[0].shape[2]
    max_N = max(s.shape[1] for s in states_list)

    B = len(states_list)
    states_padded = torch.zeros(B, T_h, max_N, F)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    
    for i, s in enumerate(states_list):
        n = s.shape[1]
        states_padded[i, :, :n, :] = s
        mask[i, :n] = True

    # gt_goals: (N_i-1, 4) — pad similarly
    max_Nv = max_N - 1
    gt_goals_padded = torch.zeros(B, max_Nv, 4)
    goal_mask_padded = torch.zeros(B, max_Nv, dtype=torch.bool)
    for i, (g, goal_mask) in enumerate(zip(gt_goals_list, goal_masks)):
        n = g.shape[0]
        gt_goals_padded[i, :n, :] = g
        goal_mask_padded[i, :n] = goal_mask

    return states_padded, gt_positions, gt_goals_padded, mask, goal_mask_padded

