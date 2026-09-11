"""NGSIM data loader — fetches from the Socrata SODA API and caches locally.

API endpoint : https://data.transportation.gov/resource/8ect-6jqj.json
Documentation: https://dev.socrata.com/foundry/data.transportation.gov/8ect-6jqj

Available locations: us-101, i-80, lankershim, peachtree
STRAP-relevant   : us-101, i-80  (freeway datasets)

Units in the raw API response are **feet / ft·s⁻¹ / ft·s⁻²**.
This loader converts everything to **metres / m·s⁻¹ / m·s⁻²** on load.
"""

import os
import time
import json
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
    cache_dir: str = "data/raw/ngsim",
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

    # ---- Use cache if available ----
    if os.path.exists(cache_path) and not force_download:
        print(f"[NGSIMLoader] Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        if max_rows is not None:
            df = df.head(max_rows)
        return df

    # ---- Paginated download ----
    print(f"[NGSIMLoader] Downloading '{location}' data from SODA API …")
    all_records: list = []
    offset = 0
    page_num = 0

    while True:
        remaining = None
        if max_rows is not None:
            remaining = max_rows - len(all_records)
            if remaining <= 0:
                break
        fetch_limit = min(page_size, remaining) if remaining else page_size

        page_num += 1
        t0 = time.time()
        try:
            page = _fetch_page(location, limit=fetch_limit, offset=offset,
                               app_token=app_token)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Rate-limited — back off and retry
                wait = 10
                print(f"  Rate-limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            raise

        elapsed = time.time() - t0
        if not page:
            break

        all_records.extend(page)
        print(f"  Page {page_num}: fetched {len(page):,} rows "
              f"(total {len(all_records):,}) in {elapsed:.1f}s")

        if len(page) < fetch_limit:
            # Last page
            break

        offset += len(page)

        # Polite pause between requests to avoid rate-limiting
        time.sleep(0.5)

    if not all_records:
        raise RuntimeError(
            f"No records returned for location='{location}'. "
            f"Valid options: us-101, i-80, lankershim, peachtree."
        )

    print(f"[NGSIMLoader] Downloaded {len(all_records):,} rows for '{location}'")

    # ---- Build DataFrame ----
    df = pd.DataFrame(all_records)

    # Convert numeric columns from strings
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Unit conversion: feet → metres ----
    for col in _FT_COLS:
        if col in df.columns:
            df[col] = df[col] * _FT_TO_M
    for col in _FT_PER_S_COLS:
        if col in df.columns:
            df[col] = df[col] * _FT_TO_M
    for col in _FT_PER_S2_COLS:
        if col in df.columns:
            df[col] = df[col] * _FT_TO_M

    # Sort by vehicle then frame
    df.sort_values(["vehicle_id", "frame_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- Cache to disk ----
    df.to_csv(cache_path, index=False)
    print(f"[NGSIMLoader] Cached to {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Scene construction helpers
# ---------------------------------------------------------------------------

def _build_frame_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray,
                                                     np.ndarray]:
    """Convert a DataFrame subset into aligned numpy arrays indexed by
    (frame, vehicle).

    Returns:
        frame_ids    : sorted unique frame IDs
        vehicle_ids  : sorted unique vehicle IDs
        positions    : (n_frames, n_vehicles, 2) — local_x, local_y (metres)
        velocities   : (n_frames, n_vehicles, 2) — v_vel duplicated as (vx, vy)
        vehicle_types: (n_vehicles,)
        vehicle_lengths: (n_vehicles,)  (metres)
        vehicle_widths : (n_vehicles,)  (metres)
        lane_ids     : (n_vehicles,) — last-seen lane per vehicle
        presence     : (n_frames, n_vehicles) bool — True if vehicle present
    """
    frame_ids = np.sort(df["frame_id"].unique())
    vehicle_ids = np.sort(df["vehicle_id"].unique().astype(int))

    n_frames = len(frame_ids)
    n_vehicles = len(vehicle_ids)

    frame_to_idx = {int(f): i for i, f in enumerate(frame_ids)}
    veh_to_idx = {int(v): i for i, v in enumerate(vehicle_ids)}

    # Pre-allocate arrays (NaN = missing)
    positions = np.full((n_frames, n_vehicles, 2), np.nan)
    vel_arr = np.full((n_frames, n_vehicles), np.nan)
    acc_arr = np.full((n_frames, n_vehicles), np.nan)
    presence = np.zeros((n_frames, n_vehicles), dtype=bool)

    # Per-vehicle static attributes (take first occurrence)
    v_types = np.zeros(n_vehicles)
    v_lengths = np.zeros(n_vehicles)
    v_widths = np.zeros(n_vehicles)
    v_lanes = np.zeros(n_vehicles)

    # Group by vehicle for efficiency
    for vid, grp in df.groupby("vehicle_id"):
        vi = veh_to_idx[int(vid)]
        v_types[vi] = grp["v_class"].iloc[0]
        v_lengths[vi] = grp["v_length"].iloc[0]
        v_widths[vi] = grp["v_width"].iloc[0]
        v_lanes[vi] = grp["lane_id"].iloc[-1]

        for _, row in grp.iterrows():
            fi = frame_to_idx[int(row["frame_id"])]
            positions[fi, vi, 0] = row["local_x"]
            positions[fi, vi, 1] = row["local_y"]
            vel_arr[fi, vi] = row["v_vel"]
            acc_arr[fi, vi] = row["v_acc"]
            presence[fi, vi] = True

    # Build a (n_frames, n_vehicles, 4) array matching synthetic format:
    # [x, y, vx, vy].  NGSIM provides scalar velocity; approximate vx≈0
    # (lateral speed is small on freeways) and vy=v_vel (longitudinal).
    frames_data = np.zeros((n_frames, n_vehicles, 4))
    frames_data[:, :, 0] = positions[:, :, 0]  # local_x (lateral)
    frames_data[:, :, 1] = positions[:, :, 1]  # local_y (longitudinal)
    frames_data[:, :, 2] = 0.0                 # vx ≈ 0 on freeway
    frames_data[:, :, 3] = np.nan_to_num(vel_arr, nan=0.0)  # vy = v_vel

    return (frame_ids, vehicle_ids, frames_data, v_types,
            v_lengths, v_widths, v_lanes, presence)


# ---------------------------------------------------------------------------
# NGSIMDataLoader
# ---------------------------------------------------------------------------

class NGSIMDataLoader:
    """Load NGSIM data from the Socrata SODA API and produce scene dicts
    compatible with ``create_sample()`` from ``generate_synthetic.py``.

    Usage::

        loader = NGSIMDataLoader(location="us-101", max_rows=500_000)
        loader.fetch()                    # download (or load cache)
        scenes = loader.build_scenes()    # list of scene dicts
        samples = loader.build_samples()  # list of (states, gt_pos, gt_goals, mask)
    """

    def __init__(
        self,
        location: str = "us-101",
        data_dir: str = "data/raw/ngsim",
        max_rows: Optional[int] = None,
        app_token: Optional[str] = None,
        window_frames: int = 80,     # 8 s at 10 Hz
        history_frames: int = 30,    # 3 s
        future_frames: int = 50,     # 5 s
        stride: int = 10,            # sliding-window stride (1 s)
        max_neighbors: int = 15,
        min_vehicle_frames: int = 80,  # skip vehicles with < 8 s of data
    ):
        self.location = location
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

    # ------------------------------------------------------------------ #
    #  Step 1: Fetch / load data                                          #
    # ------------------------------------------------------------------ #

    def fetch(self, force_download: bool = False) -> pd.DataFrame:
        """Download data from the SODA API (or load from cache).

        Returns:
            The full DataFrame for the selected location.
        """
        self.df = fetch_ngsim(
            location=self.location,
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
        """Create scene dicts from the loaded DataFrame.

        Each scene is an 8-second contiguous window centred on a
        *target vehicle*, containing up to ``max_neighbors`` nearby
        vehicles that are present throughout the window.

        The returned dict has the **same schema** as
        ``generate_synthetic_trajectory()``::

            {
                "vehicle_ids":    np.ndarray (N,),
                "frames":         np.ndarray (window_frames, N, 4),
                "vehicle_types":  np.ndarray (N,),
                "vehicle_lengths":np.ndarray (N,),
                "vehicle_widths": np.ndarray (N,),
                "lane_ids":       np.ndarray (N,),
            }

        Where the **first vehicle (index 0) is always the target**.
        """
        if self.df is None:
            self.fetch()

        df = self.df
        T = self.window_frames  # total frames in a scene (80)

        scenes: List[dict] = []

        # Group by vehicle to find candidates with enough frames
        vehicle_groups = df.groupby("vehicle_id")
        eligible_vids = [
            int(vid) for vid, grp in vehicle_groups
            if len(grp) >= self.min_vehicle_frames
        ]
        print(f"[NGSIMLoader] {len(eligible_vids)} vehicles with "
              f"≥{self.min_vehicle_frames} frames")

        for target_vid in eligible_vids:
            target_df = df[df["vehicle_id"] == target_vid].sort_values("frame_id")
            target_frames = target_df["frame_id"].values.astype(int)

            # Sliding windows over this vehicle's lifetime
            for start_pos in range(0, len(target_frames) - T + 1, self.stride):
                window_fids = target_frames[start_pos: start_pos + T]

                # Must be contiguous (consecutive frame IDs)
                if window_fids[-1] - window_fids[0] != T - 1:
                    continue

                fid_start, fid_end = int(window_fids[0]), int(window_fids[-1])

                # Find all vehicles present in this frame range
                window_df = df[
                    (df["frame_id"] >= fid_start) &
                    (df["frame_id"] <= fid_end)
                ]

                # Keep only vehicles present in ALL frames of the window
                veh_counts = window_df.groupby("vehicle_id")["frame_id"].nunique()
                fully_present = veh_counts[veh_counts == T].index.astype(int).tolist()

                if target_vid not in fully_present:
                    continue  # shouldn't happen, but guard

                # Select neighbours closest to target at the history midpoint
                mid_fid = fid_start + self.history_frames // 2
                mid_frame = window_df[window_df["frame_id"] == mid_fid]
                if mid_frame.empty:
                    continue

                target_row = mid_frame[mid_frame["vehicle_id"] == target_vid]
                if target_row.empty:
                    continue

                tx = float(target_row["local_x"].iloc[0])
                ty = float(target_row["local_y"].iloc[0])

                # Rank neighbours by Euclidean distance to target
                neighbor_candidates = [v for v in fully_present if v != target_vid]
                if not neighbor_candidates:
                    continue

                dists = {}
                for nv in neighbor_candidates:
                    nr = mid_frame[mid_frame["vehicle_id"] == nv]
                    if nr.empty:
                        continue
                    nx = float(nr["local_x"].iloc[0])
                    ny = float(nr["local_y"].iloc[0])
                    dists[nv] = np.sqrt((nx - tx) ** 2 + (ny - ty) ** 2)

                sorted_neighbors = sorted(dists, key=dists.get)
                selected = sorted_neighbors[: self.max_neighbors]

                # Order: target first, then neighbours
                scene_vids = [target_vid] + selected
                N = len(scene_vids)

                # Build the (T, N, 4) array: [x, y, vx, vy]
                frames_arr = np.zeros((T, N, 4), dtype=np.float32)
                v_types = np.zeros(N, dtype=np.float32)
                v_lengths = np.zeros(N, dtype=np.float32)
                v_widths = np.zeros(N, dtype=np.float32)
                v_lanes = np.zeros(N, dtype=np.float32)

                for vi, vid in enumerate(scene_vids):
                    vdf = window_df[window_df["vehicle_id"] == vid].sort_values("frame_id")
                    frames_arr[:, vi, 0] = vdf["local_x"].values[:T]
                    frames_arr[:, vi, 1] = vdf["local_y"].values[:T]
                    frames_arr[:, vi, 2] = 0.0   # vx ≈ 0 on freeway
                    frames_arr[:, vi, 3] = vdf["v_vel"].values[:T]

                    v_types[vi] = vdf["v_class"].iloc[0]
                    v_lengths[vi] = vdf["v_length"].iloc[0]
                    v_widths[vi] = vdf["v_width"].iloc[0]
                    v_lanes[vi] = vdf["lane_id"].iloc[0]

                scene = {
                    "vehicle_ids": np.array(scene_vids),
                    "frames": frames_arr,
                    "vehicle_types": v_types,
                    "vehicle_lengths": v_lengths,
                    "vehicle_widths": v_widths,
                    "lane_ids": v_lanes,
                }
                scenes.append(scene)

                if max_scenes is not None and len(scenes) >= max_scenes:
                    break

            if max_scenes is not None and len(scenes) >= max_scenes:
                break

        self.scenes = scenes
        print(f"[NGSIMLoader] Built {len(scenes)} scenes")
        return scenes

    # ------------------------------------------------------------------ #
    #  Step 3: Build training samples                                     #
    # ------------------------------------------------------------------ #

    def build_samples(
        self,
        max_samples: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert scenes to training-ready (states, gt_pos, gt_goals, mask)
        tuples, using the same logic as ``create_sample()`` from
        ``generate_synthetic.py``.

        Each returned tuple has:
            states      : (1, T_h, N, F)   float32
            gt_pos      : (1, T_f, 2)      float32
            gt_goals    : (1, N-1, 4)       float32
            mask        : (1, N)            bool
        """
        from src.data.generate_synthetic import create_sample

        if not self.scenes:
            self.build_scenes(max_scenes=max_samples)

        samples = []
        for scene in self.scenes:
            try:
                sample = create_sample(
                    scene,
                    target_idx=0,
                    T_h=self.history_frames,
                    T_f=self.future_frames,
                )
                samples.append(sample)
            except Exception as exc:
                # Some scenes may fail (e.g. NaN in data) — skip them
                continue

            if max_samples is not None and len(samples) >= max_samples:
                break

        print(f"[NGSIMLoader] Built {len(samples)} training samples")
        return samples

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def get_dataset_stats(self) -> dict:
        """Return basic dataset statistics."""
        stats = {"location": self.location, "num_scenes": len(self.scenes)}
        if self.df is not None:
            stats.update({
                "total_rows": len(self.df),
                "unique_vehicles": int(self.df["vehicle_id"].nunique()),
                "frame_range": (
                    int(self.df["frame_id"].min()),
                    int(self.df["frame_id"].max()),
                ),
                "vehicle_classes": self.df["v_class"].value_counts().to_dict(),
            })
        return stats

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[list, list, list]:
        """Split samples into train / val / test sets.

        The split is done at the sample level with shuffling.

        Returns:
            (train_samples, val_samples, test_samples)
        """
        samples = self.build_samples() if not hasattr(self, "_samples_cache") else self._samples_cache
        self._samples_cache = samples

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(samples))

        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train: n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train = [samples[i] for i in train_idx]
        val = [samples[i] for i in val_idx]
        test = [samples[i] for i in test_idx]

        print(f"[NGSIMLoader] Split: train={len(train)}, "
              f"val={len(val)}, test={len(test)}")
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
        states, gt_pos, gt_goals, mask = self.samples[idx]
        # Remove the batch dim that create_sample adds (DataLoader re-batches)
        return states.squeeze(0), gt_pos.squeeze(0), gt_goals.squeeze(0)


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

    Each item in *batch* is ``(states, gt_pos, gt_goals)`` **without** a
    leading batch dimension (TrafficDataset squeezes it).

    Because different scenes may have different numbers of vehicles, we
    pad to the maximum N in the batch and return a mask.
    """
    states_list, gt_pos_list, gt_goals_list = zip(*batch)

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
    for i, g in enumerate(gt_goals_list):
        n = g.shape[0]
        gt_goals_padded[i, :n, :] = g

    return states_padded, gt_positions, gt_goals_padded, mask
