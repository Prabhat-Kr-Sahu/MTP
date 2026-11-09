#!/usr/bin/env python3
"""STGAT-EDGRU + Collision Risk Index, paper-faithful pipeline on NGSIM.

Su, Wang & Wang, Electronics 2025, 14, 3388 -- run the way the paper runs it on
HighD, but on NGSIM: 5 Hz, 3 s history / 5 s future, six features
(x, y, vx, vy, ax, ay) in a frame fixed at the main vehicle's first history
position, lane-change-centred samples balanced against lane keeping, every
vehicle predicted, teacher forcing, 120 epochs, batch 64, Adam 2e-4.
The repo-pipeline version is experiments/stgat_edgru.ipynb; STGAT_EDGRU.md
compares the two.

  python experiments/stgat_paper.py --prepare             # samples + split
  python experiments/stgat_paper.py --train [--ablation m1|m2|m3]
  python experiments/stgat_paper.py --eval                # RMSE/NLL/ADE/FDE tables
  python experiments/stgat_paper.py --risk                # CRI + ROC/PR on test
  python experiments/stgat_paper.py --case                # Fig. 6 analogue
Add --debug to any of them for a quick run on DEBUG_MAX_ROWS of us-101; debug
runs write to *_debug folders and never touch the real ones.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import DEBUG_MAX_ROWS, DEVICE, NGSIM_LOCATIONS, SEED  # noqa: E402
from src.data.loader import NGSIMDataLoader  # noqa: E402
from src.evaluation.collision import (  # noqa: E402
    auc_placements, brier_score, compute_pr_auc, compute_roc_auc, event_bootstrap,
    recall_at_precision, roc_curve,
)
from src.evaluation.risk_metrics import (  # noqa: E402
    METRICS, collision_radius, cri_aggregate, cri_fuse, cri_steps,
)
from src.losses import gaussian_nll  # noqa: E402
from src.models.stgat_edgru import STGATEDGRU  # noqa: E402

STEP = 2              # 10 Hz -> 5 Hz
DT_P = 0.2            # s per step after downsampling
TH, TF = 15, 25       # 3 s history, 5 s future
W = TH + TF
N_MAX = 16            # main vehicle + 15 neighbours
D_CLOSE = 50.0        # m, neighbourhood (paper)
SMOOTH = 11           # frames at 10 Hz (~1 s) centred moving average on positions
MAX_LANE = 6          # lanes >= 7 are ramps
# ponytail: the paper defers t_start to its ref [44], which is not available.
# t_start = last frame before the lane crossing whose |vy| is below this, at
# most BACK steps back; events with no such frame are dropped.
VY_START = 0.2        # m/s
BACK = 25             # steps (5 s)
EPOCHS, BATCH, LR = 120, 64, 2e-4
HORIZONS = (4, 9, 14, 19, 24)          # 1..5 s at 5 Hz
DIST_LABELS = (2.0, 3.0, 5.0)          # m, GT min centre distance (repo convention)
TTC_LABELS = (1.5, 3.0)                # s, GT min box TTC
SCORED = ("p_coll_gauss", "p_coll_mc", "min_dist", "ttc_true", "tet")
ABLATIONS = {"full": {}, "m1": {"use_transformer": False},
             "m2": {"use_attention": False}, "m3": {"use_gmu": False}}


def dirs(debug):
    tag = "_debug" if debug else ""
    out = ROOT / "experiments" / "results" / f"stgat_paper{tag}"
    ckpt = ROOT / "experiments" / "checkpoints" / f"stgat_paper{tag}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    return out, ckpt


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_tracks(debug):
    """Per-vehicle 5 Hz tracks: positions smoothed, velocity/acceleration by
    central differences of the smoothed positions (NGSIM has no lateral speed)."""
    loader = NGSIMDataLoader(locations=["us-101"] if debug else NGSIM_LOCATIONS,
                             max_rows=DEBUG_MAX_ROWS if debug else None)
    df = loader.fetch().sort_values(["vehicle_id", "frame_id"])
    tracks, dropped = {}, 0
    for vid, g in df.groupby("vehicle_id", sort=False):
        f = g["frame_id"].to_numpy(np.int64)
        if len(f) < 2 * STEP or (np.diff(f) != 1).any():
            dropped += 1                  # ponytail: gappy/duplicated tracks are dropped
            continue
        smooth = lambda c: g[c].rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
        lon, lat = smooth("local_y"), smooth("local_x")
        keep = f % STEP == 0
        f, lon, lat = f[keep], lon[keep], lat[keep]
        if len(f) < 3:
            continue
        vx, vy = np.gradient(lon, DT_P), np.gradient(lat, DT_P)
        tracks[int(vid)] = {
            "f0": int(f[0]), "n": len(f),
            "feat": np.stack([lon, lat, vx, vy, np.gradient(vx, DT_P),
                              np.gradient(vy, DT_P)], 1).astype(np.float32),
            "lane": g["lane_id"].to_numpy()[keep].astype(int),
            "dims": (float(g["v_length"].iloc[0]), float(g["v_width"].iloc[0])),
            "loc": str(g["location"].iloc[0]),
        }
    by_frame = defaultdict(list)
    for vid, t in tracks.items():
        for k in range(t["n"]):
            by_frame[t["f0"] + STEP * k].append(vid)
    print(f"tracks: {len(tracks)} kept, {dropped} gappy dropped")
    return tracks, by_frame


def _idx(t, frame):
    return (frame - t["f0"]) // STEP


def _covers(t, first, last):
    return t["f0"] <= first and t["f0"] + STEP * (t["n"] - 1) >= last


def neighbours_at(tracks, by_frame, vid, frame, first, last):
    """Vehicles within D_CLOSE of `vid` at `frame` whose track covers [first, last]."""
    p = tracks[vid]["feat"][_idx(tracks[vid], frame), :2]
    found = []
    for o in by_frame[frame]:
        t = tracks[o]
        if o != vid and _covers(t, first, last):
            d = float(np.linalg.norm(t["feat"][_idx(t, frame), :2] - p))
            if d <= D_CLOSE:
                found.append((d, o))
    return [o for _, o in sorted(found)[:N_MAX - 1]]


def window(tracks, ids, first, length):
    """(length, N_MAX, 6) features, mask, dims; positions relative to ids[0] at `first`."""
    feat = np.zeros((length, N_MAX, 6), np.float32)
    dims = np.zeros((N_MAX, 2), np.float32)
    for n, o in enumerate(ids):
        t = tracks[o]
        i = _idx(t, first)
        feat[:, n] = t["feat"][i:i + length]
        dims[n] = t["dims"]
    feat[:, :len(ids), :2] -= feat[0, 0, :2].copy()
    mask = np.zeros(N_MAX, bool)
    mask[:len(ids)] = True
    return feat, mask, dims


def find_events(tracks):
    """Lane changes (LCL/LCR) anchored at t_start, and 8 s lane-keeping windows."""
    lc, lk = [], []
    for vid, t in tracks.items():
        lane, vy = t["lane"], t["feat"][:, 3]
        for k in np.flatnonzero(lane[1:] != lane[:-1]) + 1:
            if max(lane[k - 1], lane[k]) > MAX_LANE:
                continue
            calm = [j for j in range(k - 1, max(k - 1 - BACK, -1), -1) if abs(vy[j]) < VY_START]
            if not calm:
                continue
            s = calm[0]                                   # t_start (index)
            # lane_id rises to the right in NGSIM (lane 1 = leftmost)
            kind = "LCR" if lane[k] > lane[k - 1] else "LCL"
            lc.append((vid, t["f0"] + STEP * (s - TH), kind, int(t["f0"] + STEP * k),
                       int(lane[k])))
        for s in range(0, t["n"] - W + 1, W):
            seg = lane[s:s + W]
            if (seg == seg[0]).all() and seg[0] <= MAX_LANE:
                lk.append((vid, t["f0"] + STEP * s, "LK", -1, int(seg[0])))
    return list(dict.fromkeys(lc)), lk


def prepare(args):
    out, ckpt = dirs(args.debug)
    tracks, by_frame = load_tracks(args.debug)
    lc, lk = find_events(tracks)
    rng = np.random.default_rng(SEED)
    n_lk = max(sum(e[2] == "LCL" for e in lc), sum(e[2] == "LCR" for e in lc))
    lk = [lk[i] for i in rng.choice(len(lk), size=min(n_lk, len(lk)), replace=False)]
    feats, masks, dims, meta = [], [], [], []
    for vid, first, kind, t_c, target_lane in lc + lk:
        last = first + STEP * (W - 1)
        if not _covers(tracks[vid], first, last):
            continue
        obs = first + STEP * (TH - 1)
        ids = [vid] + neighbours_at(tracks, by_frame, vid, obs, first, last)
        f, m, d = window(tracks, ids, first, W)
        feats.append(f), masks.append(m), dims.append(d)
        meta.append({"vid": vid, "first": first, "kind": kind, "t_c": t_c,
                     "target_lane": target_lane, "loc": tracks[vid]["loc"],
                     "ids": ids})
    vids = np.unique([m["vid"] for m in meta])
    order = rng.permutation(vids)
    cut1, cut2 = int(0.8 * len(order)), int(0.9 * len(order))
    which = {v: ("train" if i < cut1 else "val" if i < cut2 else "test")
             for i, v in enumerate(order)}
    split = np.array([which[m["vid"]] for m in meta])
    feat = torch.as_tensor(np.stack(feats))
    mask = torch.as_tensor(np.stack(masks))
    hist = feat[torch.as_tensor(split == "train"), :TH][mask[torch.as_tensor(split == "train")]
                                                         .unsqueeze(1).expand(-1, TH, -1)]
    stats = {"mean": hist.mean(0).tolist(), "std": hist.std(0).clamp(min=1e-6).tolist()}
    torch.save({"feat": feat, "mask": mask, "dims": torch.as_tensor(np.stack(dims)),
                "meta": meta, "split": split.tolist(), "stats": stats}, ckpt / "data.pt")
    counts = pd.crosstab(pd.Series([m["kind"] for m in meta], name="kind"),
                         pd.Series(split, name="split"))
    print(counts)
    (out / "data.json").write_text(json.dumps({
        "counts": counts.to_dict(), "n_samples": len(meta), "stats": stats,
        "lane_change_events_found": len(lc), "constants": {
            "STEP": STEP, "TH": TH, "TF": TF, "N_MAX": N_MAX, "D_CLOSE": D_CLOSE,
            "SMOOTH": SMOOTH, "VY_START": VY_START, "BACK": BACK, "MAX_LANE": MAX_LANE},
    }, indent=1))


def load_data(ckpt):
    data = torch.load(ckpt / "data.pt", weights_only=True)
    split = np.array(data["split"])
    data["idx"] = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    return data


def batch_tensors(data, idx):
    """Normalised history x (B,TH,N,6), mask, dims, GT displacement (B,N,TF,2), last pos."""
    feat = data["feat"][idx].to(DEVICE)
    mask = data["mask"][idx].to(DEVICE)
    mean = torch.tensor(data["stats"]["mean"], device=DEVICE)
    std = torch.tensor(data["stats"]["std"], device=DEVICE)
    x = ((feat[:, :TH] - mean) / std) * mask[:, None, :, None]
    last = feat[:, TH - 1, :, :2]                                   # (B,N,2)
    gt = (feat[:, TH:, :, :2] - last[:, None]).permute(0, 2, 1, 3)  # (B,N,TF,2)
    return x, mask, data["dims"][idx].to(DEVICE), gt, last, feat


def build_model(data, flags):
    return STGATEDGRU(in_dim=6, t_f=TF, d=128, heads=8, enc_layers=2, dropout=0.2,
                      d_close=D_CLOSE, pos_mean=data["stats"]["mean"][:2],
                      pos_std=data["stats"]["std"][:2], **flags).to(DEVICE)


def nll_all(pred, gt, mask):
    """Eq. 37 over every real vehicle: mean NLL per vehicle-step."""
    B, N, T, _ = pred.shape
    p, g = pred.reshape(B * N, T, 5), gt.reshape(B * N, T, 2)
    nll = gaussian_nll(p[..., :2], p[..., 2], p[..., 3], p[..., 4], g, reduction="none")
    m = mask.reshape(-1).float()
    return (nll * m).sum() / m.sum()


def predict(model, data, idx, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            x, mask, _, _, _, _ = batch_tensors(data, idx[s:s + batch])
            out.append(model.forward_all(x, mask))
    return torch.cat(out)


# --------------------------------------------------------------------------- #
#  Train / eval
# --------------------------------------------------------------------------- #
def train(args):
    out, ckpt = dirs(args.debug)
    data = load_data(ckpt)
    torch.manual_seed(SEED)
    model = build_model(data, ABLATIONS[args.ablation])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    run = ckpt / args.ablation
    run.mkdir(exist_ok=True)
    epochs = 2 if args.debug else EPOCHS
    rng = np.random.default_rng(SEED)
    log, best, t0 = [], float("inf"), time.time()
    val_idx = data["idx"]["val"]
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(data["idx"]["train"])
        total, n = 0.0, 0
        for s in range(0, len(order), BATCH):
            x, mask, _, gt, _, _ = batch_tensors(data, order[s:s + BATCH])
            loss = nll_all(model.forward_all(x, mask, gt), gt, mask)   # teacher forcing
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total, n = total + loss.item(), n + 1
        pred = predict(model, data, val_idx)
        _, vmask, _, vgt, _, _ = batch_tensors(data, val_idx)
        val_nll = nll_all(pred, vgt, vmask).item()
        val_rmse = horizon_rmse(pred[:, 0, :, :2], vgt[:, 0])["rmse_avg"]
        log.append({"epoch": epoch, "train_nll": total / n, "val_nll": val_nll,
                    "val_rmse_main": val_rmse})
        # Checkpoint on val RMSE of the main vehicle (the paper's reported metric;
        # it does not say how it selects). Val NLL is useless for this: teacher
        # forcing makes sigma overconfident, so free-running val NLL rises from
        # epoch 1 while RMSE falls -- picking on it would keep the epoch-1 model.
        if val_rmse < best:
            best = val_rmse
            torch.save(model.state_dict(), run / "best_model.pt")
        print(f"[{args.ablation}] epoch {epoch}: train {total / n:.3f} val nll {val_nll:.3f} "
              f"val rmse {val_rmse:.3f}", flush=True)
    (out / f"train_{args.ablation}.json").write_text(json.dumps({
        "ablation": args.ablation, "flags": ABLATIONS[args.ablation], "epochs": epochs,
        "batch": BATCH, "lr": LR, "minutes": (time.time() - t0) / 60, "best_val_rmse_main": best,
        "log": log}, indent=1))


def horizon_rmse(mu, gt):
    """mu, gt (M,TF,2) -> rmse at 1..5 s, average, ADE, FDE."""
    err = (mu - gt).norm(dim=-1)
    r = {f"rmse_{i + 1}s": err[:, h].square().mean().sqrt().item()
         for i, h in enumerate(HORIZONS)}
    r["rmse_avg"] = float(np.mean(list(r.values())))
    r["ade"], r["fde"] = err.mean().item(), err[:, -1].mean().item()
    return r


def scores_for(pred, gt, mask, meta, idx):
    """Main-vehicle, all-vehicle and per-class metrics of one prediction."""
    main = {**horizon_rmse(pred[:, 0, :, :2], gt[:, 0]),
            "nll": nll_all(pred[:, :1], gt[:, :1], mask[:, :1]).item(), "n": len(idx)}
    real = mask.reshape(-1)
    flat_p = pred.reshape(-1, TF, 5)[real]
    every = {**horizon_rmse(flat_p[..., :2], gt.reshape(-1, TF, 2)[real]),
             "nll": nll_all(pred, gt, mask).item(), "n": int(real.sum())}
    kinds = np.array([meta[i]["kind"] for i in idx])
    per = {}
    for k in ("LK", "LCL", "LCR"):
        sel = torch.as_tensor(kinds == k, device=pred.device)
        if sel.any():
            per[k] = horizon_rmse(pred[sel, 0, :, :2], gt[sel, 0])
    return {"main": main, "all": every, "per_class": per}


def evaluate(args):
    out, ckpt = dirs(args.debug)
    data = load_data(ckpt)
    idx = data["idx"]["test"]
    _, mask, _, gt, _, feat = batch_tensors(data, idx)
    results = {}
    # constant-velocity reference: last observed (vx, vy) held for 5 s
    t = torch.arange(1, TF + 1, device=DEVICE, dtype=torch.float32) * DT_P
    cv_mu = feat[:, TH - 1, :, 2:4].unsqueeze(2) * t[None, None, :, None]   # (B,N,TF,2)
    cv = torch.cat([cv_mu, torch.full_like(cv_mu, 1.0), torch.zeros_like(cv_mu[..., :1])], -1)
    results["cv"] = scores_for(cv, gt, mask, data["meta"], idx)
    for name, flags in ABLATIONS.items():
        weights = ckpt / name / "best_model.pt"
        if not weights.exists():
            continue
        model = build_model(data, flags)
        model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
        results[name] = scores_for(predict(model, data, idx), gt, mask, data["meta"], idx)
    (out / "eval.json").write_text(json.dumps(results, indent=1))
    cols = [f"rmse_{i}s" for i in range(1, 6)] + ["rmse_avg", "ade", "fde", "nll"]
    for part in ("main", "all"):
        print(f"\n{part}-vehicle test metrics (m):")
        print(pd.DataFrame({k: v[part] for k, v in results.items()}).T[cols].round(3))


# --------------------------------------------------------------------------- #
#  Risk
# --------------------------------------------------------------------------- #
def box_ttc(rel_p, rel_v, half_long, half_lat):
    """Time until two boxes overlap under constant relative velocity; 0 if they
    already do, inf if never. rel_* (...,2) = other - ego; half_* (...)."""
    lo, hi = [], []
    for axis, h in ((0, half_long), (1, half_lat)):
        p, v = rel_p[..., axis], rel_v[..., axis]
        moving = v.abs() > 1e-6
        safe_v = torch.where(moving, v, torch.ones_like(v))
        a, b = (-h - p) / safe_v, (h - p) / safe_v
        inside = p.abs() < h
        lo.append(torch.where(moving, torch.minimum(a, b),
                              torch.where(inside, torch.full_like(p, -np.inf),
                                          torch.full_like(p, np.inf))))
        hi.append(torch.where(moving, torch.maximum(a, b),
                              torch.where(inside, torch.full_like(p, np.inf),
                                          torch.full_like(p, -np.inf))))
    enter = torch.maximum(torch.maximum(lo[0], lo[1]), torch.zeros_like(lo[0]))
    leave = torch.minimum(hi[0], hi[1])
    return torch.where(enter < leave, enter, torch.full_like(enter, np.inf))


def pair_scores(pred, last, mask, dims):
    """Every risk score for main vs each neighbour, plus the multi-vehicle CRI.

    pred (B,N,TF,5) displacements, last (B,N,2) last observed positions."""
    mu = (pred[..., :2] + last[:, :, None]).permute(0, 2, 1, 3)         # (B,TF,N,2)
    sig, rho = pred[..., 2:4].permute(0, 2, 1, 3), pred[..., 4].permute(0, 2, 1)
    nb = mask[:, 1:]
    mu_j = torch.where(nb[:, None, :, None], mu[:, :, 1:], torch.full_like(mu[:, :, 1:], np.nan))
    radius = collision_radius(dims[:, 0], dims[:, 1:]).unsqueeze(1)
    out = {}
    for name, fixed in (("cri_paper", False), ("cri_fixed", True)):
        f, p, _ = cri_steps(mu[:, :, :1], sig[:, :, :1], rho[:, :, :1], mu_j,
                            sig[:, :, 1:], rho[:, :, 1:], radius, DT_P, fixed)
        out[name] = cri_fuse(f)                                         # (B,N-1)
        out[f"{name}_multi"] = cri_fuse(cri_aggregate(f, p).unsqueeze(-1))[:, 0]
    ctx = {"sigma": sig[:, :, 0], "rho": rho[:, :, 0], "ego_dims": dims[:, 0],
           "other_dims": dims[:, 1:]}
    for name in SCORED:
        out[name] = METRICS[name][0](mu[:, :, 0], mu_j, dt=DT_P, **ctx)
    return out


def gt_labels(feat, mask, dims):
    """Per pair: GT min centre distance and min box TTC over the 5 s future."""
    pos, vel = feat[:, TH:, :, :2], feat[:, TH:, :, 2:4]                 # (B,TF,N,2)
    rel_p, rel_v = pos[:, :, 1:] - pos[:, :, :1], vel[:, :, 1:] - vel[:, :, :1]
    half = 0.5 * (dims[:, :1] + dims[:, 1:])                            # (B,N-1,2)
    dist = rel_p.norm(dim=-1).min(1).values
    ttc = box_ttc(rel_p, rel_v, half[:, None, :, 0], half[:, None, :, 1]).min(1).values
    real = mask[:, 1:]
    return dist.masked_fill(~real, np.inf), ttc.masked_fill(~real, np.inf)


def roc_block(scores, labels, events, higher):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    keep = ~np.isnan(s)
    s, y, events = s[keep], y[keep], events[keep]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return {"n_pos": int(y.sum()), "n": int(len(y))}
    s = np.nan_to_num(s, posinf=1e9, neginf=-1e9)
    place = auc_placements(s, y, higher)
    lo, hi = event_bootstrap(place, events[y == 1])[0]
    return {"n_pos": int(y.sum()), "n": int(len(y)),
            "roc_auc": compute_roc_auc(s, y, higher), "ci": [float(lo), float(hi)],
            "pr_auc": compute_pr_auc(s, y, higher),
            "recall_at_p50": recall_at_precision(s, y, higher, 0.5)}


def risk(args):
    out, ckpt = dirs(args.debug)
    data = load_data(ckpt)
    idx = data["idx"]["test"]
    model = build_model(data, {})
    model.load_state_dict(torch.load(ckpt / "full" / "best_model.pt", map_location=DEVICE,
                                     weights_only=True))
    pred = predict(model, data, idx)
    scores = defaultdict(list)
    dist, ttc = [], []
    for s in range(0, len(idx), 128):
        sl = slice(s, s + 128)
        _, mask, dims, _, last, feat = batch_tensors(data, idx[sl])
        for k, v in pair_scores(pred[sl], last, mask, dims).items():
            scores[k].append(v.cpu())
        d, t = gt_labels(feat, mask, dims)
        dist.append(d.cpu()), ttc.append(t.cpu())
    scores = {k: torch.cat(v).numpy() for k, v in scores.items()}
    dist, ttc = torch.cat(dist).numpy(), torch.cat(ttc).numpy()
    real = np.isfinite(dist)                                        # real pairs
    sample = np.repeat(np.arange(len(idx))[:, None], N_MAX - 1, 1)
    labels = {**{f"dist<{d:g}m": dist < d for d in DIST_LABELS},
              **{f"ttc<{t:g}s": ttc < t for t in TTC_LABELS}}
    higher = {**{k: METRICS[k][1] for k in SCORED}, "cri_paper": True, "cri_fixed": True}
    report = {"n_samples": len(idx), "n_pairs": int(real.sum()), "pair": {}, "window": {},
              "brier": {}}
    for lname, lab in labels.items():
        report["pair"][lname] = {
            m: roc_block(scores[m][real], lab[real], sample[real], higher[m]) for m in higher}
        win = lab.any(1)
        wscore = {m: (np.where(real, scores[m], -np.inf).max(1) if higher[m]
                      else np.where(real, scores[m], np.inf).min(1)) for m in higher}
        wscore.update({f"{m}_multi": scores[f"{m}_multi"] for m in ("cri_paper", "cri_fixed")})
        hw = {**higher, "cri_paper_multi": True, "cri_fixed_multi": True}
        report["window"][lname] = {m: roc_block(v, win, np.arange(len(idx)), hw[m])
                                   for m, v in wscore.items()}
        report["brier"][lname] = {m: brier_score(scores[m][real], lab[real])
                                  for m in ("p_coll_gauss", "p_coll_mc")}
        plot_roc(out / f"roc_window_{lname.replace('<', '_lt_')}.png", wscore, win, hw, lname)
    (out / "risk.json").write_text(json.dumps(report, indent=1))
    rows = [{"label": l, "level": lvl, "measure": m, **{k: v for k, v in r.items() if k != "ci"},
             "ci_lo": r.get("ci", [np.nan])[0], "ci_hi": r.get("ci", [np.nan, np.nan])[1]}
            for lvl in ("pair", "window") for l, block in report[lvl].items()
            for m, r in block.items()]
    table = pd.DataFrame(rows).reindex(columns=[
        "label", "level", "measure", "n_pos", "n", "roc_auc", "ci_lo", "ci_hi", "pr_auc",
        "recall_at_p50"])
    table.to_csv(out / "risk_auc.csv", index=False)
    print(table.round(3).to_string(index=False))


def plot_roc(path, scores, labels, higher, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if labels.sum() < 5:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for name in ("cri_paper_multi", "cri_fixed_multi", "p_coll_gauss", "min_dist", "ttc_true"):
        s = np.nan_to_num(np.asarray(scores[name], float), posinf=1e9, neginf=-1e9)
        fpr, tpr, _ = roc_curve(s, labels, higher[name])
        ax.plot(fpr, tpr, label=f"{name} ({compute_roc_auc(s, labels, higher[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title=f"window-level ROC, label {title}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Paper-style case (Fig. 6 analogue)
# --------------------------------------------------------------------------- #
def case(args):
    out, ckpt = dirs(args.debug)
    data = load_data(ckpt)
    tracks, by_frame = load_tracks(args.debug)
    model = build_model(data, {})
    model.load_state_dict(torch.load(ckpt / "full" / "best_model.pt", map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    mean = torch.tensor(data["stats"]["mean"], device=DEVICE)
    std = torch.tensor(data["stats"]["std"], device=DEVICE)

    # candidates: test-split lane changes by GT min box TTC over the 8 s window
    cands = []
    for i in data["idx"]["test"]:
        m = data["meta"][i]
        if m["kind"] == "LK":
            continue
        feat, mask, dims = data["feat"][i], data["mask"][i], data["dims"][i]
        rel_p = feat[:, 1:, :2] - feat[:, :1, :2]
        rel_v = feat[:, 1:, 2:4] - feat[:, :1, 2:4]
        half = 0.5 * (dims[:1] + dims[1:])
        ttc = box_ttc(rel_p, rel_v, half[None, :, 0], half[None, :, 1])[:, mask[1:]]
        if not ttc.numel():
            continue
        overlap = (ttc == 0).any().item()              # boxes overlapping = track error
        jerky = feat[:, 0, 4].abs().max().item() > 8.0
        cands.append({"index": int(i), "vid": m["vid"], "kind": m["kind"], "t_c": m["t_c"],
                      "min_ttc": float(ttc.min()), "plausible": not (overlap or jerky)})
    cands.sort(key=lambda c: (not c["plausible"], c["min_ttc"]))
    (out / "cases.json").write_text(json.dumps(cands[:20], indent=1))
    pick = cands[0] if args.case_index is None else next(
        c for c in cands if c["index"] == args.case_index)
    print("case:", pick)

    meta = data["meta"][pick["index"]]
    vid, t_c, target_lane = meta["vid"], meta["t_c"], meta["target_lane"]
    main = tracks[vid]
    rows = []
    end_first = max(t_c - 80, main["f0"] + STEP * (TH - 1))     # 8 s before the crossing
    end_last = min(t_c + 40, main["f0"] + STEP * (main["n"] - 1))
    for e in range(end_first - end_first % STEP, end_last + 1, STEP):
        first = e - STEP * (TH - 1)
        if not _covers(main, first, e):
            continue
        ids = [vid] + neighbours_at(tracks, by_frame, vid, e, first, e)
        feat, mask, dims = window(tracks, ids, first, TH)
        feat_t = torch.as_tensor(feat, device=DEVICE)[None]
        mask_t = torch.as_tensor(mask, device=DEVICE)[None]
        dims_t = torch.as_tensor(dims, device=DEVICE)[None]
        x = ((feat_t - mean) / std) * mask_t[:, None, :, None]
        with torch.no_grad():
            pred = model.forward_all(x, mask_t)
        sc = pair_scores(pred, feat_t[:, -1, :, :2], mask_t, dims_t)
        rows.append({"t": (e - t_c) / 10.0, "frame": e, "n_neighbours": len(ids) - 1,
                     "cri_paper": float(sc["cri_paper_multi"][0]),
                     "cri_fixed": float(sc["cri_fixed_multi"][0]),
                     "p_coll_gauss_max": float(np.nan_to_num(sc["p_coll_gauss"][0].max().item())),
                     **surrogates(tracks, by_frame, vid, e, target_lane)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "case_scores.csv", index=False)
    summary = case_summary(frame)
    (out / "case_summary.json").write_text(json.dumps({"case": pick, **summary}, indent=1))
    print(frame.round(3).to_string(index=False))
    print(summary)
    plot_case(out / "risk_timeline_paper_case.png", frame, pick, summary)


def surrogates(tracks, by_frame, vid, e, target_lane):
    """ITTC and THW to the in-path leader, LCTTC to target-lane vehicles, at frame e.

    In-path leader: the nearest vehicle ahead overlapping the main vehicle
    laterally. LCTTC (ref [49] not available): 2-D box TTC under constant
    velocity against every vehicle currently in the target lane."""
    me = tracks[vid]
    p = me["feat"][_idx(me, e)]
    L, Wd = me["dims"]
    lead_gap, lead_dv, lcttc = np.inf, 0.0, np.inf
    for o in by_frame[e]:
        if o == vid:
            continue
        t = tracks[o]
        q = t["feat"][_idx(t, e)]
        dx, dy = q[0] - p[0], q[1] - p[1]
        if dx > 0 and abs(dy) < 0.5 * (Wd + t["dims"][1]):
            gap = dx - 0.5 * (L + t["dims"][0])
            if gap < lead_gap:
                lead_gap, lead_dv = gap, p[2] - q[2]
        if t["lane"][_idx(t, e)] == target_lane:
            rp = torch.tensor([dx, dy])
            rv = torch.tensor([q[2] - p[2], q[3] - p[3]])
            ttc = box_ttc(rp, rv, torch.tensor(0.5 * (L + t["dims"][0])),
                          torch.tensor(0.5 * (Wd + t["dims"][1])))
            lcttc = min(lcttc, float(ttc))
    gap = max(lead_gap, 0.1)
    return {"ittc": max(lead_dv, 0.0) / gap if np.isfinite(lead_gap) else 0.0,
            "thw_s": gap / max(p[2], 0.1) if np.isfinite(lead_gap) else np.inf,
            "lcttc_s": lcttc, "lead_gap_m": lead_gap}


def case_summary(frame):
    def peak(col, low=False):
        s = frame[col].replace([np.inf, -np.inf], np.nan)
        if s.isna().all():
            return None
        i = s.idxmin() if low else s.idxmax()
        return {"t": float(frame.t[i]), "value": float(s[i])}
    peaks = {"cri_paper": peak("cri_paper"), "cri_fixed": peak("cri_fixed"),
             "ittc": peak("ittc"), "thw_s": peak("thw_s", low=True),
             "lcttc_s": peak("lcttc_s", low=True)}
    lead = {f"{c}_vs_{o}": round(peaks[o]["t"] - peaks[c]["t"], 2)
            for c in ("cri_paper", "cri_fixed") for o in ("ittc", "thw_s", "lcttc_s")
            if peaks[c] and peaks[o]}
    return {"peaks": peaks, "lead_s": lead}


def plot_case(path, frame, pick, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    top.plot(frame.t, frame.cri_paper, label="CRI (paper, as printed)")
    top.plot(frame.t, frame.cri_fixed, label="CRI (fixed)")
    top.set_ylabel("CRI")
    top.legend(fontsize=8)
    bot.plot(frame.t, frame.ittc, color="C3", label="ITTC (1/s)")
    bot.set_ylabel("ITTC (1/s)")
    right = bot.twinx()
    right.plot(frame.t, frame.thw_s.replace(np.inf, np.nan), color="C2", label="THW (s)")
    right.plot(frame.t, frame.lcttc_s.replace(np.inf, np.nan), color="C4", label="LCTTC (s)")
    right.set_ylabel("THW, LCTTC (s)")
    for ax in (top, bot):
        ax.axvline(0, color="k", lw=0.6, ls=":")
        for name, p in summary["peaks"].items():
            if p:
                ax.axvline(p["t"], lw=0.5, ls="--", color="grey")
    lines = bot.get_legend_handles_labels()
    rlines = right.get_legend_handles_labels()
    bot.legend(lines[0] + rlines[0], lines[1] + rlines[1], fontsize=8)
    bot.set_xlabel("time from lane crossing (s)")
    fig.suptitle(f"NGSIM {pick['kind']} vehicle {pick['vid']}, GT min box TTC "
                 f"{pick['min_ttc']:.2f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for m in ("prepare", "train", "eval", "risk", "case"):
        mode.add_argument(f"--{m}", action="store_true")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    parser.add_argument("--case-index", type=int, default=None,
                        help="sample index from cases.json to plot instead of the top one")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    {"prepare": prepare, "train": train, "eval": evaluate, "risk": risk,
     "case": case}[next(m for m in ("prepare", "train", "eval", "risk", "case")
                        if getattr(args, m))](args)


if __name__ == "__main__":
    main()
