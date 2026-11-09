"""STGAT-EDGRU (Su, Wang & Wang, Electronics 2025, 14, 3388).

Transformer temporal encoder + distance-masked graph attention over vehicles +
gated multimodal unit (GMU) fusion + GRU encoder / autoregressive GRU decoder
emitting a bivariate Gaussian per future step.

Where the paper is ambiguous or self-contradictory, the choice made here:
  * Spatial input: the paper says the GAT takes the "spatio-temporal fusion
    feature f_j", but fusion happens after the GAT. Here the GAT takes the
    per-step input embedding, so the two streams stay independent until the GMU.
  * GMU: Eq. 22 (f = z*fs + (1-z)*fr, z = sigmoid(W[fs, fr])) is followed;
    Algorithm 1 swaps z and 1-z and adds instead of concatenating.
  * Output: displacement of each vehicle from its own last observed position
    (the repo's convention); callers add that position back if they want
    absolute coordinates.

Ablations (paper Table 3): use_transformer=False is M1 (linear embedding only),
use_attention=False is M2 (uniform weights over the same neighbourhood),
use_gmu=False is M3 (concatenate temporal and spatial features).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoid(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class STGATEDGRU(nn.Module):
    def __init__(self, in_dim, t_f, d=128, heads=8, enc_layers=2, dropout=0.2,
                 d_close=50.0, pos_mean=None, pos_std=None,
                 use_transformer=True, use_attention=True, use_gmu=True):
        super().__init__()
        self.t_f, self.d, self.heads, self.d_close = t_f, d, heads, d_close
        self.use_transformer, self.use_attention, self.use_gmu = (
            use_transformer, use_attention, use_gmu)
        # Features 0/1 are positions; (x * std + mean) recovers metres for the
        # 50 m neighbourhood. Identity when the inputs are already raw.
        self.register_buffer("pos_mean", torch.as_tensor(
            pos_mean if pos_mean is not None else [0.0, 0.0], dtype=torch.float32))
        self.register_buffer("pos_std", torch.as_tensor(
            pos_std if pos_std is not None else [1.0, 1.0], dtype=torch.float32))

        self.embed = nn.Linear(in_dim, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, enc_layers)

        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.attn_out = nn.Linear(d, d)
        self.attn_norm = nn.LayerNorm(d)
        self.attn_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d, d))

        self.w_s, self.w_r = nn.Linear(d, d), nn.Linear(d, d)
        self.w_z = nn.Linear(2 * d, d)
        nn.init.zeros_(self.w_z.bias)   # gate starts at 0.5 (paper: avoid saturation)

        self.e_gru = nn.GRU(d if use_gmu else 2 * d, d, batch_first=True)
        self.d_gru = nn.GRUCell(2, d)
        self.head = nn.Linear(d, 5)
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def _spatial(self, e, pos, mask):
        """Graph attention at every history step. e (B,T,N,d), pos (B,T,N,2)."""
        B, T, N, d = e.shape
        dist = torch.cdist(pos.reshape(B * T, N, 2), pos.reshape(B * T, N, 2))
        allowed = (dist <= self.d_close) & mask.repeat_interleave(T, 0).unsqueeze(1)
        allowed = allowed | torch.eye(N, dtype=torch.bool, device=e.device)  # self always
        x = e.reshape(B * T, N, d)
        v = self.v(x).view(B * T, N, self.heads, d // self.heads).transpose(1, 2)
        if self.use_attention:
            q = self.q(x).view(B * T, N, self.heads, d // self.heads).transpose(1, 2)
            k = self.k(x).view(B * T, N, self.heads, d // self.heads).transpose(1, 2)
            score = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
            score = score.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            w = score.softmax(-1)
        else:                       # M2: same neighbourhood, equal weights
            w = allowed.float() / allowed.float().sum(-1, keepdim=True)
            w = w.unsqueeze(1).expand(-1, self.heads, -1, -1)
        r = (w @ v).transpose(1, 2).reshape(B * T, N, d)
        r = self.attn_norm(x + self.drop(self.attn_out(r)))
        return self.attn_mlp(r).view(B, T, N, d)

    def encode(self, x, mask):
        """x (B,T,N,F) -> hidden state at the observation time, (B,N,d)."""
        B, T, N, _ = x.shape
        pos = x[..., :2] * self.pos_std + self.pos_mean
        e = self.embed(x)                                           # (B,T,N,d)
        seq = e.permute(0, 2, 1, 3).reshape(B * N, T, self.d)
        if self.use_transformer:
            seq = self.temporal(seq + _sinusoid(T, self.d, x.device))
        temporal = seq.view(B, N, T, self.d)
        spatial = self._spatial(e, pos, mask).permute(0, 2, 1, 3)  # (B,N,T,d)
        if self.use_gmu:
            fs, fr = torch.tanh(self.w_s(temporal)), torch.tanh(self.w_r(spatial))
            z = torch.sigmoid(self.w_z(torch.cat([fs, fr], -1)))
            fused = z * fs + (1 - z) * fr
        else:                       # M3: plain concatenation
            fused = torch.cat([temporal, spatial], -1)
        _, h = self.e_gru(fused.reshape(B * N, T, -1))
        return h[-1].view(B, N, self.d)

    def decode(self, h, gt=None):
        """h (M,d) -> (M,t_f,5). gt (M,t_f,2) enables teacher forcing."""
        prev = h.new_zeros(h.shape[0], 2)
        out = []
        for t in range(self.t_f):
            h = self.d_gru(prev, h)
            o = self.head(self.drop(h))
            out.append(o)
            prev = gt[:, t] if gt is not None else o[:, :2]
        o = torch.stack(out, 1)
        sigma = F.softplus(o[..., 2:4]) + 1e-3
        return torch.cat([o[..., :2], sigma, torch.tanh(o[..., 4:])], -1)

    def forward_all(self, x, mask, gt=None):
        """Every vehicle: (B,N,t_f,5). gt (B,N,t_f,2) for teacher forcing."""
        B, _, N, _ = x.shape
        h = self.encode(x, mask).reshape(B * N, self.d)
        g = gt.reshape(B * N, self.t_f, 2) if gt is not None else None
        return self.decode(h, g).view(B, N, self.t_f, 5)

    def forward(self, states, mask, dims=None):
        """Repo contract: target only (vehicle 0); decodes just that vehicle."""
        B, _, N, _ = states.shape
        traj = self.decode(self.encode(states, mask)[:, 0])
        return traj, states.new_zeros(B, N - 1, 4), None
