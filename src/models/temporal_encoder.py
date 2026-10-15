"""Temporal Encoder: Positional Encoding + MultiHead Attention + GLU + LayerNorm."""
import torch
import torch.nn as nn
import math


class GLU(nn.Module):
    """Gated Linear Unit."""
    def forward(self, x):
        d = x.shape[-1] // 2
        return x[..., :d] * torch.sigmoid(x[..., d:])


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalEncoder(nn.Module):
    """Models temporal dependencies across historical timestamps.

    Input: (B, T_h, N_v+1, D) -> Output: (B, T_h, N_v+1, D)
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.glu = GLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_h, N_v+1, D) spatial features
            mask: (B, N_v+1) boolean mask
        Returns:
            C: (B, T_h, N_v+1, D) temporal features
        """
        B, T_h, N, D = x.shape

        # Add sinusoidal positional encoding per timestep.
        # pe: (1, T_h, D) -> (1, T_h, 1, D) to broadcast over vehicles.
        pe = self.pos_encoder.pe[:, :T_h, :]  # (1, T_h, D)
        x = x + pe.unsqueeze(2)  # broadcast over N

        # Temporal self-attention across time for each vehicle independently.
        # Reshape to (B*N, T_h, D), attend over T_h, reshape back.
        x_seq = x.transpose(1, 2).contiguous().view(B * N, T_h, D)  # (B*N, T_h, D)
        attn_out, _ = self.multi_head_attn(
            query=x_seq, key=x_seq, value=x_seq,
            need_weights=False,
        )  # (B*N, T_h, D)
        out = attn_out.view(B, N, T_h, D).transpose(1, 2).contiguous()  # (B, T_h, N, D)

        # Zero out padded vehicles so they do not contribute downstream.
        if mask is not None:
            out = out * mask.view(B, 1, N, 1).float()

        # GLU + residual + LayerNorm
        proj_out = self.proj(out)
        glu_out = self.glu(proj_out)
        out = self.layer_norm(glu_out + out)

        return out
