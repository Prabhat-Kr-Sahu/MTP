"""Spatial Encoder: MultiHead Attention + GLU + LayerNorm."""
import torch
import torch.nn as nn


class GLU(nn.Module):
    """Gated Linear Unit."""

    def forward(self, x):
        """
        Args:
            x: (..., 2*D) split into two halves
        Returns:
            (..., D) gated output
        """
        d = x.shape[-1] // 2
        return x[..., :d] * torch.sigmoid(x[..., d:])


class SpatialEncoder(nn.Module):
    """
    Models vehicle-to-vehicle interactions via multi-head self-attention.

    For each timestep, applies multi-head attention across all vehicles.
    Input: (B, T_h, N_v+1, D)
    Output: (B, T_h, N_v+1, D)
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.glu = GLU()
        self.layer_norm = nn.LayerNorm(d_model)
        # Projection to 2*D for GLU
        self.proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_h, N_v+1, D) motion embeddings
            mask: (B, N_v+1) boolean mask, True for real vehicles
        Returns:
            H_s: (B, T_h, N_v+1, D) spatial features
        """
        B, T_h, N, D = x.shape

        # Process each timestep independently with attention across vehicles
        out = torch.zeros_like(x)
        for t in range(T_h):
            xt = x[:, t, :, :]  # (B, N, D)

            if mask is not None:
                # key_padding_mask: True means ignore (padding)
                # nn.MultiheadAttention with batch_first expects (B, N)
                key_padding_mask = ~mask  # True for padding
                attn_out, _ = self.multi_head_attn(
                    query=xt, key=xt, value=xt,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
            else:
                attn_out, _ = self.multi_head_attn(
                    query=xt, key=xt, value=xt,
                    need_weights=False,
                )

            out[:, t, :, :] = attn_out

        # GLU + residual + LayerNorm
        proj_out = self.proj(out)  # (B, T_h, N, 2*D)
        glu_out = self.glu(proj_out)  # (B, T_h, N, D)
        out = self.layer_norm(glu_out + out)

        return out
