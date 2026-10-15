"""Motion Encoder: FC + ELU + LSTM."""
import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    Encodes vehicle states and risk features into motion embeddings.

    Takes:  vehicle state + risk features, shape (B, T_h, N_v+1, F)
    Returns: motion embedding H_m, shape (B, T_h, N_v+1, D)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        # NOTE: single-layer LSTM must use dropout=0 (PyTorch warns otherwise).
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_h, N_v+1, F) vehicle states with risk features
        Returns:
            H_m: (B, T_h, N_v+1, D) motion embedding
        """
        B, T_h, N, F = x.shape
        # Reshape to process each vehicle independently
        x_flat = x.view(B * T_h * N, F)  # (B*T_h*N, F)
        emb = self.elu(self.fc(x_flat))  # (B*T_h*N, D)
        emb = emb.view(B, T_h, N, -1)  # (B, T_h, N, D)

        # Process temporal sequence for each vehicle
        # LSTM expects (B*N, T_h, D)
        emb_per_vehicle = emb.transpose(1, 2).contiguous()  # (B, N, T_h, D)
        B, N, T_h, D = emb_per_vehicle.shape
        emb_seq = emb_per_vehicle.view(B * N, T_h, D)  # (B*N, T_h, D)

        lstm_out, _ = self.lstm(emb_seq)  # (B*N, T_h, D)
        lstm_out = lstm_out.view(B, N, T_h, D).transpose(1, 2)  # (B, T_h, N, D)
        lstm_out = self.layer_norm(lstm_out)

        return lstm_out
