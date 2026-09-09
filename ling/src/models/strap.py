"""Full STRAP model: Spatial-Temporal Risk-Attentive trajectory Prediction."""
import torch
import torch.nn as nn
from src.models.motion_encoder import MotionEncoder
from src.models.spatial_encoder import SpatialEncoder
from src.models.temporal_encoder import TemporalEncoder
from src.models.goal_predictor import GoalPredictor
from src.models.risk_decoder import RiskAttentiveDecoder
from src.config import D, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NUM_HEADS, DROPOUT


class STRAP(nn.Module):
    """
    Full STRAP model.

    Architecture:
    - Motion Encoder: FC + ELU + LSTM
    - Spatial Encoder: MultiHeadAttn + GLU + LayerNorm (x3)
    - Temporal Encoder: PE + MultiHeadAttn + GLU + LayerNorm (x3)
    - Goal Predictor: MLP for neighbor goals
    - Risk-Attentive Decoder: cross-attention + LSTM trajectory generator

    Input:
        states: (B, T_h, N_v+1, F) vehicle states with risk features
        mask: (B, N_v+1) boolean mask for real vehicles
    Output:
        traj_dist: (B, T_f, 5) [mu_x, mu_y, sigma_x, sigma_y, rho]
        neighbor_goals: (B, N_v, 4)
        risk_field: (B, K, 2)
    """

    def __init__(
        self,
        input_dim: int = 10,  # 8 base state features + 2 risk features
        d_model: int = 64,
        num_heads: int = 4,
        k_intentions: int = 100,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.k = k_intentions

        # Motion Encoder
        self.motion_encoder = MotionEncoder(input_dim, d_model, dropout)

        # Spatial Encoder layers
        self.spatial_encoders = nn.ModuleList([
            SpatialEncoder(d_model, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Temporal Encoder layers
        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(d_model, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Goal Predictor
        self.goal_predictor = GoalPredictor(d_model, 128, dropout)

        # Risk-Attentive Decoder
        self.risk_decoder = RiskAttentiveDecoder(
            d_model, num_heads, k_intentions, num_decoder_layers, dropout
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            states: (B, T_h, N_v+1, F) vehicle states with risk features
            mask: (B, N_v+1) bool mask, True = real vehicle
        Returns:
            traj_dist: (B, T_f, 5) trajectory distribution
            neighbor_goals: (B, N_v, 4)
            risk_field: (B, K, 2)
        """
        B, T_h, N, F = states.shape

        # Motion encoding
        H_m = self.motion_encoder(states)  # (B, T_h, N, D)

        # Spatial encoding (stacked layers)
        H_s = H_m
        for enc in self.spatial_encoders:
            H_s = enc(H_s, mask)

        # Temporal encoding (stacked layers)
        C = H_s
        for enc in self.temporal_encoders:
            C = enc(C, mask)

        # Separate target and neighbors
        target_enc = C[:, :, 0, :]  # (B, T_h, D) target vehicle encoding
        neighbor_enc = C[:, :, 1:, :]  # (B, T_h, N_v, D)

        # Neighbor goal prediction
        neighbor_goals = self.goal_predictor(neighbor_enc)  # (B, N_v, 4)

        # Neighbor risk features (from input states)
        # states[:, :, 1:, -2:] contains S-field and O-field placeholders
        # average over time -> (B, N_v, 2)
        neighbor_risk = states[:, :, 1:, -2:].mean(dim=1)  # average over time

        # Risk-attentive decoding
        neighbor_mask = mask[:, 1:] if mask is not None else None
        traj_dist, risk_field = self.risk_decoder(
            target_enc, neighbor_goals, neighbor_risk, neighbor_mask
        )

        return traj_dist, neighbor_goals, risk_field
