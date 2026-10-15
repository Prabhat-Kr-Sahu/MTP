"""Risk-Attentive Feature Fusion Decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import (
    GAMMA_X, GAMMA_Y, ALPHA_X, ALPHA_Y,
    D_STAR, T_STAR, BETA_1, BETA_2, DT,
)
from src.risk.o_field import closest_approach, compute_o_field
from src.risk.s_field import compute_s_field


class RiskAttentiveDecoder(nn.Module):
    """
    Risk-attentive feature fusion decoder.

    Integrates encoded features with prospective risk for each
    spatial intention mode to enhance trajectory prediction.

    Steps:
    1. Generate K intention modes (k-means codebook)
    2. Predict neighbor goals
    3. Compute risk for each intention mode
    4. Embed risk+intention into queries
    5. Cross-attend to target representation
    6. Generate trajectory distribution
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        k_intentions: int = 100,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        t_future: int = 50,
    ):
        super().__init__()
        self.k = k_intentions
        self.d_model = d_model
        self.t_future = t_future

        # Intention mode codebook (initialized randomly, fitted during training)
        self.intention_means = nn.Parameter(
            torch.randn(k_intentions, 4) * 0.01  # [x, y, vx, vy]
        )

        # Risk field embedding MLP: (2 + 4) -> D
        self.risk_embed = nn.Sequential(
            nn.Linear(2 + 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Multi-head cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_decoder_layers)
        ])

        # Trajectory generator: LSTM + MLP
        # NOTE: single-layer LSTM must use dropout=0 (PyTorch warns otherwise).
        self.trajectory_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
            num_layers=1,
            dropout=0,
        )
        self.trajectory_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 5),  # [mu_x, mu_y, sigma_x, sigma_y, rho]
        )

        # Output projection for target encoding
        self.target_proj = nn.Linear(d_model, d_model)

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_decoder_layers)
        ])

    def set_intentions(self, centers: torch.Tensor):
        """Load a k-means fitted intention codebook (K, 4)."""
        assert centers.shape == (self.k, 4), f"Expected ({self.k},4), got {tuple(centers.shape)}"
        with torch.no_grad():
            self.intention_means.copy_(centers.to(self.intention_means.device).float())

    def forward(
        self,
        target_encoded: torch.Tensor,  # (B, T_h, D)
        neighbor_goals: torch.Tensor,   # (B, N_v, 4)
        neighbor_risk: torch.Tensor,    # (B, N_v, 2) [S-field, O-field]
        neighbor_mask: torch.Tensor,    # (B, N_v) bool
    ) -> tuple:
        """
        Args:
            target_encoded: (B, T_h, D) encoded target history
            neighbor_goals: (B, N_v, 4) predicted neighbor goals
            neighbor_risk: (B, N_v, 2) subjective + objective risk
            neighbor_mask: (B, N_v) boolean mask

        Returns:
            traj_dist: (B, T_f, 5) trajectory distribution parameters
            risk_field: (B, K, 2) risk per intention mode
        """
        B, T_h, D = target_encoded.shape
        N_v = neighbor_goals.shape[1]
        T_f = self.t_future  # future timesteps (5s at 0.1s = 50)

        # Expand target encoding for cross-attention
        target_key = self.target_proj(target_encoded)  # (B, T_h, D)
        target_value = target_encoded  # (B, T_h, D)

        # Build intention modes (K modes)
        # Each mode: [x, y, vx, vy]
        intentions = self.intention_means.unsqueeze(0).expand(B, -1, -1)  # (B, K, 4)

        intention_pos = intentions[:, :, None, :2]
        intention_vel = intentions[:, :, None, 2:4]
        neighbor_pos = neighbor_goals[:, None, :, :2]
        neighbor_vel = neighbor_goals[:, None, :, 2:4]
        relative_position = intention_pos - neighbor_pos
        relative_velocity = intention_vel - neighbor_vel

        s_field = compute_s_field(
            relative_position[..., 0], relative_position[..., 1],
            GAMMA_X, GAMMA_Y, ALPHA_X, ALPHA_Y,
        )
        min_distance, time_to_closest = closest_approach(
            relative_position, relative_velocity
        )
        o_field = compute_o_field(
            min_distance, time_to_closest.clamp(min=DT),
            D_STAR, T_STAR, BETA_1, BETA_2,
        )
        if neighbor_mask is not None:
            valid_neighbors = neighbor_mask[:, None, :].to(torch.bool)
            s_field = s_field.masked_fill(~valid_neighbors, 0.0)
            o_field = o_field.masked_fill(~valid_neighbors, 0.0)

        R_s = s_field.sum(dim=-1)
        R_o = o_field.sum(dim=-1)

        # Risk field: (B, K, 2)
        risk_field = torch.stack([R_s, R_o], dim=-1)

        # Concatenate risk + intention and embed
        intent_risk = torch.cat([risk_field, intentions], dim=-1)  # (B, K, 6)
        risk_query = self.risk_embed(intent_risk)  # (B, K, D)

        # Cross-attention: risk queries attend to target encoding
        # target_key: (B, T_h, D) -> (B, T_h, D) for key/value
        # risk_query: (B, K, D) -> query
        attn_out = risk_query  # (B, K, D)

        for i, attn_layer in enumerate(self.cross_attn_layers):
            # Cross-attention: query=risk, key/value=target
            # Use iteratively updated queries (attn_out) per decoder layer.
            attn_out, _ = attn_layer(
                query=attn_out,  # (B, K, D)
                key=target_key,    # (B, T_h, D)
                value=target_value, # (B, T_h, D)
                key_padding_mask=None,
                need_weights=False,
            )  # (B, K, D)

            attn_out = self.layer_norms[i](attn_out)

        risk_query = attn_out

        # The paper does not specify how the K conditioned modes become one
        # bivariate-Gaussian output. Use risk-softmax weights and moment matching.
        mode_weights = torch.softmax(-risk_field.sum(dim=-1), dim=-1)
        mode_sequence = risk_query.reshape(B * self.k, 1, D).expand(-1, T_f, -1)
        mode_features, _ = self.trajectory_lstm(mode_sequence)
        raw = self.trajectory_mlp(mode_features).reshape(B, self.k, T_f, 5)
        mode_mu = raw[..., :2]
        mode_sigma = F.softplus(raw[..., 2:4]) + 1e-6
        mode_rho = torch.tanh(raw[..., 4])

        weights = mode_weights[:, :, None, None]
        mean = (weights * mode_mu).sum(dim=1)
        centered = mode_mu - mean[:, None, :, :]
        var_x = (weights.squeeze(-1) * (mode_sigma[..., 0].square() + centered[..., 0].square())).sum(dim=1)
        var_y = (weights.squeeze(-1) * (mode_sigma[..., 1].square() + centered[..., 1].square())).sum(dim=1)
        covariance = (
            weights.squeeze(-1)
            * (mode_rho * mode_sigma[..., 0] * mode_sigma[..., 1]
               + centered[..., 0] * centered[..., 1])
        ).sum(dim=1)
        sigma_x = torch.sqrt(var_x.clamp(min=1e-12))
        sigma_y = torch.sqrt(var_y.clamp(min=1e-12))
        rho = (covariance / (sigma_x * sigma_y).clamp(min=1e-12)).clamp(-0.999, 0.999)
        traj_dist = torch.stack([mean[..., 0], mean[..., 1], sigma_x, sigma_y, rho], dim=-1)

        return traj_dist, risk_field
