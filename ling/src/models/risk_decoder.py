"""Risk-Attentive Feature Fusion Decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Compute risk for each intention mode
        # For each mode k, compute risk based on neighbor goals and intention
        # Risk = sum of S-field + O-field between target intention and neighbors
        # Simplified: use neighbor goals + intention to compute future risk

        # Compute S-field for each intention-neighbor pair
        # intention: (B, K, 4) -> (B, K, 2) [x, y]
        # neighbor_goals: (B, N_v, 4) -> (B, N_v, 2) [x, y]
        intention_pos = intentions[:, :, :2]  # (B, K, 2)
        neighbor_pos = neighbor_goals[:, :, :2]  # (B, N_v, 2)

        # Expand for pairwise computation
        int_exp = intention_pos.unsqueeze(2)  # (B, K, 1, 2)
        neigh_exp = neighbor_pos.unsqueeze(1)  # (B, 1, N_v, 2)
        delta = int_exp - neigh_exp  # (B, K, N_v, 2)

        delta_x = delta[..., 0]  # (B, K, N_v)
        delta_y = delta[..., 1]

        # S-field (same parameters as config)
        s_field = torch.exp(-(
            (delta_x / 1.0).abs() ** 2 + (delta_y / 1.0).abs() ** 2
        ))  # (B, K, N_v)

        # O-field approximation using distance
        dist = torch.norm(delta, dim=-1).clamp(min=0.1)  # (B, K, N_v)
        o_field = torch.exp(-(dist / 5.0) ** 1.0)  # (B, K, N_v)

        # Sum over neighbors
        R_s = s_field.sum(dim=-1)  # (B, K)
        R_o = o_field.sum(dim=-1)  # (B, K)

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

        # Trajectory generation
        # Use the first (highest attention) risk query as the trajectory seed
        # Actually, we need to generate a trajectory per intention
        # For simplicity, use the mean of the risk-attended features
        # Or use the lowest-risk intention's attended features

        # Weighted combination by inverse risk (lower risk = higher weight)
        risk_sum = risk_field.sum(dim=-1).clamp(min=1e-6)  # (B, K)
        weights = 1.0 / risk_sum  # (B, K)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize

        # Weighted sum of attended features
        weighted_features = torch.bmm(
            weights.unsqueeze(1),  # (B, 1, K)
            risk_query  # (B, K, D)
        )  # (B, 1, D)

        # LSTM trajectory generator
        lstm_input = weighted_features.expand(-1, T_f, -1)  # (B, T_f, D)
        lstm_out, _ = self.trajectory_lstm(lstm_input)  # (B, T_f, D)

        # MLP to produce distribution parameters
        traj_dist = self.trajectory_mlp(lstm_out)  # (B, T_f, 5)

        # Ensure sigma > 0 and rho in (-1, 1)
        mu_x = traj_dist[:, :, 0]
        mu_y = traj_dist[:, :, 1]
        sigma_x = F.softplus(traj_dist[:, :, 2]) + 1e-6
        sigma_y = F.softplus(traj_dist[:, :, 3]) + 1e-6
        rho = torch.tanh(traj_dist[:, :, 4])

        traj_dist = torch.stack([mu_x, mu_y, sigma_x, sigma_y, rho], dim=-1)

        return traj_dist, risk_field
