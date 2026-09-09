"""Goal Predictor: MLP for predicting surrounding vehicle future goals."""
import torch
import torch.nn as nn


class GoalPredictor(nn.Module):
    """Predicts final position and velocity goals for surrounding vehicles.

    Takes encoded neighbor features and outputs goal predictions.
    Input: (B, T_h, N_v, D) encoded neighbor features
    Output: (B, N_v, 4) predicted goals [x, y, vx, vy]
    """

    def __init__(self, d_model: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self._d_model = d_model
        # Time-agnostic: mean-pool over T_h then MLP (handles any T_h,
        # e.g. T_h=5 in unit tests and T_h=30 in full pipeline).
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # [x, y, vx, vy]
        )

    def forward(self, encoded_neighbors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded_neighbors: (B, T_h, N_v, D) encoded features for neighbors
        Returns:
            goals: (B, N_v, 4) predicted goals [x, y, vx, vy]
        """
        B, T_h, N_v, D = encoded_neighbors.shape
        pooled = encoded_neighbors.mean(dim=1)  # (B, N_v, D) temporal mean
        flat = pooled.reshape(B * N_v, D)
        goals = self.mlp(flat)  # (B*N_v, 4)
        goals = goals.view(B, N_v, 4)
        return goals
