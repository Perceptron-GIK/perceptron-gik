"""Simple LSTM classifier for character prediction."""

import torch.nn as nn


class SimpleLSTMClassifier(nn.Module):
    """LSTM acting directly on (T, D). input_size=D (features per timestep)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 40,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
