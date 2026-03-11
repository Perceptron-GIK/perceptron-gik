import torch
from torch import nn


class TransformerModel(nn.Module):
    """Transformer encoder for sequence processing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_projection = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)          # (B, T, hidden_dim)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1) # (B, T+1, hidden_dim)
        x = self.transformer(x)
        return x[:, 0, :]                     # (B, hidden_dim)
