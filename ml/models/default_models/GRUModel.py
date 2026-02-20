import torch
from torch import nn

class GRUModel(nn.Module):
    """Bidirectional GRU for sequence processing."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(out_dim, hidden_dim) if bidirectional else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        if self.bidirectional:
            last_out = torch.cat([out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]], dim=-1)
        else:
            last_out = out[:, -1, :]
        return self.projection(last_out)