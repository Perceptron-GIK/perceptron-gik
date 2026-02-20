import torch
from torch import nn
from typing import List

class CNNModel(nn.Module):
    """1D CNN for sequence processing."""
    
    def __init__(
        self,
        hidden_dim: int,
        kernel_sizes: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, k, padding=k//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])
        
        self.projection = nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            c = conv(x)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)
            conv_outputs.append(c)
        
        x = torch.cat(conv_outputs, dim=-1)
        return self.projection(x)