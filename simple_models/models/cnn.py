"""Simple 1D CNN classifier for character prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassifier(nn.Module):
    """1D CNN acting directly on (time, features). D features = channels, T = sequence."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 40,
        hidden_dim: int = 64,
        kernel_sizes=(3, 5, 7),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, k, padding=k // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for k in kernel_sizes
            ]
        )
        self.head = nn.Linear(hidden_dim * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (B, T, D) -> transpose to (B, D, T)
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            c = conv(x)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)
            conv_outputs.append(c)
        x = torch.cat(conv_outputs, dim=-1)
        return self.head(x)
