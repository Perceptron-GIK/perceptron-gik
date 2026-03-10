"""CNN-LSTM classifier: 2 ResNet blocks (feature extractor) + LSTM + FC."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock1D(nn.Module):
    """1D ResNet block: two Conv1d layers with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class SimpleCNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM: 2 ResNet blocks (feature extractor) -> LSTM -> dropout -> FC.
    Input: (B, T, D). ResNet acts on (B, D, T); LSTM on (B, T, hidden).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 40,
        cnn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # 2 ResNet blocks: input_dim -> cnn_hidden_dim -> cnn_hidden_dim
        self.resblock1 = ResNetBlock1D(input_dim, cnn_hidden_dim, kernel_size=3)
        self.resblock2 = ResNetBlock1D(cnn_hidden_dim, cnn_hidden_dim, kernel_size=3)

        self.lstm = nn.LSTM(
            cnn_hidden_dim,
            lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.resblock1(x)
        x = self.resblock2(x)
        # (B, cnn_hidden_dim, T) -> (B, T, cnn_hidden_dim) for LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out)
