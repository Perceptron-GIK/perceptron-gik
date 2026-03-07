import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _SpatialTemporalAttentionA(nn.Module):
    """
    Attention Module A — temporal focus.

    Identifies *which timesteps* carry the most discriminative information by
    computing a per-timestep scalar weight map M_a over the time axis.

    Paper equations (1) & (2):
        M_a = sigmoid(relu(BN(Conv2d(F))))     shape: (B, 1, T, 1)
        F'  = M_a ⊗ F                          element-wise broadcast

    Input / output shape: (B, T, H)  →  (B, T, H)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Conv2d with kernel (3,1): looks across 3 adjacent timesteps,
        # collapses the feature axis entirely (kernel height = 1) to produce
        # a single channel weight per timestep.
        self.conv = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        # Treat T as spatial-H and H as spatial-W for a single-channel 2-D conv
        B, T, H = x.shape
        f = x.unsqueeze(1)                          # (B, 1, T, H)
        m_a = torch.sigmoid(F.relu(self.bn(self.conv(f))))  # (B, 1, T, H)
        # Average over the feature axis to get a per-timestep scalar weight
        m_a = m_a.mean(dim=-1, keepdim=True)        # (B, 1, T, 1)
        # Broadcast over H and multiply
        f_prime = f * m_a                           # (B, 1, T, H)
        return f_prime.squeeze(1)                   # (B, T, H)


class _ResidualBlock1D(nn.Module):
    """
    Single 1-D residual conv block.

    Applies two Conv1d layers with a skip connection.  If in_channels ≠
    out_channels a 1×1 projection is added to the residual path so dimensions
    always match.

    Block:  Conv1d → BN → ReLU → Dropout → Conv1d → BN  ⊕  skip  → ReLU
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.downsample(x))


class _ResNetBlocks(nn.Module):
    """
    Stack of 1-D ResNet blocks.

    Paper: three Conv layers with 128, 256, and 512 feature maps.
    Here we use hidden_dim as the final channel count, with intermediate
    channels scaled to 0.5× and 1× of hidden_dim (matching the paper's
    progressive widening pattern).

    Input / output: (B, H, T)  →  (B, H, T)   (channels-first convention)
    """

    def __init__(self, hidden_dim: int, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        mid = max(hidden_dim // 2, 16)

        # Build a channel plan of length (num_blocks + 1):
        #   first entry = hidden_dim (input channels)
        #   intermediate entries = mid (hidden channels)
        #   last entry = hidden_dim (output channels)
        if num_blocks == 1:
            channel_plan = [hidden_dim, hidden_dim]
        else:
            channel_plan = [hidden_dim] + [mid] * (num_blocks - 1) + [hidden_dim]

        blocks: List[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(
                _ResidualBlock1D(channel_plan[i], channel_plan[i + 1],
                                 dropout=dropout)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _SpatialTemporalAttentionB(nn.Module):
    """
    Attention Module B — cross-channel (feature/sensor) focus.

    Identifies *which feature channels (IMU axes / sensors)* are most
    informative by producing a per-channel scalar weight map M_b.

    Paper equations (4) & (5):
        M_b = sigmoid( MLP(MaxPool(F'')) + MLP(AvgPool(F'')) )
        F''' = M_b ⊗ F''

    The two pooling branches share MLP weights (channel-wise squeeze-excite).

    Input / output: (B, H, T)  →  (B, H)   (temporal average pooled out)
    """

    def __init__(self, hidden_dim: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(hidden_dim // reduction, 8)
        # Shared MLP applied to each pooling descriptor independently
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_dim),
        )
        self.avg_pool_final = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T)  – channels-first from ResNet
        d_max = x.max(dim=-1).values           # (B, H)
        d_avg = x.mean(dim=-1)                 # (B, H)
        m_b = torch.sigmoid(self.mlp(d_max) + self.mlp(d_avg))  # (B, H)
        # Reweight channels, then pool time
        f_triple = x * m_b.unsqueeze(-1)       # (B, H, T)
        out = self.avg_pool_final(f_triple).squeeze(-1)  # (B, H)
        return out


class CNNSTRNet(nn.Module):
    """
    CNN-STRNet inner model — a faithful adaptation of the GloveTyping paper's
    CNN + Spatial-Temporal Residual Network.

    Designed as a drop-in inner model for GIKModelWrapper, i.e.:
        input:  (B, T, H)   — after GIKModelWrapper.input_projection
        output: (B, H)      — fed into GIKModelWrapper.project_from_inner

    Architecture (four stages):

    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 1 – CNN Feature Extraction                               │
    │  Conv1d(H→H, k=1) to enrich per-timestep features              │
    ├─────────────────────────────────────────────────────────────────┤
    │  Stage 2 – Spatial-Temporal Attention A  (temporal focus)       │
    │  Conv2d on (B,1,T,H) → M_a per-timestep weight → F' = M_a ⊗ F │
    ├─────────────────────────────────────────────────────────────────┤
    │  Stage 3 – 1-D ResNet Blocks                                    │
    │  num_res_blocks of (Conv1d→BN→ReLU→Conv1d→BN ⊕ skip) → F''    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Stage 4 – Spatial-Temporal Attention B  (channel focus)        │
    │  MaxPool+AvgPool → MLP → M_b per-channel weight → AvgPool time  │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        hidden_dim:     Feature width H (must match GIKModelWrapper inner_model_dim).
        num_res_blocks: Number of 1-D residual blocks (default 3, paper uses 3).
        dropout:        Dropout rate inside residual blocks (default 0.2).
        attn_reduction: Bottleneck reduction ratio for Attention B MLP (default 4).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_res_blocks: int = 3,
        dropout: float = 0.2,
        attn_reduction: int = 4,
        input_dim: Optional[int] = None,
        sensor_input_dim: Optional[int] = None,
        fsr_feature_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.raw_input_mode = input_dim is not None

        self.sensor_input_dim = sensor_input_dim if sensor_input_dim is not None else input_dim
        if self.raw_input_mode and self.sensor_input_dim is None:
            raise ValueError("sensor_input_dim cannot be None when input_dim is provided")

        self.context_input_dim = 0
        if self.raw_input_mode:
            self.context_input_dim = max(0, int(input_dim) - int(self.sensor_input_dim))

        self.fsr_feature_indices: List[int] = []
        self.non_fsr_feature_indices: List[int] = []
        if self.raw_input_mode:
            all_sensor_idx = list(range(int(self.sensor_input_dim)))
            fsr_idx = sorted(set(fsr_feature_indices or []))
            fsr_idx = [i for i in fsr_idx if 0 <= i < int(self.sensor_input_dim)]
            self.fsr_feature_indices = fsr_idx
            fsr_set = set(fsr_idx)
            self.non_fsr_feature_indices = [i for i in all_sensor_idx if i not in fsr_set]

            self.register_buffer(
                "_fsr_idx_tensor",
                torch.tensor(self.fsr_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_nonfsr_idx_tensor",
                torch.tensor(self.non_fsr_feature_indices, dtype=torch.long),
                persistent=False,
            )

            non_fsr_dim = len(self.non_fsr_feature_indices)
            fsr_dim = len(self.fsr_feature_indices)
            context_dim = self.context_input_dim
            sensor_fusion_in = non_fsr_dim + fsr_dim + context_dim
            if sensor_fusion_in <= 0:
                raise ValueError("CNNSTRNet raw_input_mode requires at least one input feature")
            self.raw_fusion = nn.Sequential(
                nn.Linear(sensor_fusion_in, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

        # Stage 1: CNN feature extraction (kernel=1 → pointwise across channels per timestep)
        self.cnn_feature = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Stage 2: Spatial-Temporal Attention A
        self.attn_a = _SpatialTemporalAttentionA(hidden_dim)

        # Stage 3: 1-D ResNet stack (operates in channels-first: B, H, T)
        self.resnet = _ResNetBlocks(hidden_dim, num_blocks=num_res_blocks,
                                    dropout=dropout)

        # Stage 4: Spatial-Temporal Attention B + global pool
        self.attn_b = _SpatialTemporalAttentionB(hidden_dim,
                                                  reduction=attn_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        if self.raw_input_mode:
            # x shape is (B, T, F_total) where:
            #   - first sensor_input_dim features are sensor channels
            #   - optional trailing features are context (e.g., prev-char embedding)
            x_sensor = x[..., :self.sensor_input_dim]
            x_context = x[..., self.sensor_input_dim:] if self.context_input_dim > 0 else None

            parts = []
            if self._nonfsr_idx_tensor.numel() > 0:
                parts.append(x_sensor.index_select(dim=-1, index=self._nonfsr_idx_tensor))
            if self._fsr_idx_tensor.numel() > 0:
                # Explicitly keep FSR channels in the inner block pipeline.
                parts.append(x_sensor.index_select(dim=-1, index=self._fsr_idx_tensor))
            if x_context is not None and x_context.shape[-1] > 0:
                parts.append(x_context)

            x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            x = self.raw_fusion(x)

        # --- Stage 1: CNN feature extraction ---
        # Conv1d expects (B, C, L) so transpose to channels-first
        f = self.cnn_feature(x.transpose(1, 2))   # (B, H, T)
        f = f.transpose(1, 2)                      # (B, T, H)

        # --- Stage 2: Attention A — temporal weighting ---
        f_prime = self.attn_a(f)                   # (B, T, H)

        # --- Stage 3: ResNet blocks — channels-first ---
        f_double = self.resnet(f_prime.transpose(1, 2))  # (B, H, T)

        # --- Stage 4: Attention B — channel weighting + temporal pool ---
        out = self.attn_b(f_double)                # (B, H)

        return out
