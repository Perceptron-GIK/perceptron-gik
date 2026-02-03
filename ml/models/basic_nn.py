"""
Neural Network Framework for GIK (Gesture Input Keyboard)

This module provides:
1. GIKDataset - PyTorch Dataset that aligns IMU data with keyboard events
2. GIKModelWrapper - Wraps any nn.Module with input projection and classification head
3. GIKTrainer - Training utilities for the model

Supports single-hand (right or left) or dual-hand training configurations.

Usage:
    # Define your custom inner model
    class MyModel(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return out[:, -1, :]  # Return last timestep
    
    # Example 1: Dual-hand training
    dataset = GIKDataset(
        keyboard_csv="data/Keyboard_2.csv",
        right_csv="data/Right_2.csv",
        left_csv="data/Left_2.csv"
    )
    model = create_model_from_dataset(dataset, 'lstm', hidden_dim=128)
    
    # Example 2: Single-hand training (right hand only)
    dataset = GIKDataset(
        keyboard_csv="data/Keyboard_2.csv",
        right_csv="data/Right_2.csv"
        # left_csv is omitted
    )
    model = create_model_from_dataset(dataset, 'lstm', hidden_dim=128)
    
    # Example 3: Single-hand training (left hand only)
    dataset = GIKDataset(
        keyboard_csv="data/Keyboard_2.csv",
        left_csv="data/Left_2.csv"
        # right_csv is omitted
    )
    model = create_model_from_dataset(dataset, 'lstm', hidden_dim=128)
    
    # Train
    trainer = GIKTrainer(model, dataset)
    trainer.train(epochs=100)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import os


# ============================================================================
# Constants and Configuration
# ============================================================================

# Number of output classes (a-z: 26, 0-9: 10, space, enter, backspace, tab = 40)
NUM_CLASSES = 40

# IMU features per hand (excluding sample_id and timestamp)
# base (6) + thumb (7) + index (7) + middle (7) + ring (7) + pinky (7) = 41
FEATURES_PER_HAND = 41

# Total features when combining both hands
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 82 features

# Scan code to character mapping (based on typical keyboard scan codes)
SCAN_CODE_TO_CHAR = {
    # Letters (scan codes vary by keyboard, these are common)
    0: 'a', 1: 's', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'j', 7: 'k', 8: 'l',
    9: 'q', 10: 'w', 11: 'e', 12: 'r', 13: 't', 14: 'y', 15: 'u', 16: 'i', 17: 'o', 18: 'p',
    29: 'z', 30: 'x', 31: 'c', 32: 'v', 33: 'b', 34: 'n', 35: 'm',
    # Additional letter mappings from observed data
    37: 'l', 38: ';', 45: 'n', 46: 'm',
    # Numbers
    19: '1', 20: '2', 21: '3', 22: '4', 23: '5', 24: '6', 25: '7', 26: '8', 27: '9', 28: '0',
    # Special keys
    49: ' ',      # space
    36: '\n',     # enter
    51: '\b',     # backspace/delete
    48: '\t',     # tab
}

# Character to index mapping for one-hot encoding
CHAR_TO_INDEX = {}
idx = 0
# Lowercase letters a-z
for c in 'abcdefghijklmnopqrstuvwxyz':
    CHAR_TO_INDEX[c] = idx
    idx += 1
# Digits 0-9
for c in '0123456789':
    CHAR_TO_INDEX[c] = idx
    idx += 1
# Special characters
CHAR_TO_INDEX[' '] = idx      # space -> 36
CHAR_TO_INDEX['\n'] = idx + 1  # enter -> 37
CHAR_TO_INDEX['\b'] = idx + 2  # backspace -> 38
CHAR_TO_INDEX['\t'] = idx + 3  # tab -> 39

INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}


# ============================================================================
# Dataset Class
# ============================================================================

class GIKDataset(Dataset):
    """
    PyTorch Dataset for GIK data.
    
    Aligns IMU data from one or both hands with keyboard events to create
    labeled sequences for character prediction.
    
    Each sample consists of:
    - Features: IMU data from available hand(s) between consecutive key presses
    - Label: One-hot encoded character that was pressed
    
    Supports three modes:
    - Both hands: Provide both right_csv and left_csv
    - Right hand only: Provide only right_csv
    - Left hand only: Provide only left_csv
    
    Args:
        keyboard_csv: Path to keyboard events CSV file (required)
        right_csv: Path to right hand IMU CSV file (optional)
        left_csv: Path to left hand IMU CSV file (optional)
        max_seq_length: Maximum sequence length (pads/truncates to this)
        normalize: Whether to normalize IMU features
        
    Raises:
        ValueError: If neither right_csv nor left_csv is provided
    
    Properties:
        input_dim: Number of input features (41 per hand)
        num_hands: Number of hands being used (1 or 2)
        has_right: Whether right hand data is available
        has_left: Whether left hand data is available
    """
    
    def __init__(
        self,
        keyboard_csv: str,
        right_csv: Optional[str] = None,
        left_csv: Optional[str] = None,
        max_seq_length: int = 100,
        normalize: bool = True
    ):
        # Validate inputs - at least one hand CSV is required
        if right_csv is None and left_csv is None:
            raise ValueError("At least one of right_csv or left_csv must be provided")
        
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        
        # Track which hands are available
        self.has_right = right_csv is not None
        self.has_left = left_csv is not None
        self.num_hands = int(self.has_right) + int(self.has_left)
        
        # Load raw data
        self.right_df = pd.read_csv(right_csv) if self.has_right else None
        self.left_df = pd.read_csv(left_csv) if self.has_left else None
        self.keyboard_df = pd.read_csv(keyboard_csv)
        
        # Calculate input dimension based on available hands
        self._input_dim = FEATURES_PER_HAND * self.num_hands
        
        # Process and align data
        self.samples, self.labels = self._process_data()
        
        # Compute normalization stats if needed
        if self.normalize and len(self.samples) > 0:
            self._compute_normalization_stats()
    
    @property
    def input_dim(self) -> int:
        """Number of input features based on available hands."""
        return self._input_dim
    
    def _process_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Process raw data to create aligned samples.
        
        Handles single-hand or dual-hand data based on what's available.
        
        Returns:
            Tuple of (samples list, labels list)
        """
        samples = []
        labels = []
        
        # Filter keyboard events to only 'down' events (key press)
        key_events = self.keyboard_df[self.keyboard_df['event_type'] == 'down'].copy()
        key_events = key_events.sort_values('time').reset_index(drop=True)
        
        # Get IMU feature columns and sorted data for available hands
        right_feature_cols = None
        left_feature_cols = None
        right_sorted = None
        left_sorted = None
        
        if self.has_right:
            right_feature_cols = [c for c in self.right_df.columns 
                                 if c not in ['sample_id', 'time_stamp']]
            right_sorted = self.right_df.sort_values('time_stamp').reset_index(drop=True)
        
        if self.has_left:
            left_feature_cols = [c for c in self.left_df.columns 
                                if c not in ['sample_id', 'time_stamp']]
            left_sorted = self.left_df.sort_values('time_stamp').reset_index(drop=True)
        
        # For each consecutive pair of key presses, extract IMU data
        for i in range(len(key_events) - 1):
            current_event = key_events.iloc[i]
            next_event = key_events.iloc[i + 1]
            
            # Get the character for the NEXT key press (what we're predicting)
            scan_code = int(next_event['scan_code'])
            char = self._scan_code_to_char(scan_code)
            
            # Skip if character is not in our vocabulary
            if char not in CHAR_TO_INDEX:
                continue
            
            label = CHAR_TO_INDEX[char]
            
            # Get time window
            start_time = current_event['time']
            end_time = next_event['time']
            
            # Extract IMU data in this time window for available hands
            right_window = None
            left_window = None
            
            if self.has_right:
                right_window = right_sorted[
                    (right_sorted['time_stamp'] >= start_time) & 
                    (right_sorted['time_stamp'] < end_time)
                ][right_feature_cols].values
                
                # Skip if right hand required but no data
                if len(right_window) == 0:
                    continue
            
            if self.has_left:
                left_window = left_sorted[
                    (left_sorted['time_stamp'] >= start_time) & 
                    (left_sorted['time_stamp'] < end_time)
                ][left_feature_cols].values
                
                # Skip if left hand required but no data
                if len(left_window) == 0:
                    continue
            
            # Combine/process hand data based on availability
            combined = self._combine_hand_data(right_window, left_window)
            
            if combined is not None:
                samples.append(torch.tensor(combined, dtype=torch.float32))
                labels.append(label)
        
        return samples, labels
    
    def _scan_code_to_char(self, scan_code: int) -> str:
        """Convert scan code to character."""
        return SCAN_CODE_TO_CHAR.get(scan_code, '')
    
    def _combine_hand_data(
        self, 
        right_data: Optional[np.ndarray] = None, 
        left_data: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Combine or process hand IMU data.
        
        Handles three cases:
        - Both hands: Concatenates features after interpolation alignment
        - Right hand only: Returns interpolated right hand data
        - Left hand only: Returns interpolated left hand data
        
        Uses linear interpolation to align sequences of different lengths.
        
        Args:
            right_data: Right hand IMU data (seq_len_r, features) or None
            left_data: Left hand IMU data (seq_len_l, features) or None
            
        Returns:
            Processed data (seq_len, features) or None if invalid
        """
        # Determine available data
        has_right = right_data is not None and len(right_data) > 0
        has_left = left_data is not None and len(left_data) > 0
        
        # Need at least one hand's data
        if not has_right and not has_left:
            return None
        
        # Single hand cases
        if has_right and not has_left:
            # Right hand only
            target_len = min(len(right_data), self.max_seq_length)
            return self._interpolate_sequence(right_data, target_len)
        
        if has_left and not has_right:
            # Left hand only
            target_len = min(len(left_data), self.max_seq_length)
            return self._interpolate_sequence(left_data, target_len)
        
        # Both hands - align and concatenate
        target_len = max(len(right_data), len(left_data))
        target_len = min(target_len, self.max_seq_length)
        
        # Interpolate both to target length
        right_interp = self._interpolate_sequence(right_data, target_len)
        left_interp = self._interpolate_sequence(left_data, target_len)
        
        # Concatenate features (right first, then left)
        combined = np.concatenate([right_interp, left_interp], axis=1)
        
        return combined
    
    def _interpolate_sequence(
        self, 
        data: np.ndarray, 
        target_len: int
    ) -> np.ndarray:
        """
        Interpolate sequence to target length.
        
        Args:
            data: Input sequence (seq_len, features)
            target_len: Target sequence length
            
        Returns:
            Interpolated sequence (target_len, features)
        """
        current_len = len(data)
        
        if current_len == target_len:
            return data
        
        # Create interpolation indices
        old_indices = np.linspace(0, current_len - 1, current_len)
        new_indices = np.linspace(0, current_len - 1, target_len)
        
        # Interpolate each feature
        interpolated = np.zeros((target_len, data.shape[1]))
        for f in range(data.shape[1]):
            interpolated[:, f] = np.interp(new_indices, old_indices, data[:, f])
        
        return interpolated
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        all_data = torch.cat([s for s in self.samples], dim=0)
        self.mean = all_data.mean(dim=0)
        self.std = all_data.std(dim=0)
        self.std[self.std == 0] = 1.0  # Avoid division by zero
    
    def _normalize_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Normalize a sample using precomputed stats."""
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            return (sample - self.mean) / self.std
        return sample
    
    def _pad_sequence(self, sample: torch.Tensor) -> torch.Tensor:
        """Pad or truncate sequence to max_seq_length."""
        seq_len = sample.shape[0]
        
        if seq_len >= self.max_seq_length:
            return sample[:self.max_seq_length]
        
        # Pad with zeros
        padding = torch.zeros(self.max_seq_length - seq_len, sample.shape[1])
        return torch.cat([sample, padding], dim=0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample and its label.
        
        Returns:
            Tuple of (features, one_hot_label)
            - features: (max_seq_length, num_features) tensor
            - one_hot_label: (num_classes,) tensor
        """
        sample = self.samples[idx]
        
        # Normalize if enabled
        if self.normalize:
            sample = self._normalize_sample(sample)
        
        # Pad to fixed length
        sample = self._pad_sequence(sample)
        
        # Create one-hot label
        label = torch.zeros(NUM_CLASSES)
        label[self.labels[idx]] = 1.0
        
        return sample, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        
        Returns:
            Tensor of shape (num_classes,) with inverse frequency weights
        """
        counts = torch.zeros(NUM_CLASSES)
        for label in self.labels:
            counts[label] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * NUM_CLASSES  # Normalize
        
        return weights


# ============================================================================
# Model Wrapper Class
# ============================================================================

class GIKModelWrapper(nn.Module):
    """
    Wrapper that adds input projection and classification head to any model.
    
    This class provides a standardized interface for GIK character prediction:
    1. Input projection layer: Projects raw features to hidden dimension
    2. Inner model: Any nn.Module that processes sequences
    3. Classification head: Projects to num_classes outputs
    
    The inner model should:
    - Accept input of shape (batch, seq_len, hidden_dim)
    - Return output of shape (batch, hidden_dim) or (batch, seq_len, hidden_dim)
    
    Args:
        inner_model: The core neural network (LSTM, Transformer, CNN, etc.)
        input_dim: Number of input features (default: 82 for both hands)
        hidden_dim: Hidden dimension size for the inner model
        num_classes: Number of output classes (default: 40)
        dropout: Dropout probability for classification head
        pool_output: If True, applies global average pooling to sequence output
    
    Example:
        >>> class SimpleLSTM(nn.Module):
        ...     def __init__(self, hidden_dim):
        ...         super().__init__()
        ...         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        ...     def forward(self, x):
        ...         out, _ = self.lstm(x)
        ...         return out[:, -1, :]  # Return last timestep
        >>> 
        >>> inner = SimpleLSTM(hidden_dim=128)
        >>> model = GIKModelWrapper(inner, hidden_dim=128)
    """
    
    def __init__(
        self,
        inner_model: nn.Module,
        input_dim: int = TOTAL_FEATURES,
        hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pool_output: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pool_output = pool_output
        
        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Inner model (user-provided)
        self.inner_model = inner_model
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Project input features to hidden dimension
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        x = self.input_projection(x)
        
        # Pass through inner model
        x = self.inner_model(x)
        
        # Handle different output shapes
        if len(x.shape) == 3:
            if self.pool_output:
                # Global average pooling over sequence
                x = x.mean(dim=1)
            else:
                # Take last timestep
                x = x[:, -1, :]
        
        # Classification head
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Predicted class indices of shape (batch,)
        """
        logits = self.forward(x)
        return logits.argmax(dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Probabilities of shape (batch, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# ============================================================================
# Pre-built Inner Models
# ============================================================================

class LSTMModel(nn.Module):
    """Simple LSTM model for sequence processing."""
    
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
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project back to hidden_dim if bidirectional
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(out_dim, hidden_dim) if bidirectional else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, hidden_dim)
        """
        out, (h_n, c_n) = self.lstm(x)
        
        # Take the last output
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            last_out = torch.cat([out[:, -1, :self.hidden_dim], 
                                  out[:, 0, self.hidden_dim:]], dim=-1)
        else:
            last_out = out[:, -1, :]
        
        return self.projection(last_out)


class TransformerModel(nn.Module):
    """Transformer encoder for sequence processing."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
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
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Return CLS token representation
        return x[:, 0, :]


class CNNModel(nn.Module):
    """1D CNN for sequence processing."""
    
    def __init__(
        self,
        hidden_dim: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.2
    ):
        super().__init__()
        
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
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, hidden_dim)
        """
        # Transpose for Conv1d: (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply each conv and global max pool
        conv_outputs = []
        for conv in self.convs:
            c = conv(x)  # (batch, hidden_dim, seq_len)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (batch, hidden_dim)
            conv_outputs.append(c)
        
        # Concatenate and project
        x = torch.cat(conv_outputs, dim=-1)
        return self.projection(x)


# ============================================================================
# Trainer Class
# ============================================================================

class GIKTrainer:
    """
    Training utilities for GIK models.
    
    Args:
        model: GIKModelWrapper instance
        dataset: GIKDataset instance
        device: Device to train on ('cuda', 'mps', or 'cpu')
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization
        batch_size: Training batch size
        val_split: Fraction of data for validation
    """
    
    def __init__(
        self,
        model: GIKModelWrapper,
        dataset: GIKDataset,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        val_split: float = 0.2
    ):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        
        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Use class weights for imbalanced data
        class_weights = dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            
            # Compute loss (CrossEntropyLoss expects class indices, not one-hot)
            loss = self.criterion(logits, batch_y.argmax(dim=-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == batch_y.argmax(dim=-1)).sum().item()
            total += batch_x.size(0)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y.argmax(dim=-1))
            
            total_loss += loss.item() * batch_x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == batch_y.argmax(dim=-1)).sum().item()
            total += batch_x.size(0)
        
        return total_loss / total, correct / total
    
    def train(
        self, 
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        save_path: str = 'best_model.pt'
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            epochs: Number of training epochs
            early_stopping_patience: Stop if no improvement for this many epochs
            save_best: Whether to save the best model
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("-" * 60)
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        
        return self.history
    
    def load_best_model(self, path: str = 'best_model.pt'):
        """Load the best saved model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from {path}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_model(
    model_type: str = 'lstm',
    hidden_dim: int = 128,
    input_dim: Optional[int] = None,
    num_hands: int = 2,
    **kwargs
) -> GIKModelWrapper:
    """
    Factory function to create a GIK model.
    
    Args:
        model_type: One of 'lstm', 'transformer', 'cnn'
        hidden_dim: Hidden dimension size
        input_dim: Number of input features. If None, calculated from num_hands
        num_hands: Number of hands (1 or 2). Used to calculate input_dim if not provided
        **kwargs: Additional arguments for the inner model
        
    Returns:
        GIKModelWrapper instance
    """
    # Calculate input_dim if not provided
    if input_dim is None:
        input_dim = FEATURES_PER_HAND * num_hands
    
    if model_type == 'lstm':
        inner_model = LSTMModel(hidden_dim, **kwargs)
    elif model_type == 'transformer':
        inner_model = TransformerModel(hidden_dim, **kwargs)
    elif model_type == 'cnn':
        inner_model = CNNModel(hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return GIKModelWrapper(inner_model, input_dim=input_dim, hidden_dim=hidden_dim)


def create_model_from_dataset(
    dataset: GIKDataset,
    model_type: str = 'lstm',
    hidden_dim: int = 128,
    **kwargs
) -> GIKModelWrapper:
    """
    Factory function to create a GIK model with input_dim from dataset.
    
    Automatically configures the model's input dimension based on the dataset,
    making it easy to handle single-hand or dual-hand configurations.
    
    Args:
        dataset: GIKDataset instance to get input_dim from
        model_type: One of 'lstm', 'transformer', 'cnn'
        hidden_dim: Hidden dimension size
        **kwargs: Additional arguments for the inner model
        
    Returns:
        GIKModelWrapper instance configured for the dataset
    """
    return create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        input_dim=dataset.input_dim,
        **kwargs
    )


def decode_predictions(
    predictions: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None
) -> List[Dict[str, Any]]:
    """
    Decode model predictions to characters.
    
    Args:
        predictions: Tensor of predicted class indices
        probabilities: Optional tensor of class probabilities
        
    Returns:
        List of dicts with 'char' and optionally 'confidence'
    """
    results = []
    for i, pred in enumerate(predictions):
        result = {'char': INDEX_TO_CHAR.get(pred.item(), '?')}
        if probabilities is not None:
            result['confidence'] = probabilities[i, pred].item()
        results.append(result)
    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example: Create and train a model
    print("GIK Neural Network Framework")
    print("=" * 60)
    
    # Paths to data files (adjust as needed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
    
    right_csv = os.path.join(data_dir, "Right_2.csv")
    left_csv = os.path.join(data_dir, "Left_2.csv")
    keyboard_csv = os.path.join(data_dir, "Keyboard_2.csv")
    
    # Check if keyboard file exists (required)
    if not os.path.exists(keyboard_csv):
        print("Keyboard data file not found. Creating dummy example...")
        
        # Demo 1: Single hand model (right hand only)
        print("\n--- Demo: Single Hand Model (Right) ---")
        inner_model = LSTMModel(hidden_dim=128)
        model_single = GIKModelWrapper(
            inner_model, 
            input_dim=FEATURES_PER_HAND,  # 41 features for one hand
            hidden_dim=128
        )
        print(f"Single hand input dim: {FEATURES_PER_HAND}")
        
        dummy_single = torch.randn(4, 100, FEATURES_PER_HAND)
        output_single = model_single(dummy_single)
        print(f"Input shape: {dummy_single.shape}")
        print(f"Output shape: {output_single.shape}")
        
        # Demo 2: Dual hand model
        print("\n--- Demo: Dual Hand Model ---")
        inner_model2 = LSTMModel(hidden_dim=128)
        model_dual = GIKModelWrapper(
            inner_model2, 
            input_dim=TOTAL_FEATURES,  # 82 features for both hands
            hidden_dim=128
        )
        print(f"Dual hand input dim: {TOTAL_FEATURES}")
        
        dummy_dual = torch.randn(4, 100, TOTAL_FEATURES)
        output_dual = model_dual(dummy_dual)
        print(f"Input shape: {dummy_dual.shape}")
        print(f"Output shape: {output_dual.shape}")
        
    else:
        print(f"Loading data from {data_dir}")
        
        # Determine available data files
        has_right = os.path.exists(right_csv)
        has_left = os.path.exists(left_csv)
        
        print(f"Right hand data: {'Found' if has_right else 'Not found'}")
        print(f"Left hand data: {'Found' if has_left else 'Not found'}")
        
        if not has_right and not has_left:
            print("Error: At least one hand CSV file is required.")
        else:
            # Create dataset with available hand data
            # keyboard_csv is first argument (required), hand CSVs are optional
            dataset = GIKDataset(
                keyboard_csv=keyboard_csv,
                right_csv=right_csv if has_right else None,
                left_csv=left_csv if has_left else None,
                max_seq_length=100
            )
            
            print(f"\nDataset Configuration:")
            print(f"  - Hands: {dataset.num_hands} ({'right' if dataset.has_right else ''}"
                  f"{' + ' if dataset.has_right and dataset.has_left else ''}"
                  f"{'left' if dataset.has_left else ''})")
            print(f"  - Input features: {dataset.input_dim}")
            print(f"  - Samples: {len(dataset)}")
            
            if len(dataset) > 0:
                # Create model using the dataset's input_dim
                model = create_model_from_dataset(dataset, 'lstm', hidden_dim=128)
                
                print(f"\nModel created with input_dim={dataset.input_dim}")
                
                # Create trainer
                trainer = GIKTrainer(
                    model=model,
                    dataset=dataset,
                    batch_size=16,
                    learning_rate=1e-3
                )
                
                # Train
                history = trainer.train(epochs=50, early_stopping_patience=10)
                
                # Plot training history
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                axes[0].plot(history['train_loss'], label='Train')
                axes[0].plot(history['val_loss'], label='Validation')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].set_title('Training Loss')
                
                axes[1].plot(history['train_acc'], label='Train')
                axes[1].plot(history['val_acc'], label='Validation')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].set_title('Training Accuracy')
                
                plt.tight_layout()
                plt.savefig(os.path.join(data_dir, 'training_history.png'))
                print(f"\nSaved training plot to {os.path.join(data_dir, 'training_history.png')}")
