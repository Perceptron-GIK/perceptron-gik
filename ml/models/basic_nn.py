"""
GIK Neural Network Models and Training

Models: transformer (best: 24.5% test acc), attention_lstm, lstm, gru, rnn, cnn

Usage:
    from pretraining import load_preprocessed_dataset
    from ml.models.basic_nn import create_model_from_dataset, GIKTrainer
    
    dataset = load_preprocessed_dataset("data/processed.pt")
    model = create_model_from_dataset(dataset, 'transformer', hidden_dim=128)
    trainer = GIKTrainer(model, dataset, learning_rate=5e-4)
    trainer.train(epochs=50)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch import optim
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import os
import sys

# Add parent directory to path to import pretraining
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from pretraining import (
        PreprocessedGIKDataset,
        load_preprocessed_dataset,
    )
except ImportError:
    PreprocessedGIKDataset = None
    load_preprocessed_dataset = None

# Fixed constants
NUM_CLASSES = 40

# Character mappings
CHAR_TO_INDEX = {}
idx = 0
for c in 'abcdefghijklmnopqrstuvwxyz':
    CHAR_TO_INDEX[c] = idx
    idx += 1
for c in '0123456789':
    CHAR_TO_INDEX[c] = idx
    idx += 1
CHAR_TO_INDEX[' '] = idx
CHAR_TO_INDEX['\n'] = idx + 1
CHAR_TO_INDEX['\b'] = idx + 2
CHAR_TO_INDEX['\t'] = idx + 3
INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}


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
        input_dim: Number of input features (required - get from dataset.input_dim)
        hidden_dim: Hidden dimension size for the inner model
        num_classes: Number of output classes (default: 40)
        dropout: Dropout probability for classification head
        pool_output: If True, applies global average pooling to sequence output
    """
    
    def __init__(
        self,
        inner_model: nn.Module,
        input_dim: int,
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
        """Forward pass."""
        x = self.input_projection(x)
        x = self.inner_model(x)
        
        if len(x.shape) == 3:
            if self.pool_output:
                x = x.mean(dim=1)
            else:
                x = x[:, -1, :]
        
        return self.classifier(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class LSTMModel(nn.Module):
    """Bidirectional LSTM for sequence processing."""
    
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
        
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(out_dim, hidden_dim) if bidirectional else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            last_out = torch.cat([out[:, -1, :self.hidden_dim], 
                                  out[:, 0, self.hidden_dim:]], dim=-1)
        else:
            last_out = out[:, -1, :]
        
        return self.projection(last_out)


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


class RNNModel(nn.Module):
    """Bidirectional Elman RNN for sequence processing."""

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

        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(out_dim, hidden_dim) if bidirectional else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.rnn(x)
        if self.bidirectional:
            last_out = torch.cat([out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]], dim=-1)
        else:
            last_out = out[:, -1, :]
        return self.projection(last_out)


class AttentionLSTM(nn.Module):
    """LSTM with self-attention mechanism (inspired by GloveTyping paper)."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_heads: int = 4
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
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.projection = nn.Linear(lstm_out_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        
        if self.bidirectional:
            last_out = torch.cat([out[:, -1, :self.hidden_dim], 
                                  out[:, 0, self.hidden_dim:]], dim=-1)
        else:
            last_out = out[:, -1, :]
        
        return self.projection(last_out)


class TransformerModel(nn.Module):
    """Transformer encoder for sequence processing. Best performing model (24.5% test acc)."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        return x[:, 0, :]


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


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance. Best performance with gamma=2.0."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class GIKTrainer:
    """
    Training utilities for GIK models.
    
    Uses contiguous train/val/test splits (no shuffle) to preserve causality.
    Uses Focal Loss by default for better handling of class imbalance.
    
    Args:
        model: GIKModelWrapper instance
        dataset: Dataset instance (PreprocessedGIKDataset or compatible)
        device: Device to train on ('cuda', 'mps', or 'cpu')
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization
        batch_size: Training batch size
        train_ratio: Fraction of data for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for test (default 0.1)
        use_focal_loss: Whether to use Focal Loss (recommended for imbalanced data)
        focal_gamma: Gamma parameter for Focal Loss (default 2.0)
    """
    
    def __init__(
        self,
        model: GIKModelWrapper,
        dataset: Dataset,
        device: Optional[str] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
    ):
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
        
        # Contiguous splits (causality-preserving, no shuffle)
        n = len(dataset)
        t = max(0, int(n * train_ratio))
        v = max(t, t + int(n * val_ratio))
        self.train_dataset = Subset(dataset, range(0, t))
        self.val_dataset = Subset(dataset, range(t, v))
        self.test_dataset = Subset(dataset, range(v, n))
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Use Focal Loss by default (better for imbalanced classes)
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=focal_gamma)
        elif hasattr(dataset, 'get_class_weights'):
            class_weights = dataset.get_class_weights().to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y.argmax(dim=-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
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
    
    @torch.no_grad()
    def evaluate_test(self) -> Tuple[float, float]:
        """Evaluate on the held-out test set (causal split)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in self.test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y.argmax(dim=-1))
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=-1) == batch_y.argmax(dim=-1)).sum().item()
            total += batch_x.size(0)
        return (total_loss / total, correct / total) if total else (0.0, 0.0)
    
    def evaluate(self) -> Tuple[float, float]:
        """Alias for validate()."""
        return self.validate()
    
    def train(
        self, 
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        save_path: str = 'best_model.pt'
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)} (causal split, no shuffle)")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
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
        # Security Note: Loading model weights only (state_dict) is safer than full model objects
        # Ensure path points to trusted model files. See SECURITY.md for details.
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from {path}")


def create_model(
    model_type: str = 'lstm',
    hidden_dim: int = 128,
    input_dim: int = None,
    **kwargs
) -> GIKModelWrapper:
    """
    Factory function to create a GIK model.
    
    Args:
        model_type: One of 'lstm' (best with focal loss), 'transformer', 'attention_lstm', 'gru', 'rnn', 'cnn'
        hidden_dim: Hidden dimension size
        input_dim: Number of input features (required)
        **kwargs: Additional arguments for the inner model
    
    Recommended config for best accuracy (~37%): lstm with focal loss (default in GIKTrainer)
    """
    if input_dim is None:
        raise ValueError("input_dim is required - get it from dataset.input_dim")
    
    if model_type == 'transformer':
        inner_model = TransformerModel(hidden_dim, **kwargs)
    elif model_type == 'attention_lstm':
        inner_model = AttentionLSTM(hidden_dim, **kwargs)
    elif model_type == 'lstm':
        inner_model = LSTMModel(hidden_dim, **kwargs)
    elif model_type == 'gru':
        inner_model = GRUModel(hidden_dim, **kwargs)
    elif model_type == 'rnn':
        inner_model = RNNModel(hidden_dim, **kwargs)
    elif model_type == 'cnn':
        inner_model = CNNModel(hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use: transformer, attention_lstm, lstm, gru, rnn, cnn")
    
    return GIKModelWrapper(inner_model, input_dim=input_dim, hidden_dim=hidden_dim)


def create_model_from_dataset(
    dataset: Dataset,
    model_type: str = 'transformer',
    hidden_dim: int = 128,
    **kwargs
) -> GIKModelWrapper:
    """
    Factory function to create a model configured for a dataset.
    
    Args:
        dataset: Dataset with input_dim property
        model_type: One of 'transformer' (best), 'attention_lstm', 'lstm', 'gru', 'rnn', 'cnn'
        hidden_dim: Hidden dimension size
        **kwargs: Additional arguments for the inner model
    """
    input_dim = getattr(dataset, 'input_dim', None)
    if input_dim is None:
        raise ValueError("Dataset must have input_dim property")
    return create_model(model_type=model_type, hidden_dim=hidden_dim, input_dim=input_dim, **kwargs)


def decode_predictions(
    predictions: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None
) -> List[Dict[str, Any]]:
    """Decode model predictions to characters."""
    results = []
    for i, pred in enumerate(predictions):
        result = {'char': INDEX_TO_CHAR.get(pred.item(), '?')}
        if probabilities is not None:
            result['confidence'] = probabilities[i, pred].item()
        results.append(result)
    return results
