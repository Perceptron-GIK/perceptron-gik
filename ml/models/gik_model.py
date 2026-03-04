"""
GIK Neural Network Models and Training
Usage:
    from pretraining import load_preprocessed_dataset
    from ml.models.basic_nn import create_model_from_dataset, GIKTrainer
    
    dataset = load_preprocessed_dataset("data/processed.pt")
    model = create_model_from_dataset(dataset, 'transformer', hidden_dim=128)
    trainer = GIKTrainer(model, dataset, learning_rate=5e-4)
    trainer.train(epochs=50)
"""
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch import optim
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .default_models import (
    TransformerModel, AttentionLSTM, LSTMModel, GRUModel, RNNModel, CNNModel
)
from src.Constants.char_to_key import INDEX_TO_CHAR, NUM_CLASSES
from src.pre_processing.augmentation import GIKAugmentationsPerFeature, AugmentedDataset

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
        inner_model_dim : int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        n_fc_layers: int = 1,
        pool_output: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pool_output = pool_output
        
        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, inner_model_dim),
            nn.LayerNorm(inner_model_dim),
            nn.ReLU(),
            # nn.Dropout(dropout)
        )
        
        # Inner model (user-provided)
        self.inner_model = inner_model
        
        self.project_from_inner = nn.Sequential(
            nn.Linear(inner_model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # FC Layers
        layers = []
        d = hidden_dim
        for _ in range(n_fc_layers):
            d_next = max(16, d // 2) # Minimum 16 Neurons
            layers += [nn.Linear(d, d_next), nn.LayerNorm(d_next), nn.ReLU(), nn.Dropout(dropout)]
            d = d_next

        self.fc_stack = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(nn.Linear(d, num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.inner_model(x)
        
        # If we have a channel dimension from CNN then pool it 
        if len(x.shape) == 3:
            if self.pool_output:
                x = x.mean(dim=1)
            else:
                x = x[:, -1, :]
        
        x = self.project_from_inner(x)
        x = self.fc_stack(x)
        return self.classifier(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


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
        augmentation: bool = False, # Whether to use augmentation (only for training set),
        device: Optional[str] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        loss: callable = nn.CrossEntropyLoss,
        loss_kwargs: Optional[Dict] = None,
        regression: bool = False,
    ):
        """
        regression: If True, labels are continuous (e.g. 2D coords); loss(logits, batch_y) and accuracy uses L1.
        """
        self.regression = regression
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

        augment = GIKAugmentationsPerFeature()
        # augmentation boolean as the flag, if false returns the same data, if true then applies augmentation on the fly
        self.regression = regression
        self.train_dataset_aug = AugmentedDataset(self.train_dataset, augment=augment, use_augmentation=augmentation, synthetic_multiplier=5, precompute_synthetic=True, device=self.device, regression=regression)   # or False to disable
        
        self.train_loader = DataLoader(self.train_dataset_aug, batch_size=batch_size, shuffle=True, num_workers=0) # pass in the augmented dataset here for training only
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if loss is not None:
            self.criterion = loss(**loss_kwargs)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for data in self.train_loader:
            batch_x, batch_y, raw_labels = None, None, None
            if self.regression:
                batch_x, batch_y, raw_labels = data
                raw_labels = raw_labels.to(self.device)
            else:
                batch_x, batch_y = data
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            if self.regression:
                loss = self.criterion(logits, batch_y, raw_labels)
                
            else:
                loss = self.criterion(logits, batch_y.argmax(dim=-1))
                pred = logits.argmax(dim=-1)
                correct += (pred == batch_y.argmax(dim=-1)).sum().item()
            total += batch_x.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        if self.regression:
            acc = None
        else:
            acc = correct / total if total else 0.0
        return total_loss / total if total else 0.0, acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for data in self.val_loader:
            batch_x, batch_y, raw_labels = None, None, None
            if self.regression:
                batch_x, batch_y, raw_labels = data
                raw_labels = raw_labels.to(self.device)
            else:
                batch_x, batch_y = data
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.model(batch_x)
            if self.regression:
                loss = self.criterion(logits, batch_y, raw_labels)
            else:
                loss = self.criterion(logits, batch_y.argmax(dim=-1))
                pred = logits.argmax(dim=-1)
                correct += (pred == batch_y.argmax(dim=-1)).sum().item()
            total_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

        if self.regression:
            acc = None
        else:
            acc = correct / total if total else 0.0
        return total_loss / total if total else 0.0, acc
    
    @torch.no_grad()
    def evaluate_test(self) -> Tuple[float, float]:
        """Evaluate on the held-out test set (causal split)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        for data in self.test_loader:
            batch_x, batch_y, raw_labels = None, None, None
            if self.regression:
                batch_x, batch_y, raw_labels = data
                raw_labels = raw_labels.to(self.device)
            else:
                batch_x, batch_y = data
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            logits = self.model(batch_x)
            if self.regression:
                loss = self.criterion(logits, batch_y, raw_labels)
            else:
                loss = self.criterion(logits, batch_y.argmax(dim=-1))
                correct += (logits.argmax(dim=-1) == batch_y.argmax(dim=-1)).sum().item()
            total_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

        if self.regression:
            acc = None
        else:
            acc = correct / total if total else 0.0
        return total_loss / total if total else 0.0, acc
    
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
        print(f"Train: {len(self.train_dataset_aug)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)} (causal split, no shuffle)")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            if not self.regression:
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} ")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ")
            
            if not self.regression:
                print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} ")
            
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


def create_model(
    model_type: str = 'lstm',
    hidden_dim_inner_model: int = 128,
    hidden_dim_classification_head: int = 256,
    no_layers_classification_head: int = 1,
    dropout_inner_layers = 0.5, 
    output_logits = NUM_CLASSES,
    input_dim: int = None,
    inner_model_kwargs = {},
) -> GIKModelWrapper:
    """
    Function to create a GIK model.
    
    Args:
        model_type: One of 'lstm' (best with focal loss), 'transformer', 'attention_lstm', 'gru', 'rnn', 'cnn'
        hidden_dim_inner_model: Hidden dimension size for the inner model
        hidden_dim_classification_head: Hidden dimension size for the classification Head
        input_dim: Number of input features (required)
        **kwargs: Additional arguments for the inner model
    
    """
    if input_dim is None:
        raise ValueError("input_dim is required - get it from dataset.input_dim")
    
    if model_type == 'transformer':
        inner_model = TransformerModel(hidden_dim_inner_model, **inner_model_kwargs)
    elif model_type == 'attention_lstm':
        inner_model = AttentionLSTM(hidden_dim_inner_model, **inner_model_kwargs)
    elif model_type == 'lstm':
        inner_model = LSTMModel(hidden_dim_inner_model, **inner_model_kwargs)
    elif model_type == 'gru':
        inner_model = GRUModel(hidden_dim_inner_model, **inner_model_kwargs)
    elif model_type == 'rnn':
        inner_model = RNNModel(hidden_dim_inner_model, **inner_model_kwargs)
    elif model_type == 'cnn':
        inner_model = CNNModel(hidden_dim_inner_model, **inner_model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use: transformer, attention_lstm, lstm, gru, rnn, cnn")
    
    return GIKModelWrapper(inner_model, 
                           input_dim=input_dim,
                           hidden_dim=hidden_dim_classification_head, 
                           dropout = dropout_inner_layers, 
                           n_fc_layers = no_layers_classification_head,
                           inner_model_dim = hidden_dim_inner_model,
                           num_classes = output_logits)


def create_model_auto_input_dim(
    dataset: Dataset,
    model_type: str = 'lstm',
    hidden_dim_inner_model: int = 128,
    hidden_dim_classification_head: int = 256,
    no_layers_classification_head: int = 1,
    dropout_inner_layers = 0.5, 
    output_logits = NUM_CLASSES,
    inner_model_kwargs = {},
    **kwargs
) -> GIKModelWrapper:
    """
    Function to create a model configured for a dataset. This exists only to deal with automatic Input length detection
    
    Args:
        refer to create_model
    """
    input_dim = getattr(dataset, 'input_dim', None)
    if input_dim is None:
        raise ValueError("Dataset must have input_dim property")
    return create_model(model_type=model_type,
                        hidden_dim_inner_model=hidden_dim_inner_model,
                        hidden_dim_classification_head = hidden_dim_classification_head,
                        no_layers_classification_head = no_layers_classification_head,
                        dropout_inner_layers = dropout_inner_layers,
                        output_logits=output_logits,
                        input_dim=input_dim, 
                        inner_model_kwargs = inner_model_kwargs)


def decode_predictions(
    predictions: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None
) -> List[Dict[str, Any]]:
    """Decode model predictions to characters."""
    results = []
    for i, pred in enumerate(predictions):
        result = {'char': INDEX_TO_CHAR.get(pred.item())}
        if probabilities is not None:
            result['confidence'] = probabilities[i, pred].item()
        results.append(result)
    return results
