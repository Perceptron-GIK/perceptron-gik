"""
Unit tests for ML models and training utilities in basic_nn.py
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.basic_nn import (
    GIKModelWrapper,
    LSTMModel,
    GRUModel,
    RNNModel,
    AttentionLSTM,
    TransformerModel,
    CNNModel,
    FocalLoss,
    GIKTrainer,
    create_model,
    create_model_from_dataset,
    decode_predictions,
    CHAR_TO_INDEX,
    INDEX_TO_CHAR,
    NUM_CLASSES
)

# Test constants
TEST_BATCH_SIZE = 8
TEST_SEQ_LEN = 10
TEST_INPUT_DIM = 30
TEST_HIDDEN_DIM = 16


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    x = torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (TEST_BATCH_SIZE,))
    y_onehot = torch.nn.functional.one_hot(y, num_classes=NUM_CLASSES).float()
    
    return {
        'x': x,
        'y': y,
        'y_onehot': y_onehot,
        'batch_size': TEST_BATCH_SIZE,
        'seq_len': TEST_SEQ_LEN,
        'input_dim': TEST_INPUT_DIM,
        'hidden_dim': TEST_HIDDEN_DIM
    }


@pytest.fixture
def mock_dataset(sample_data):
    """Create a mock dataset for testing."""
    n_samples = 100
    x = torch.randn(n_samples, sample_data['seq_len'], sample_data['input_dim'])
    y = torch.randint(0, NUM_CLASSES, (n_samples,))
    y_onehot = torch.nn.functional.one_hot(y, num_classes=NUM_CLASSES).float()
    
    dataset = TensorDataset(x, y_onehot)
    dataset.input_dim = sample_data['input_dim']
    
    # Add method to get class weights
    class_counts = torch.zeros(NUM_CLASSES)
    for label in y:
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    dataset.get_class_weights = lambda: class_weights
    
    return dataset


class TestCharacterMappings:
    """Test character to index mappings."""
    
    def test_char_to_index_lowercase(self):
        """Test lowercase letter mappings."""
        assert CHAR_TO_INDEX['a'] == 0
        assert CHAR_TO_INDEX['z'] == 25
    
    def test_char_to_index_digits(self):
        """Test digit mappings."""
        assert CHAR_TO_INDEX['0'] == 26
        assert CHAR_TO_INDEX['9'] == 35
    
    def test_char_to_index_special(self):
        """Test special character mappings."""
        assert CHAR_TO_INDEX[' '] == 36  # space
        assert CHAR_TO_INDEX['\n'] == 37  # enter
        assert CHAR_TO_INDEX['\b'] == 38  # backspace
        assert CHAR_TO_INDEX['\t'] == 39  # tab
    
    def test_index_to_char_reverse_mapping(self):
        """Test that INDEX_TO_CHAR is reverse of CHAR_TO_INDEX."""
        for char, idx in CHAR_TO_INDEX.items():
            assert INDEX_TO_CHAR[idx] == char
    
    def test_num_classes(self):
        """Test that NUM_CLASSES matches mapping size."""
        assert NUM_CLASSES == 40
        assert len(CHAR_TO_INDEX) == NUM_CLASSES


class TestLSTMModel:
    """Test LSTM model architecture."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces correct output shape."""
        model = LSTMModel(
            hidden_dim=sample_data['hidden_dim'],
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])
    
    def test_unidirectional_lstm(self, sample_data):
        """Test unidirectional LSTM."""
        model = LSTMModel(
            hidden_dim=sample_data['hidden_dim'],
            num_layers=1,
            bidirectional=False,
            dropout=0.0
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])


class TestGRUModel:
    """Test GRU model architecture."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces correct output shape."""
        model = GRUModel(
            hidden_dim=sample_data['hidden_dim'],
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])


class TestRNNModel:
    """Test RNN model architecture."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces correct output shape."""
        model = RNNModel(
            hidden_dim=sample_data['hidden_dim'],
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])


class TestAttentionLSTM:
    """Test Attention LSTM model architecture."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces correct output shape."""
        model = AttentionLSTM(
            hidden_dim=sample_data['hidden_dim'],
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            num_heads=2
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])


class TestTransformerModel:
    """Test Transformer model architecture."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces correct output shape."""
        model = TransformerModel(
            hidden_dim=sample_data['hidden_dim'],
            num_heads=2,
            num_layers=2,
            dropout=0.2
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])
    
    def test_cls_token_injection(self, sample_data):
        """Test that CLS token is properly injected."""
        model = TransformerModel(hidden_dim=sample_data['hidden_dim'])
        assert model.cls_token.shape == (1, 1, sample_data['hidden_dim'])


class TestCNNModel:
    """Test CNN model architecture."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces correct output shape."""
        model = CNNModel(
            hidden_dim=sample_data['hidden_dim'],
            kernel_sizes=[3, 5],
            dropout=0.2
        )
        x = torch.randn(sample_data['batch_size'], sample_data['seq_len'], sample_data['hidden_dim'])
        output = model(x)
        
        assert output.shape == (sample_data['batch_size'], sample_data['hidden_dim'])
    
    def test_default_kernel_sizes(self, sample_data):
        """Test default kernel sizes."""
        model = CNNModel(hidden_dim=sample_data['hidden_dim'])
        # Default kernel sizes are [3, 5, 7]
        assert len(model.convs) == 3


class TestGIKModelWrapper:
    """Test GIKModelWrapper functionality."""
    
    def test_forward_pass(self, sample_data):
        """Test forward pass through wrapped model."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'])
        wrapper = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim'],
            num_classes=NUM_CLASSES
        )
        
        output = wrapper(sample_data['x'])
        assert output.shape == (sample_data['batch_size'], NUM_CLASSES)
    
    def test_predict(self, sample_data):
        """Test prediction returns class indices."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'])
        wrapper = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim']
        )
        
        predictions = wrapper.predict(sample_data['x'])
        assert predictions.shape == (sample_data['batch_size'],)
        assert predictions.dtype == torch.int64
        assert torch.all(predictions >= 0) and torch.all(predictions < NUM_CLASSES)
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'])
        wrapper = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim']
        )
        
        probs = wrapper.predict_proba(sample_data['x'])
        assert probs.shape == (sample_data['batch_size'], NUM_CLASSES)
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(sample_data['batch_size']), atol=1e-5)


class TestFocalLoss:
    """Test Focal Loss implementation."""
    
    def test_focal_loss_computation(self, sample_data):
        """Test focal loss computes without errors."""
        focal_loss = FocalLoss(gamma=2.0)
        logits = torch.randn(sample_data['batch_size'], NUM_CLASSES)
        targets = sample_data['y']
        
        loss = focal_loss(logits, targets)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
    
    def test_focal_loss_with_alpha(self, sample_data):
        """Test focal loss with class weights."""
        alpha = torch.ones(NUM_CLASSES)
        focal_loss = FocalLoss(gamma=2.0, alpha=alpha)
        logits = torch.randn(sample_data['batch_size'], NUM_CLASSES)
        targets = sample_data['y']
        
        loss = focal_loss(logits, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_focal_loss_gamma_effect(self, sample_data):
        """Test that gamma affects loss magnitude."""
        logits = torch.randn(sample_data['batch_size'], NUM_CLASSES)
        targets = sample_data['y']
        
        loss_gamma0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_gamma2 = FocalLoss(gamma=2.0)(logits, targets)
        
        # With gamma=0, focal loss should equal CE loss
        ce_loss = nn.CrossEntropyLoss()(logits, targets)
        assert torch.allclose(loss_gamma0, ce_loss, atol=1e-5)


class TestGIKTrainer:
    """Test GIKTrainer functionality."""
    
    def test_trainer_initialization(self, mock_dataset, sample_data):
        """Test trainer initializes correctly."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'])
        model = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim']
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=mock_dataset,
            batch_size=16,
            learning_rate=1e-3,
            device='cpu'
        )
        
        assert trainer.device.type == 'cpu'
        assert len(trainer.train_dataset) > 0
        assert len(trainer.val_dataset) > 0
        assert len(trainer.test_dataset) > 0
    
    def test_train_epoch(self, mock_dataset, sample_data):
        """Test single training epoch."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'], num_layers=1)
        model = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim']
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=mock_dataset,
            batch_size=16,
            learning_rate=1e-3,
            device='cpu'
        )
        
        train_loss, train_acc = trainer.train_epoch()
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1
    
    def test_validate(self, mock_dataset, sample_data):
        """Test validation."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'], num_layers=1)
        model = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim']
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=mock_dataset,
            batch_size=16,
            device='cpu'
        )
        
        val_loss, val_acc = trainer.validate()
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 1
    
    def test_evaluate_test(self, mock_dataset, sample_data):
        """Test evaluation on test set."""
        inner_model = LSTMModel(hidden_dim=sample_data['hidden_dim'], num_layers=1)
        model = GIKModelWrapper(
            inner_model=inner_model,
            input_dim=sample_data['input_dim'],
            hidden_dim=sample_data['hidden_dim']
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=mock_dataset,
            batch_size=16,
            device='cpu'
        )
        
        test_loss, test_acc = trainer.evaluate_test()
        assert isinstance(test_loss, float)
        assert isinstance(test_acc, float)
        assert test_loss >= 0
        assert 0 <= test_acc <= 1


class TestModelCreation:
    """Test model factory functions."""
    
    def test_create_model_lstm(self, sample_data):
        """Test creating LSTM model."""
        model = create_model(
            model_type='lstm',
            hidden_dim=sample_data['hidden_dim'],
            input_dim=sample_data['input_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert isinstance(model.inner_model, LSTMModel)
    
    def test_create_model_gru(self, sample_data):
        """Test creating GRU model."""
        model = create_model(
            model_type='gru',
            hidden_dim=sample_data['hidden_dim'],
            input_dim=sample_data['input_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert isinstance(model.inner_model, GRUModel)
    
    def test_create_model_rnn(self, sample_data):
        """Test creating RNN model."""
        model = create_model(
            model_type='rnn',
            hidden_dim=sample_data['hidden_dim'],
            input_dim=sample_data['input_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert isinstance(model.inner_model, RNNModel)
    
    def test_create_model_transformer(self, sample_data):
        """Test creating Transformer model."""
        model = create_model(
            model_type='transformer',
            hidden_dim=sample_data['hidden_dim'],
            input_dim=sample_data['input_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert isinstance(model.inner_model, TransformerModel)
    
    def test_create_model_attention_lstm(self, sample_data):
        """Test creating Attention LSTM model."""
        model = create_model(
            model_type='attention_lstm',
            hidden_dim=sample_data['hidden_dim'],
            input_dim=sample_data['input_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert isinstance(model.inner_model, AttentionLSTM)
    
    def test_create_model_cnn(self, sample_data):
        """Test creating CNN model."""
        model = create_model(
            model_type='cnn',
            hidden_dim=sample_data['hidden_dim'],
            input_dim=sample_data['input_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert isinstance(model.inner_model, CNNModel)
    
    def test_create_model_invalid_type(self, sample_data):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            create_model(
                model_type='invalid_model',
                hidden_dim=sample_data['hidden_dim'],
                input_dim=sample_data['input_dim']
            )
    
    def test_create_model_missing_input_dim(self, sample_data):
        """Test error when input_dim is missing."""
        with pytest.raises(ValueError):
            create_model(
                model_type='lstm',
                hidden_dim=sample_data['hidden_dim']
            )
    
    def test_create_model_from_dataset(self, mock_dataset, sample_data):
        """Test creating model from dataset."""
        model = create_model_from_dataset(
            dataset=mock_dataset,
            model_type='lstm',
            hidden_dim=sample_data['hidden_dim']
        )
        assert isinstance(model, GIKModelWrapper)
        assert model.input_dim == sample_data['input_dim']


class TestDecodePredictions:
    """Test prediction decoding utilities."""
    
    def test_decode_predictions_basic(self):
        """Test basic prediction decoding."""
        predictions = torch.tensor([0, 1, 26, 36])  # a, b, 0, space
        results = decode_predictions(predictions)
        
        assert len(results) == 4
        assert results[0]['char'] == 'a'
        assert results[1]['char'] == 'b'
        assert results[2]['char'] == '0'
        assert results[3]['char'] == ' '
    
    def test_decode_predictions_with_probabilities(self):
        """Test decoding with confidence scores."""
        predictions = torch.tensor([0, 1])
        probs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        
        results = decode_predictions(predictions, probs)
        
        assert len(results) == 2
        assert results[0]['char'] == 'a'
        assert abs(results[0]['confidence'] - 0.8) < 1e-5
        assert results[1]['char'] == 'b'
        assert abs(results[1]['confidence'] - 0.7) < 1e-5
    
    def test_decode_predictions_unknown(self):
        """Test decoding unknown indices."""
        predictions = torch.tensor([999])  # Invalid index
        results = decode_predictions(predictions)
        
        assert results[0]['char'] == '?'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
