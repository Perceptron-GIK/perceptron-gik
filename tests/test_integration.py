"""
Integration tests for end-to-end ML workflow.
"""
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretraining import preprocess_and_export, load_preprocessed_dataset
from ml.models.basic_nn import create_model_from_dataset, GIKTrainer
from src.pre_processing.alignment import NUM_CLASSES


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_dataset_files(temp_dir):
    """Create sample dataset files for end-to-end testing."""
    # Create synthetic IMU data
    n_samples = 200
    imu_data = pd.DataFrame({
        'time_stamp': np.linspace(0, 2.0, n_samples),
        'ax_base': np.random.randn(n_samples) * 0.1,
        'ay_base': np.random.randn(n_samples) * 0.1,
        'az_base': np.random.randn(n_samples) * 0.1 + 9.8,
        'gx_base': np.random.randn(n_samples) * 0.01,
        'gy_base': np.random.randn(n_samples) * 0.01,
        'gz_base': np.random.randn(n_samples) * 0.01,
        'ax_thumb': np.random.randn(n_samples) * 0.1,
        'ay_thumb': np.random.randn(n_samples) * 0.1,
        'az_thumb': np.random.randn(n_samples) * 0.1 + 9.8,
        'gx_thumb': np.random.randn(n_samples) * 0.01,
        'gy_thumb': np.random.randn(n_samples) * 0.01,
        'gz_thumb': np.random.randn(n_samples) * 0.01,
    })
    
    # Create keyboard events
    times = np.linspace(0.1, 1.9, 15)
    keys = list('abcdefghijklmno')
    keyboard_data = pd.DataFrame({
        'time': times,
        'name': keys,
        'event_type': ['down'] * len(times),
    })
    
    # Save to files
    left_file = os.path.join(temp_dir, 'Left_test.csv')
    keyboard_file = os.path.join(temp_dir, 'Keyboard_test.csv')
    
    imu_data.to_csv(left_file, index=False)
    keyboard_data.to_csv(keyboard_file, index=False)
    
    return {
        'data_dir': temp_dir,
        'left_file': 'Left_test.csv',
        'keyboard_file': 'Keyboard_test.csv'
    }


class TestEndToEndWorkflow:
    """Test complete workflow from data preprocessing to model training."""
    
    def test_preprocessing_pipeline(self, sample_dataset_files, temp_dir):
        """Test data preprocessing pipeline."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        metadata = preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False  # Skip filtering for speed
        )
        
        # Check that preprocessing completed
        assert os.path.exists(output_path)
        assert metadata['num_samples'] > 0
        assert metadata['input_dim'] > 0
        
        # Check metadata file was created
        metadata_json = output_path.replace('.pt', '_metadata.json')
        assert os.path.exists(metadata_json)
    
    def test_dataset_loading(self, sample_dataset_files, temp_dir):
        """Test loading preprocessed dataset."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        dataset = load_preprocessed_dataset(output_path)
        
        assert len(dataset) > 0
        assert dataset.input_dim > 0
        
        # Test getting a sample
        sample, label = dataset[0]
        assert sample.ndim == 2
        assert label.ndim == 1
        assert label.shape[0] == NUM_CLASSES
    
    def test_model_creation_from_dataset(self, sample_dataset_files, temp_dir):
        """Test creating model from dataset."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        dataset = load_preprocessed_dataset(output_path)
        
        # Test different model types
        for model_type in ['lstm', 'gru', 'transformer']:
            model = create_model_from_dataset(
                dataset,
                model_type=model_type,
                hidden_dim=16,
                num_layers=1
            )
            
            assert model is not None
            assert model.input_dim == dataset.input_dim
    
    def test_trainer_initialization(self, sample_dataset_files, temp_dir):
        """Test trainer initialization with real dataset."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        dataset = load_preprocessed_dataset(output_path)
        model = create_model_from_dataset(
            dataset,
            model_type='lstm',
            hidden_dim=16,
            num_layers=1
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=dataset,
            batch_size=8,
            learning_rate=1e-3,
            device='cpu'
        )
        
        assert trainer is not None
        assert len(trainer.train_dataset) > 0
        assert len(trainer.val_dataset) >= 0
        assert len(trainer.test_dataset) >= 0
    
    def test_single_training_epoch(self, sample_dataset_files, temp_dir):
        """Test running a single training epoch."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        dataset = load_preprocessed_dataset(output_path)
        model = create_model_from_dataset(
            dataset,
            model_type='lstm',
            hidden_dim=8,
            num_layers=1
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=dataset,
            batch_size=4,
            learning_rate=1e-3,
            device='cpu',
            use_focal_loss=True
        )
        
        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss > 0
        assert 0 <= train_acc <= 1
    
    def test_validation_after_training(self, sample_dataset_files, temp_dir):
        """Test validation after training."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        dataset = load_preprocessed_dataset(output_path)
        model = create_model_from_dataset(
            dataset,
            model_type='lstm',
            hidden_dim=8,
            num_layers=1
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=dataset,
            batch_size=4,
            device='cpu'
        )
        
        # Train and validate
        trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss > 0
        assert 0 <= val_acc <= 1
    
    def test_model_save_and_load(self, sample_dataset_files, temp_dir):
        """Test saving and loading trained model."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        model_path = os.path.join(temp_dir, 'test_model.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        dataset = load_preprocessed_dataset(output_path)
        model = create_model_from_dataset(
            dataset,
            model_type='lstm',
            hidden_dim=8,
            num_layers=1
        )
        
        trainer = GIKTrainer(
            model=model,
            dataset=dataset,
            batch_size=4,
            device='cpu'
        )
        
        # Train briefly and save
        trainer.train_epoch()
        torch.save(model.state_dict(), model_path)
        
        # Create new model and load weights
        new_model = create_model_from_dataset(
            dataset,
            model_type='lstm',
            hidden_dim=8,
            num_layers=1
        )
        new_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Verify models have same weights
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


class TestMultipleModelTypes:
    """Test different model architectures work end-to-end."""
    
    @pytest.fixture
    def prepared_dataset(self, sample_dataset_files, temp_dir):
        """Prepare dataset once for all model tests."""
        output_path = os.path.join(temp_dir, 'processed_dataset.pt')
        
        preprocess_and_export(
            data_dir=sample_dataset_files['data_dir'],
            keyboard_file=sample_dataset_files['keyboard_file'],
            left_file=sample_dataset_files['left_file'],
            output_path=output_path,
            max_seq_length=10,
            normalize=True,
            apply_filtering=False
        )
        
        return load_preprocessed_dataset(output_path)
    
    @pytest.mark.parametrize("model_type", ["lstm", "gru", "rnn", "transformer", "attention_lstm", "cnn"])
    def test_model_type_trains(self, prepared_dataset, model_type):
        """Test that each model type can train."""
        # Different models need different kwargs
        if model_type == 'cnn':
            model = create_model_from_dataset(
                prepared_dataset,
                model_type=model_type,
                hidden_dim=8
            )
        else:
            model = create_model_from_dataset(
                prepared_dataset,
                model_type=model_type,
                hidden_dim=8,
                num_layers=1
            )
        
        trainer = GIKTrainer(
            model=model,
            dataset=prepared_dataset,
            batch_size=4,
            device='cpu'
        )
        
        # Should complete without errors
        train_loss, train_acc = trainer.train_epoch()
        
        assert train_loss > 0
        assert 0 <= train_acc <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
