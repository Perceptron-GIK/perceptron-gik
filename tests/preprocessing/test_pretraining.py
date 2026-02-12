"""
Unit tests for preprocessing and dataset creation in pretraining.py
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pretraining import (
    PreprocessedGIKDataset,
    load_preprocessed_dataset,
    filter_imu_data,
    export_dataset_to_csv
)
from src.pre_processing.alignment import NUM_CLASSES, INDEX_TO_CHAR

# Test constants
TEST_N_SAMPLES = 50
TEST_SEQ_LEN = 10
TEST_FEAT_DIM = 30


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_imu_dataframe():
    """Create a sample IMU dataframe for testing."""
    n_samples = 100
    data = {
        'time_stamp': np.arange(n_samples) * 0.01,  # 100Hz sampling
        'sample_id': np.arange(n_samples),
    }
    
    # Add IMU data for base sensor
    imu_parts = ['base', 'thumb', 'index']
    for part in imu_parts:
        data[f'ax_{part}'] = np.random.randn(n_samples) * 0.1
        data[f'ay_{part}'] = np.random.randn(n_samples) * 0.1
        data[f'az_{part}'] = np.random.randn(n_samples) * 0.1 + 9.8  # gravity
        data[f'gx_{part}'] = np.random.randn(n_samples) * 0.01
        data[f'gy_{part}'] = np.random.randn(n_samples) * 0.01
        data[f'gz_{part}'] = np.random.randn(n_samples) * 0.01
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_preprocessed_data(temp_dir):
    """Create sample preprocessed data file."""
    samples = torch.randn(TEST_N_SAMPLES, TEST_SEQ_LEN, TEST_FEAT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (TEST_N_SAMPLES,))
    labels_onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
    
    prev_labels = torch.randint(-1, NUM_CLASSES, (TEST_N_SAMPLES,))
    prev_labels_onehot = torch.zeros(TEST_N_SAMPLES, NUM_CLASSES)
    valid_prev = prev_labels >= 0
    prev_labels_onehot[valid_prev] = F.one_hot(prev_labels[valid_prev], num_classes=NUM_CLASSES).float()
    
    mean = samples.mean(dim=(0, 1))
    std = samples.std(dim=(0, 1))
    std[std == 0] = 1.0
    
    class_counts = torch.zeros(NUM_CLASSES)
    for label in labels:
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    
    metadata = {
        'num_samples': TEST_N_SAMPLES,
        'num_hands': 1,
        'has_right': False,
        'has_left': True,
        'input_dim': TEST_FEAT_DIM + NUM_CLASSES,
        'feat_dim': TEST_FEAT_DIM,
        'features_per_hand': TEST_FEAT_DIM,
        'max_seq_length': TEST_SEQ_LEN,
        'num_classes': NUM_CLASSES,
        'skipped_chars': {},
    }
    
    save_dict = {
        'samples': samples,
        'labels': labels_onehot,
        'prev_labels': prev_labels_onehot,
        'mean': mean,
        'std': std,
        'class_weights': class_weights,
        'metadata': metadata,
        'normalize': True,
        'max_seq_length': TEST_SEQ_LEN,
    }
    
    pt_path = os.path.join(temp_dir, 'test_dataset.pt')
    torch.save(save_dict, pt_path)
    
    return pt_path


class TestFilterIMUData:
    """Test IMU data filtering functionality."""
    
    def test_filter_preserves_shape(self, sample_imu_dataframe):
        """Test that filtering preserves dataframe shape."""
        original_shape = sample_imu_dataframe.shape
        filtered_df = filter_imu_data(sample_imu_dataframe)
        
        # Should have same number of rows
        assert filtered_df.shape[0] == original_shape[0]
        # Should have additional columns (x, y, z positions)
        assert filtered_df.shape[1] >= original_shape[1]
    
    def test_filter_adds_position_columns(self, sample_imu_dataframe):
        """Test that filtering adds position columns."""
        filtered_df = filter_imu_data(sample_imu_dataframe)
        
        # Check that position columns are added
        imu_parts = ['base', 'thumb', 'index']
        for part in imu_parts:
            assert f'x_{part}' in filtered_df.columns
            assert f'y_{part}' in filtered_df.columns
            assert f'z_{part}' in filtered_df.columns
    
    def test_filter_handles_nan_values(self, sample_imu_dataframe):
        """Test that filtering handles NaN values properly."""
        # Add some NaN values
        sample_imu_dataframe.iloc[10:20, 2:5] = np.nan
        
        filtered_df = filter_imu_data(sample_imu_dataframe)
        
        # Check that no NaN or Inf values remain
        for col in filtered_df.columns:
            if col.startswith(('ax_', 'ay_', 'az_', 'x_', 'y_', 'z_')):
                assert not filtered_df[col].isna().any()
                assert not np.isinf(filtered_df[col]).any()
    
    def test_filter_handles_missing_parts(self):
        """Test filtering with missing IMU parts."""
        # Create dataframe with only base sensor
        n_samples = 50
        data = {
            'time_stamp': np.arange(n_samples) * 0.01,
            'ax_base': np.random.randn(n_samples),
            'ay_base': np.random.randn(n_samples),
            'az_base': np.random.randn(n_samples),
            'gx_base': np.random.randn(n_samples),
            'gy_base': np.random.randn(n_samples),
            'gz_base': np.random.randn(n_samples),
        }
        df = pd.DataFrame(data)
        
        # Should not raise error
        filtered_df = filter_imu_data(df)
        assert 'x_base' in filtered_df.columns


class TestPreprocessedGIKDataset:
    """Test PreprocessedGIKDataset class."""
    
    def test_dataset_loading(self, sample_preprocessed_data):
        """Test loading dataset from file."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        assert isinstance(dataset, PreprocessedGIKDataset)
        assert len(dataset) == TEST_N_SAMPLES
        assert dataset.input_dim > 0
    
    def test_dataset_getitem(self, sample_preprocessed_data):
        """Test getting items from dataset."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        sample, label = dataset[0]
        
        # Sample should have prev-char concatenated
        assert sample.ndim == 2  # (seq_len, features)
        assert label.ndim == 1  # one-hot label
        assert label.shape[0] == NUM_CLASSES
        assert torch.allclose(label.sum(), torch.tensor(1.0))
    
    def test_dataset_normalization(self, sample_preprocessed_data):
        """Test that normalization is applied."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        assert dataset.normalize is True
        assert dataset.mean is not None
        assert dataset.std is not None
        
        # Get a sample and check it's normalized
        sample, _ = dataset[0]
        # Sample values should be roughly standardized (not a strict test)
        assert sample.abs().mean() < 10  # Reasonable range for normalized data
    
    def test_dataset_class_weights(self, sample_preprocessed_data):
        """Test class weights computation."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        class_weights = dataset.get_class_weights()
        assert class_weights.shape[0] == NUM_CLASSES
        assert torch.all(class_weights > 0)
    
    def test_dataset_metadata(self, sample_preprocessed_data):
        """Test metadata is preserved."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        assert 'num_samples' in dataset.metadata
        assert 'input_dim' in dataset.metadata
        assert 'max_seq_length' in dataset.metadata
        assert dataset.metadata['num_classes'] == NUM_CLASSES
    
    def test_dataset_handles_legacy_format(self, temp_dir):
        """Test loading dataset with legacy index labels (1D)."""
        n_samples = 20
        seq_len = 10
        feat_dim = 30
        
        samples = torch.randn(n_samples, seq_len, feat_dim)
        labels = torch.randint(0, NUM_CLASSES, (n_samples,))  # 1D indices
        
        metadata = {
            'num_samples': n_samples,
            'num_hands': 1,
            'has_right': False,
            'has_left': True,
            'input_dim': feat_dim + NUM_CLASSES,
            'feat_dim': feat_dim,
            'features_per_hand': feat_dim,
            'max_seq_length': seq_len,
            'num_classes': NUM_CLASSES,
            'skipped_chars': {},
        }
        
        save_dict = {
            'samples': samples,
            'labels': labels,  # 1D format
            'prev_labels': None,
            'mean': None,
            'std': None,
            'class_weights': torch.ones(NUM_CLASSES),
            'metadata': metadata,
            'normalize': False,
            'max_seq_length': seq_len,
        }
        
        pt_path = os.path.join(temp_dir, 'legacy_dataset.pt')
        torch.save(save_dict, pt_path)
        
        # Should successfully load and convert to one-hot
        dataset = load_preprocessed_dataset(pt_path)
        sample, label = dataset[0]
        assert label.shape == (NUM_CLASSES,)


class TestExportDatasetToCSV:
    """Test CSV export functionality."""
    
    def test_export_creates_files(self, sample_preprocessed_data, temp_dir):
        """Test that export creates CSV files."""
        summary_path = export_dataset_to_csv(
            sample_preprocessed_data,
            output_dir=temp_dir,
            include_features=True,
            max_samples=10
        )
        
        assert os.path.exists(summary_path)
        assert os.path.exists(os.path.join(temp_dir, 'dataset_features.csv'))
    
    def test_export_summary_content(self, sample_preprocessed_data, temp_dir):
        """Test summary CSV content."""
        summary_path = export_dataset_to_csv(
            sample_preprocessed_data,
            output_dir=temp_dir,
            include_features=False,
            max_samples=10
        )
        
        df = pd.read_csv(summary_path)
        
        assert 'sample_idx' in df.columns
        assert 'label_idx' in df.columns
        assert 'character' in df.columns
        assert len(df) == 10  # max_samples
    
    def test_export_special_characters(self, temp_dir):
        """Test export handles special characters correctly."""
        # Create dataset with special characters
        n_samples = 4
        seq_len = 5
        feat_dim = 10
        
        samples = torch.randn(n_samples, seq_len, feat_dim)
        # Labels for space, enter, backspace, tab
        labels = torch.tensor([36, 37, 38, 39])
        labels_onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        
        metadata = {
            'num_samples': n_samples,
            'num_hands': 1,
            'has_right': False,
            'has_left': True,
            'input_dim': feat_dim + NUM_CLASSES,
            'feat_dim': feat_dim,
            'features_per_hand': feat_dim,
            'max_seq_length': seq_len,
            'num_classes': NUM_CLASSES,
            'skipped_chars': {},
        }
        
        save_dict = {
            'samples': samples,
            'labels': labels_onehot,
            'prev_labels': torch.zeros(n_samples, NUM_CLASSES),
            'mean': None,
            'std': None,
            'class_weights': torch.ones(NUM_CLASSES),
            'metadata': metadata,
            'normalize': False,
            'max_seq_length': seq_len,
        }
        
        pt_path = os.path.join(temp_dir, 'special_chars.pt')
        torch.save(save_dict, pt_path)
        
        summary_path = export_dataset_to_csv(pt_path, output_dir=temp_dir, include_features=False)
        df = pd.read_csv(summary_path)
        
        # Check that special characters are properly labeled
        assert 'SPACE' in df['character'].values
        assert 'ENTER' in df['character'].values
        assert 'BACKSPACE' in df['character'].values
        assert 'TAB' in df['character'].values


class TestDataConsistency:
    """Test data consistency and integrity."""
    
    def test_no_data_leakage(self, sample_preprocessed_data):
        """Test that data doesn't have obvious leakage issues."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        # Check that samples are unique (not all the same)
        sample1, _ = dataset[0]
        sample2, _ = dataset[1] if len(dataset) > 1 else dataset[0]
        
        # Samples should be different (unless by chance)
        assert not torch.allclose(sample1, sample2)
    
    def test_label_distribution(self, sample_preprocessed_data):
        """Test label distribution is reasonable."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        label_counts = torch.zeros(NUM_CLASSES)
        for i in range(len(dataset)):
            _, label = dataset[i]
            label_idx = label.argmax()
            label_counts[label_idx] += 1
        
        # At least some labels should be present
        assert (label_counts > 0).sum() > 0
    
    def test_no_nan_in_samples(self, sample_preprocessed_data):
        """Test that samples don't contain NaN or Inf values."""
        dataset = load_preprocessed_dataset(sample_preprocessed_data)
        
        for i in range(min(10, len(dataset))):
            sample, _ = dataset[i]
            assert not torch.isnan(sample).any()
            assert not torch.isinf(sample).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
