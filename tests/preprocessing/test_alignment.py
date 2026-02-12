"""
Unit tests for IMU-keyboard alignment in alignment.py
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pre_processing.alignment import (
    TrainingData,
    Preprocessing,
    CHAR_TO_INDEX,
    INDEX_TO_CHAR,
    NUM_CLASSES,
    SPECIAL_KEY_MAP
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_imu_csv(temp_dir):
    """Create a sample IMU CSV file."""
    n_samples = 100
    data = {
        'time_stamp': np.linspace(0, 1.0, n_samples),
        'sample_id': np.arange(n_samples),
        'ax_base': np.random.randn(n_samples),
        'ay_base': np.random.randn(n_samples),
        'az_base': np.random.randn(n_samples),
        'gx_base': np.random.randn(n_samples),
        'gy_base': np.random.randn(n_samples),
        'gz_base': np.random.randn(n_samples),
    }
    df = pd.DataFrame(data)
    
    filepath = os.path.join(temp_dir, 'imu_data.csv')
    df.to_csv(filepath, index=False)
    return filepath, temp_dir


@pytest.fixture
def sample_keyboard_csv(temp_dir):
    """Create a sample keyboard events CSV file."""
    # Create keyboard events at regular intervals
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    
    data = {
        'time': times,
        'name': keys,
        'event_type': ['down'] * len(times),
    }
    df = pd.DataFrame(data)
    
    filepath = os.path.join(temp_dir, 'keyboard_data.csv')
    df.to_csv(filepath, index=False)
    return filepath, temp_dir


class TestCharacterMappings:
    """Test character mapping constants."""
    
    def test_char_to_index_complete(self):
        """Test that all expected characters are mapped."""
        # 26 letters + 10 digits + 4 special = 40
        assert len(CHAR_TO_INDEX) == NUM_CLASSES
    
    def test_special_key_map(self):
        """Test special key mappings."""
        assert SPECIAL_KEY_MAP['enter'] == '\n'
        assert SPECIAL_KEY_MAP['space'] == ' '
        assert SPECIAL_KEY_MAP['tab'] == '\t'
        assert SPECIAL_KEY_MAP['backspace'] == '\b'
    
    def test_index_to_char_inverse(self):
        """Test that INDEX_TO_CHAR is inverse of CHAR_TO_INDEX."""
        for char, idx in CHAR_TO_INDEX.items():
            assert INDEX_TO_CHAR[idx] == char
        
        assert len(INDEX_TO_CHAR) == len(CHAR_TO_INDEX)


class TestTrainingData:
    """Test TrainingData class."""
    
    def test_load_csv(self, sample_imu_csv):
        """Test loading CSV file."""
        filepath, data_dir = sample_imu_csv
        filename = os.path.basename(filepath)
        
        training_data = TrainingData(filename, data_dir)
        
        assert training_data.df is not None
        assert len(training_data.df) == 100
    
    def test_feature_columns(self, sample_imu_csv):
        """Test feature column extraction."""
        filepath, data_dir = sample_imu_csv
        filename = os.path.basename(filepath)
        
        training_data = TrainingData(filename, data_dir)
        features = training_data.feature_columns
        
        # Should exclude time_stamp and sample_id
        assert 'time_stamp' not in features
        assert 'sample_id' not in features
        # Should include IMU columns
        assert 'ax_base' in features
        assert 'gx_base' in features
    
    def test_sorted_df(self, temp_dir):
        """Test dataframe sorting by timestamp."""
        # Create unsorted data
        data = {
            'time_stamp': [0.5, 0.1, 0.9, 0.2],
            'sample_id': [2, 0, 3, 1],
            'ax_base': [1, 2, 3, 4],
        }
        df = pd.DataFrame(data)
        
        filepath = os.path.join(temp_dir, 'unsorted.csv')
        df.to_csv(filepath, index=False)
        
        training_data = TrainingData('unsorted.csv', temp_dir)
        sorted_df = training_data.sorted_df
        
        # Check that timestamps are sorted
        timestamps = sorted_df['time_stamp'].values
        assert np.all(timestamps[:-1] <= timestamps[1:])
    
    def test_add_data(self, temp_dir):
        """Test adding additional data files."""
        # Create two CSV files
        data1 = {
            'time_stamp': [0.1, 0.2],
            'sample_id': [0, 1],
            'ax_base': [1, 2],
        }
        data2 = {
            'time_stamp': [0.3, 0.4],
            'sample_id': [2, 3],
            'ax_base': [3, 4],
        }
        
        file1 = os.path.join(temp_dir, 'data1.csv')
        file2 = os.path.join(temp_dir, 'data2.csv')
        pd.DataFrame(data1).to_csv(file1, index=False)
        pd.DataFrame(data2).to_csv(file2, index=False)
        
        training_data = TrainingData('data1.csv', temp_dir)
        original_len = len(training_data.df)
        
        training_data.add_data('data2.csv')
        
        assert len(training_data.df) == original_len + 2
        assert len(training_data.file_names) == 2


class TestPreprocessing:
    """Test Preprocessing class."""
    
    def test_initialization_with_both_hands(self, temp_dir):
        """Test initialization with both left and right hand data."""
        # Create minimal CSV files
        imu_data = pd.DataFrame({
            'time_stamp': [0.1, 0.2],
            'ax_base': [1, 2],
            'ay_base': [1, 2],
            'az_base': [1, 2],
            'gx_base': [1, 2],
            'gy_base': [1, 2],
            'gz_base': [1, 2],
        })
        keyboard_data = pd.DataFrame({
            'time': [0.0, 0.15],
            'name': ['a', 'b'],
            'event_type': ['down', 'down'],
        })
        
        left_file = os.path.join(temp_dir, 'left.csv')
        right_file = os.path.join(temp_dir, 'right.csv')
        keyboard_file = os.path.join(temp_dir, 'keyboard.csv')
        
        imu_data.to_csv(left_file, index=False)
        imu_data.to_csv(right_file, index=False)
        keyboard_data.to_csv(keyboard_file, index=False)
        
        preprocessor = Preprocessing(
            data_dir=temp_dir,
            keyboard_file='keyboard.csv',
            left_file='left.csv',
            right_file='right.csv'
        )
        
        assert preprocessor.has_left is True
        assert preprocessor.has_right is True
    
    def test_initialization_left_only(self, temp_dir):
        """Test initialization with only left hand data."""
        imu_data = pd.DataFrame({
            'time_stamp': [0.1, 0.2],
            'ax_base': [1, 2],
            'ay_base': [1, 2],
            'az_base': [1, 2],
            'gx_base': [1, 2],
            'gy_base': [1, 2],
            'gz_base': [1, 2],
        })
        keyboard_data = pd.DataFrame({
            'time': [0.0, 0.15],
            'name': ['a', 'b'],
            'event_type': ['down', 'down'],
        })
        
        left_file = os.path.join(temp_dir, 'left.csv')
        keyboard_file = os.path.join(temp_dir, 'keyboard.csv')
        
        imu_data.to_csv(left_file, index=False)
        keyboard_data.to_csv(keyboard_file, index=False)
        
        preprocessor = Preprocessing(
            data_dir=temp_dir,
            keyboard_file='keyboard.csv',
            left_file='left.csv'
        )
        
        assert preprocessor.has_left is True
        assert preprocessor.has_right is False
    
    def test_initialization_requires_one_hand(self, temp_dir):
        """Test that at least one hand is required."""
        keyboard_data = pd.DataFrame({
            'time': [0.0, 0.15],
            'name': ['a', 'b'],
            'event_type': ['down', 'down'],
        })
        keyboard_file = os.path.join(temp_dir, 'keyboard.csv')
        keyboard_data.to_csv(keyboard_file, index=False)
        
        with pytest.raises(ValueError):
            Preprocessing(
                data_dir=temp_dir,
                keyboard_file='keyboard.csv'
            )
    
    def test_char_from_key(self):
        """Test character extraction from key names."""
        assert Preprocessing._char_from_key('a') == 'a'
        assert Preprocessing._char_from_key('A') == 'a'
        assert Preprocessing._char_from_key('enter') == '\n'
        assert Preprocessing._char_from_key('space') == ' '
        assert Preprocessing._char_from_key('1') == '1'
    
    def test_char_from_key_handles_nan(self):
        """Test handling of NaN values."""
        assert Preprocessing._char_from_key(None) is None
        assert Preprocessing._char_from_key(np.nan) is None
    
    def test_char_to_label(self):
        """Test character to label conversion."""
        assert Preprocessing._char_to_label('a') == 0
        assert Preprocessing._char_to_label('z') == 25
        assert Preprocessing._char_to_label('0') == 26
        assert Preprocessing._char_to_label(' ') == 36
    
    def test_char_to_label_invalid(self):
        """Test invalid character returns None."""
        assert Preprocessing._char_to_label('@') is None
        assert Preprocessing._char_to_label('!') is None
    
    def test_label_to_char(self):
        """Test label to character conversion."""
        assert Preprocessing._label_to_char(0) == 'a'
        assert Preprocessing._label_to_char(25) == 'z'
        assert Preprocessing._label_to_char(36) == ' '
    
    def test_char_display(self):
        """Test character display formatting."""
        assert Preprocessing._char_display(' ') == '<space>'
        assert Preprocessing._char_display('\n') == '<enter>'
        assert Preprocessing._char_display('\t') == '<tab>'
        assert Preprocessing._char_display('\b') == '<backspace>'
        assert Preprocessing._char_display('a') == 'a'
    
    def test_pad_to_length(self):
        """Test sequence padding."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Pad to longer length
        padded = Preprocessing._pad_to_length(data, 5)
        assert padded.shape == (5, 2)
        assert np.array_equal(padded[:3], data)
        assert np.all(padded[3:] == 0)
        
        # Truncate to shorter length
        truncated = Preprocessing._pad_to_length(data, 2)
        assert truncated.shape == (2, 2)
        assert np.array_equal(truncated, data[:2])
    
    def test_combine_hands_both(self):
        """Test combining both hands data."""
        right = np.array([[1, 2], [3, 4]])
        left = np.array([[5, 6], [7, 8]])
        
        combined = Preprocessing._combine_hands(right, left, max_len=2)
        
        assert combined.shape == (2, 4)  # Concatenated features
        assert np.array_equal(combined[:, :2], right)
        assert np.array_equal(combined[:, 2:], left)
    
    def test_combine_hands_right_only(self):
        """Test with only right hand data."""
        right = np.array([[1, 2], [3, 4]])
        
        combined = Preprocessing._combine_hands(right, None, max_len=2)
        
        assert combined.shape == (2, 2)
        assert np.array_equal(combined, right)
    
    def test_combine_hands_left_only(self):
        """Test with only left hand data."""
        left = np.array([[5, 6], [7, 8]])
        
        combined = Preprocessing._combine_hands(None, left, max_len=2)
        
        assert combined.shape == (2, 2)
        assert np.array_equal(combined, left)
    
    def test_align_basic(self, temp_dir):
        """Test basic alignment functionality."""
        # Create synthetic data
        n_imu_samples = 100
        imu_data = pd.DataFrame({
            'time_stamp': np.linspace(0, 1.0, n_imu_samples),
            'ax_base': np.random.randn(n_imu_samples),
            'ay_base': np.random.randn(n_imu_samples),
            'az_base': np.random.randn(n_imu_samples),
            'gx_base': np.random.randn(n_imu_samples),
            'gy_base': np.random.randn(n_imu_samples),
            'gz_base': np.random.randn(n_imu_samples),
        })
        
        # Keyboard events at 0.0, 0.3, 0.6, 0.9
        keyboard_data = pd.DataFrame({
            'time': [0.0, 0.3, 0.6, 0.9],
            'name': ['a', 'b', 'c', 'd'],
            'event_type': ['down', 'down', 'down', 'down'],
        })
        
        left_file = os.path.join(temp_dir, 'left.csv')
        keyboard_file = os.path.join(temp_dir, 'keyboard.csv')
        imu_data.to_csv(left_file, index=False)
        keyboard_data.to_csv(keyboard_file, index=False)
        
        preprocessor = Preprocessing(
            data_dir=temp_dir,
            keyboard_file='keyboard.csv',
            left_file='left.csv'
        )
        
        samples, labels, prev_labels, metadata = preprocessor.align(max_seq_length=20)
        
        # Should have 3 samples (from 4 events, we get 3 windows)
        assert len(samples) == 3
        assert len(labels) == 3
        assert len(prev_labels) == 3
        
        # First label should be 'b' (index 1), second 'c' (index 2), third 'd' (index 3)
        assert labels[0] == CHAR_TO_INDEX['b']
        assert labels[1] == CHAR_TO_INDEX['c']
        assert labels[2] == CHAR_TO_INDEX['d']
        
        # Check metadata
        assert metadata['num_samples'] == 3
        assert metadata['num_classes'] == NUM_CLASSES
        assert metadata['has_left'] is True
        assert metadata['has_right'] is False
    
    def test_align_skips_invalid_chars(self, temp_dir):
        """Test that alignment skips characters not in vocabulary."""
        imu_data = pd.DataFrame({
            'time_stamp': [0.1, 0.2, 0.3],
            'ax_base': [1, 2, 3],
            'ay_base': [1, 2, 3],
            'az_base': [1, 2, 3],
            'gx_base': [1, 2, 3],
            'gy_base': [1, 2, 3],
            'gz_base': [1, 2, 3],
        })
        
        # Include invalid character '@'
        keyboard_data = pd.DataFrame({
            'time': [0.0, 0.15, 0.25],
            'name': ['a', '@', 'b'],
            'event_type': ['down', 'down', 'down'],
        })
        
        left_file = os.path.join(temp_dir, 'left.csv')
        keyboard_file = os.path.join(temp_dir, 'keyboard.csv')
        imu_data.to_csv(left_file, index=False)
        keyboard_data.to_csv(keyboard_file, index=False)
        
        preprocessor = Preprocessing(
            data_dir=temp_dir,
            keyboard_file='keyboard.csv',
            left_file='left.csv'
        )
        
        samples, labels, prev_labels, metadata = preprocessor.align()
        
        # Should skip the invalid character
        assert '@' in metadata['skipped_chars']
        # Should only have 1 sample (for 'b', skipping '@')
        assert len(samples) == 1
    
    def test_get_class_distribution(self, temp_dir):
        """Test class distribution computation."""
        imu_data = pd.DataFrame({
            'time_stamp': np.linspace(0, 1.0, 50),
            'ax_base': np.random.randn(50),
            'ay_base': np.random.randn(50),
            'az_base': np.random.randn(50),
            'gx_base': np.random.randn(50),
            'gy_base': np.random.randn(50),
            'gz_base': np.random.randn(50),
        })
        
        keyboard_data = pd.DataFrame({
            'time': [0.0, 0.2, 0.4, 0.6],
            'name': ['a', 'a', 'b', 'a'],
            'event_type': ['down', 'down', 'down', 'down'],
        })
        
        left_file = os.path.join(temp_dir, 'left.csv')
        keyboard_file = os.path.join(temp_dir, 'keyboard.csv')
        imu_data.to_csv(left_file, index=False)
        keyboard_data.to_csv(keyboard_file, index=False)
        
        preprocessor = Preprocessing(
            data_dir=temp_dir,
            keyboard_file='keyboard.csv',
            left_file='left.csv'
        )
        
        _, labels, _, _ = preprocessor.align()
        distribution = preprocessor.get_class_distribution(labels)
        
        # 'a' appears 2 times, 'b' appears 1 time
        assert distribution['a'] == 2
        assert distribution['b'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
