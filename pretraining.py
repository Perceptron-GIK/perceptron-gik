"""
GIK Preprocessing Pipeline

This module handles all data preprocessing for the GIK (Gesture Input Keyboard) system:
1. IMU signal filtering (using src/imu/v1/ modules)
2. Data alignment between IMU sensors and keyboard events
3. Dataset creation and export for ML training

Usage:
    # Preprocess and export dataset
    from pretraining import preprocess_and_export
    
    preprocess_and_export(
        keyboard_csv="data/Keyboard_2.csv",
        right_csv="data/Right_2.csv",
        left_csv="data/Left_2.csv",  # Optional
        output_path="data/processed_dataset.pt"
    )
    
    # Load in training script
    from pretraining import load_preprocessed_dataset
    dataset = load_preprocessed_dataset("data/processed_dataset.pt")
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import os
import sys
import json

# Add src to path to import IMU modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'imu', 'v1'))

# Import IMU filtering from existing modules
from main import IMUTracker
from mathlib import filter_signal


# ============================================================================
# Constants and Configuration
# ============================================================================

NUM_CLASSES = 40
FEATURES_PER_HAND = 41
TOTAL_FEATURES = FEATURES_PER_HAND * 2

# Scan code to character mapping
SCAN_CODE_TO_CHAR = {
    0: 'a', 1: 's', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'j', 7: 'k', 8: 'l',
    9: 'q', 10: 'w', 11: 'e', 12: 'r', 13: 't', 14: 'y', 15: 'u', 16: 'i', 17: 'o', 18: 'p',
    29: 'z', 30: 'x', 31: 'c', 32: 'v', 33: 'b', 34: 'n', 35: 'm',
    37: 'l', 38: ';', 45: 'n', 46: 'm',
    19: '1', 20: '2', 21: '3', 22: '4', 23: '5', 24: '6', 25: '7', 26: '8', 27: '9', 28: '0',
    49: ' ', 36: '\n', 51: '\b', 48: '\t',
}

# Character to index mapping for one-hot encoding
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


# ============================================================================
# IMU Filter Configuration
# ============================================================================

@dataclass
class IMUFilterConfig:
    """Configuration for IMU signal filtering."""
    apply_kalman: bool = True
    apply_drift_removal: bool = True
    apply_bandpass: bool = False
    bandpass_cutoff: Tuple[float, float] = (0.1, 15)
    noise_coefficients: Dict[str, float] = None
    drift_threshold: float = 0.2
    use_magnetometer: bool = False
    
    def __post_init__(self):
        if self.noise_coefficients is None:
            self.noise_coefficients = {'g': 10, 'a': 35, 'm': 10}


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    """Estimate sampling rate from timestamps."""
    diffs = np.diff(timestamps)
    mean_dt = np.mean(diffs)
    return 1.0 / mean_dt if mean_dt > 0 else 100.0


def filter_imu_data(
    df: pd.DataFrame, 
    config: IMUFilterConfig = None
) -> pd.DataFrame:
    """
    Apply IMU filtering pipeline to raw sensor data using src/imu/v1/ modules.
    
    Args:
        df: DataFrame with IMU data (must have time_stamp and ax_base, ay_base, etc.)
        config: Filter configuration
        
    Returns:
        DataFrame with filtered acceleration columns added
    """
    if config is None:
        config = IMUFilterConfig()
    
    if not config.apply_kalman:
        return df
    
    # Estimate sampling rate
    timestamps = df['time_stamp'].values
    sr = estimate_sampling_rate(timestamps)
    
    # Get column names for base accelerometer and gyroscope
    base_cols = ['ax_base', 'ay_base', 'az_base', 'gx_base', 'gy_base', 'gz_base']
    
    if not all(col in df.columns for col in base_cols):
        print("Warning: Base IMU columns not found, skipping Kalman filter")
        return df
    
    # Create time column (relative seconds)
    time_rel = timestamps - timestamps[0]
    
    # Prepare data array for tracker
    data = np.column_stack([
        time_rel,
        df[base_cols].values
    ])
    
    # Initialize and run tracker (using imported IMUTracker from src/imu/v1/main.py)
    tracker = IMUTracker(sr=sr, use_mag=config.use_magnetometer)
    
    try:
        init_tuple = tracker.initialise(data, config.noise_coefficients)
        a_world, *_ = tracker.track_attitude(data, init_tuple)
        
        if config.apply_drift_removal:
            a_world = tracker.remove_acc_drift(
                a_world, 
                threshold=config.drift_threshold,
                filter=config.apply_bandpass,
                cof=config.bandpass_cutoff
            )
        
        # Replace original acceleration columns with filtered values
        df = df.copy()
        df['ax_base'] = a_world[:, 0]
        df['ay_base'] = a_world[:, 1]
        df['az_base'] = a_world[:, 2]
        
    except Exception as e:
        print(f"Warning: Kalman filter failed: {e}")
    
    return df


def interpolate_sequence(data: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate sequence to target length."""
    current_len = len(data)
    
    if current_len == target_len:
        return data
    
    if current_len == 0:
        return np.zeros((target_len, data.shape[1] if len(data.shape) > 1 else 1))
    
    old_indices = np.linspace(0, current_len - 1, current_len)
    new_indices = np.linspace(0, current_len - 1, target_len)
    
    if len(data.shape) == 1:
        return np.interp(new_indices, old_indices, data)
    
    interpolated = np.zeros((target_len, data.shape[1]))
    for f in range(data.shape[1]):
        interpolated[:, f] = np.interp(new_indices, old_indices, data[:, f])
    
    return interpolated


def align_imu_with_keyboard(
    keyboard_df: pd.DataFrame,
    right_df: Optional[pd.DataFrame] = None,
    left_df: Optional[pd.DataFrame] = None,
    max_seq_length: int = 100,
    filter_config: IMUFilterConfig = None
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """
    Align IMU data with keyboard events to create labeled samples.
    
    Args:
        keyboard_df: Keyboard events DataFrame
        right_df: Right hand IMU DataFrame (optional)
        left_df: Left hand IMU DataFrame (optional)
        max_seq_length: Maximum sequence length
        filter_config: IMU filter configuration
        
    Returns:
        Tuple of (samples, labels, metadata)
    """
    if right_df is None and left_df is None:
        raise ValueError("At least one of right_df or left_df must be provided")
    
    samples = []
    labels = []
    
    # Apply filtering if enabled
    if filter_config is not None:
        if right_df is not None:
            right_df = filter_imu_data(right_df, filter_config)
        if left_df is not None:
            left_df = filter_imu_data(left_df, filter_config)
    
    # Filter keyboard events to only 'down' events
    key_events = keyboard_df[keyboard_df['event_type'] == 'down'].copy()
    key_events = key_events.sort_values('time').reset_index(drop=True)
    
    # Determine feature columns
    has_right = right_df is not None
    has_left = left_df is not None
    
    right_feature_cols = None
    left_feature_cols = None
    right_sorted = None
    left_sorted = None
    
    if has_right:
        right_feature_cols = [c for c in right_df.columns 
                            if c not in ['sample_id', 'time_stamp']]
        right_sorted = right_df.sort_values('time_stamp').reset_index(drop=True)
    
    if has_left:
        left_feature_cols = [c for c in left_df.columns 
                           if c not in ['sample_id', 'time_stamp']]
        left_sorted = left_df.sort_values('time_stamp').reset_index(drop=True)
    
    # Process each key press window
    for i in range(len(key_events) - 1):
        current_event = key_events.iloc[i]
        next_event = key_events.iloc[i + 1]
        
        scan_code = int(next_event['scan_code'])
        char = SCAN_CODE_TO_CHAR.get(scan_code, '')
        
        if char not in CHAR_TO_INDEX:
            continue
        
        label = CHAR_TO_INDEX[char]
        start_time = current_event['time']
        end_time = next_event['time']
        
        # Extract IMU windows
        right_window = None
        left_window = None
        
        if has_right:
            mask = (right_sorted['time_stamp'] >= start_time) & (right_sorted['time_stamp'] < end_time)
            right_window = right_sorted.loc[mask, right_feature_cols].values
            if len(right_window) == 0:
                continue
        
        if has_left:
            mask = (left_sorted['time_stamp'] >= start_time) & (left_sorted['time_stamp'] < end_time)
            left_window = left_sorted.loc[mask, left_feature_cols].values
            if len(left_window) == 0:
                continue
        
        # Combine hand data
        combined = _combine_hand_data(right_window, left_window, max_seq_length)
        
        if combined is not None:
            samples.append(combined)
            labels.append(label)
    
    # Metadata
    num_hands = int(has_right) + int(has_left)
    input_dim = FEATURES_PER_HAND * num_hands
    
    metadata = {
        'num_samples': len(samples),
        'num_hands': num_hands,
        'has_right': has_right,
        'has_left': has_left,
        'input_dim': input_dim,
        'max_seq_length': max_seq_length,
        'num_classes': NUM_CLASSES,
        'filter_applied': filter_config is not None and filter_config.apply_kalman,
    }
    
    return samples, labels, metadata


def _combine_hand_data(
    right_data: Optional[np.ndarray], 
    left_data: Optional[np.ndarray],
    max_seq_length: int
) -> Optional[np.ndarray]:
    """Combine or process hand IMU data."""
    has_right = right_data is not None and len(right_data) > 0
    has_left = left_data is not None and len(left_data) > 0
    
    if not has_right and not has_left:
        return None
    
    if has_right and not has_left:
        target_len = min(len(right_data), max_seq_length)
        return interpolate_sequence(right_data, target_len)
    
    if has_left and not has_right:
        target_len = min(len(left_data), max_seq_length)
        return interpolate_sequence(left_data, target_len)
    
    # Both hands
    target_len = max(len(right_data), len(left_data))
    target_len = min(target_len, max_seq_length)
    
    right_interp = interpolate_sequence(right_data, target_len)
    left_interp = interpolate_sequence(left_data, target_len)
    
    return np.concatenate([right_interp, left_interp], axis=1)


# ============================================================================
# Dataset Export/Import
# ============================================================================

def preprocess_and_export(
    keyboard_csv: str,
    output_path: str,
    right_csv: Optional[str] = None,
    left_csv: Optional[str] = None,
    max_seq_length: int = 100,
    normalize: bool = True,
    apply_filtering: bool = True,
    filter_config: IMUFilterConfig = None
) -> Dict[str, Any]:
    """
    Preprocess data and export to disk.
    
    Args:
        keyboard_csv: Path to keyboard events CSV
        output_path: Path to save processed dataset (.pt file)
        right_csv: Path to right hand IMU CSV (optional)
        left_csv: Path to left hand IMU CSV (optional)
        max_seq_length: Maximum sequence length
        normalize: Whether to normalize features
        apply_filtering: Whether to apply IMU filtering
        filter_config: Custom filter configuration
        
    Returns:
        Metadata dictionary
    """
    if right_csv is None and left_csv is None:
        raise ValueError("At least one of right_csv or left_csv must be provided")
    
    print(f"Loading data...")
    keyboard_df = pd.read_csv(keyboard_csv)
    right_df = pd.read_csv(right_csv) if right_csv else None
    left_df = pd.read_csv(left_csv) if left_csv else None
    
    # Configure filtering
    if apply_filtering and filter_config is None:
        filter_config = IMUFilterConfig()
    elif not apply_filtering:
        filter_config = None
    
    print(f"Aligning IMU data with keyboard events...")
    samples, labels, metadata = align_imu_with_keyboard(
        keyboard_df=keyboard_df,
        right_df=right_df,
        left_df=left_df,
        max_seq_length=max_seq_length,
        filter_config=filter_config
    )
    
    print(f"Processing {len(samples)} samples...")
    
    # Convert to tensors
    samples_tensor = [torch.tensor(s, dtype=torch.float32) for s in samples]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Compute normalization stats
    mean = None
    std = None
    if normalize and len(samples_tensor) > 0:
        all_data = torch.cat([s for s in samples_tensor], dim=0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        std[std == 0] = 1.0
    
    # Pad sequences to uniform length
    padded_samples = []
    for sample in samples_tensor:
        seq_len = sample.shape[0]
        if seq_len >= max_seq_length:
            padded = sample[:max_seq_length]
        else:
            padding = torch.zeros(max_seq_length - seq_len, sample.shape[1])
            padded = torch.cat([sample, padding], dim=0)
        padded_samples.append(padded)
    
    # Stack into single tensor
    if len(padded_samples) > 0:
        samples_stacked = torch.stack(padded_samples)
    else:
        samples_stacked = torch.tensor([])
    
    # Compute class weights
    class_counts = torch.zeros(NUM_CLASSES)
    for label in labels:
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    
    # Save to disk
    save_dict = {
        'samples': samples_stacked,
        'labels': labels_tensor,
        'mean': mean,
        'std': std,
        'class_weights': class_weights,
        'metadata': metadata,
        'normalize': normalize,
        'max_seq_length': max_seq_length,
    }
    
    torch.save(save_dict, output_path)
    
    # Also save metadata as JSON for easy inspection
    json_path = output_path.replace('.pt', '_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved preprocessed dataset to {output_path}")
    print(f"Saved metadata to {json_path}")
    print(f"  - Samples: {len(samples)}")
    print(f"  - Input dim: {metadata['input_dim']}")
    print(f"  - Hands: {metadata['num_hands']}")
    
    return metadata


def load_preprocessed_dataset(path: str) -> 'PreprocessedGIKDataset':
    """
    Load a preprocessed dataset from disk.
    
    Args:
        path: Path to the .pt file
        
    Returns:
        PreprocessedGIKDataset instance
    """
    return PreprocessedGIKDataset(path)


def export_dataset_to_csv(
    pt_path: str,
    output_dir: str = None,
    include_features: bool = True,
    max_samples: int = None
) -> str:
    """
    Export preprocessed dataset to CSV for inspection.
    
    Creates two CSV files:
    - dataset_samples.csv: Flattened features with labels
    - dataset_summary.csv: Summary statistics per sample
    
    Args:
        pt_path: Path to the .pt file
        output_dir: Directory for output CSVs (default: same as pt_path)
        include_features: Whether to include all features (can be large)
        max_samples: Limit number of samples to export
        
    Returns:
        Path to the summary CSV file
    """
    import pandas as pd
    
    # Load dataset
    data = torch.load(pt_path, weights_only=False)
    samples = data['samples']
    labels = data['labels']
    metadata = data['metadata']
    
    if output_dir is None:
        output_dir = os.path.dirname(pt_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(samples)
    if max_samples:
        num_samples = min(num_samples, max_samples)
    
    # Create summary DataFrame
    summary_data = []
    for i in range(num_samples):
        sample = samples[i].numpy()
        label_idx = labels[i].item()
        char = INDEX_TO_CHAR.get(label_idx, '?')
        
        # Display-friendly character names
        if char == ' ':
            char_display = 'SPACE'
        elif char == '\n':
            char_display = 'ENTER'
        elif char == '\b':
            char_display = 'BACKSPACE'
        elif char == '\t':
            char_display = 'TAB'
        else:
            char_display = char
        
        summary_data.append({
            'sample_idx': i,
            'label_idx': label_idx,
            'character': char_display,
            'seq_length': (sample.sum(axis=1) != 0).sum(),  # Non-zero rows
            'feature_mean': sample.mean(),
            'feature_std': sample.std(),
            'feature_min': sample.min(),
            'feature_max': sample.max(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'dataset_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Create detailed features CSV (optional, can be large)
    if include_features:
        features_data = []
        for i in range(num_samples):
            sample = samples[i].numpy()
            label_idx = labels[i].item()
            char = INDEX_TO_CHAR.get(label_idx, '?')
            
            if char == ' ':
                char_display = 'SPACE'
            elif char == '\n':
                char_display = 'ENTER'
            elif char == '\b':
                char_display = 'BACKSPACE'
            elif char == '\t':
                char_display = 'TAB'
            else:
                char_display = char
            
            # Add each timestep as a row
            for t in range(sample.shape[0]):
                row = {
                    'sample_idx': i,
                    'timestep': t,
                    'label_idx': label_idx,
                    'character': char_display,
                }
                # Add feature columns
                for f in range(sample.shape[1]):
                    row[f'feature_{f}'] = sample[t, f]
                features_data.append(row)
        
        features_df = pd.DataFrame(features_data)
        features_path = os.path.join(output_dir, 'dataset_features.csv')
        features_df.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
    
    # Print metadata
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Exported samples: {num_samples}")
    print(f"  Input dim: {metadata['input_dim']}")
    print(f"  Max seq length: {data['max_seq_length']}")
    print(f"  Num classes: {metadata['num_classes']}")
    
    # Print class distribution
    print(f"\nClass Distribution:")
    for char, count in summary_df['character'].value_counts().head(15).items():
        print(f"  {char}: {count}")
    
    return summary_path


# ============================================================================
# PyTorch Dataset for Preprocessed Data
# ============================================================================

class PreprocessedGIKDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed data from disk.
    
    This is the dataset class used for training after preprocessing.
    """
    
    def __init__(self, path: str):
        """
        Args:
            path: Path to preprocessed .pt file
        """
        data = torch.load(path, weights_only=False)
        
        self.samples = data['samples']
        self.labels = data['labels']
        self.mean = data['mean']
        self.std = data['std']
        self.class_weights = data['class_weights']
        self.metadata = data['metadata']
        self.normalize = data['normalize']
        self.max_seq_length = data['max_seq_length']
        
        # Expose key properties
        self._input_dim = self.metadata['input_dim']
        self.num_hands = self.metadata['num_hands']
        self.has_right = self.metadata['has_right']
        self.has_left = self.metadata['has_left']
    
    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return self._input_dim
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample and its one-hot label."""
        sample = self.samples[idx]
        
        # Normalize if enabled
        if self.normalize and self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
        
        # Create one-hot label
        label = torch.zeros(NUM_CLASSES)
        label[self.labels[idx]] = 1.0
        
        return sample, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for imbalanced data."""
        return self.class_weights


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GIK Data Preprocessing")
    parser.add_argument("--keyboard", "-k", required=True, help="Path to keyboard CSV")
    parser.add_argument("--right", "-r", help="Path to right hand IMU CSV")
    parser.add_argument("--left", "-l", help="Path to left hand IMU CSV")
    parser.add_argument("--output", "-o", default="data/processed_dataset.pt", help="Output path")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Max sequence length")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--no-filter", action="store_true", help="Disable IMU filtering")
    
    args = parser.parse_args()
    
    if args.right is None and args.left is None:
        parser.error("At least one of --right or --left must be provided")
    
    preprocess_and_export(
        keyboard_csv=args.keyboard,
        right_csv=args.right,
        left_csv=args.left,
        output_path=args.output,
        max_seq_length=args.max_seq_length,
        normalize=not args.no_normalize,
        apply_filtering=not args.no_filter
    )
