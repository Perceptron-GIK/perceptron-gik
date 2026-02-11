"""
GIK Preprocessing Pipeline

1. IMU signal filtering (using src/imu/v1/ modules)
2. Data alignment between IMU sensors and keyboard events (using src/pre_processing/alignment.py)
3. Dataset creation and export for ML training

Usage:
    from pretraining import preprocess_and_export, load_preprocessed_dataset
    
    # Preprocess and export dataset
    preprocess_and_export(
        data_dir="data_hazel_1/",
        keyboard_file="Keyboard_1.csv",
        left_file="Left_1.csv",
        output_path="data/processed_dataset.pt"
    )
    
    # Load in training script
    dataset = load_preprocessed_dataset("data/processed_dataset.pt")
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Any
import os
import sys
import json

# Import custom modules
from src.imu.v1.main import IMUTracker
from src.pre_processing.alignment import Preprocessing, INDEX_TO_CHAR, NUM_CLASSES


IMU_SAMPLING_RATE = 100.0
IMU_PARTS = ['base', 'thumb', 'index', 'middle', 'ring', 'pinky']


def filter_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply IMU filtering to each imu sensor data column (base, thumb, index, middle, ring, pinky) 
    and add the processed data to the dataframe. """
    df = df.copy()
    timestamps = df['time_stamp'].values
    time_rel = timestamps - timestamps[0]
    
    tracker = IMUTracker(sr=IMU_SAMPLING_RATE, use_mag=False)
    
    for part in IMU_PARTS:
        cols = [f'ax_{part}', f'ay_{part}', f'az_{part}', f'gx_{part}', f'gy_{part}', f'gz_{part}']
        if not all(c in df.columns for c in cols):
            continue
       
        data = np.column_stack([time_rel, df[cols].values])
        try:
            init_tuple = tracker.initialise(data)
            a, *_ = tracker.track_attitude(data, init_tuple)
            a_p = tracker.remove_acc_drift(a, threshold=0.2, filter=True, cof=(0.1, 5))
            vel = tracker.zupt(a_p, threshold=0.2)
            pos = tracker.track_position(a, vel)
            
            # Update with filtered values, replacing NaN/Inf with 0
            df[f'ax_{part}'] = np.nan_to_num(a[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            df[f'ay_{part}'] = np.nan_to_num(a[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
            df[f'az_{part}'] = np.nan_to_num(a[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
            df[f'x_{part}'] = np.nan_to_num(pos[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            df[f'y_{part}'] = np.nan_to_num(pos[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
            df[f'z_{part}'] = np.nan_to_num(pos[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
            
        except Exception as e:
            print(f"Warning: Filtering failed for {part}: {e}")
            # Add zero position columns if filtering fails
            df[f'x_{part}'] = 0.0
            df[f'y_{part}'] = 0.0
            df[f'z_{part}'] = 0.0
    
    return df


def preprocess_and_export(
    data_dir: str,
    output_path: str,
    keyboard_file: str,
    right_file: Optional[str] = None,
    left_file: Optional[str] = None,
    max_seq_length: int = 100,
    normalize: bool = True,
    apply_filtering: bool = True
) -> Dict[str, Any]:
    """
    Preprocess data and export to disk using Preprocessing class.
    
    Args:
        data_dir: Directory containing data files
        output_path: Path to save processed dataset (.pt file)
        keyboard_file: Keyboard events CSV filename
        right_file: Right hand IMU CSV filename (optional)
        left_file: Left hand IMU CSV filename (optional)
        max_seq_length: Maximum sequence length
        normalize: Whether to normalize features
        apply_filtering: Whether to apply IMU filtering
        
    Returns:
        Metadata dictionary
    """
    print(f"Loading data from {data_dir}...")
    preprocessor = Preprocessing(
        data_dir=data_dir,
        keyboard_file=keyboard_file,
        right_file=right_file,
        left_file=left_file
    )
    
    print(f"Aligning IMU data with keyboard events...")
    samples, labels, prev_labels, metadata = preprocessor.align(
        max_seq_length=max_seq_length,
        filter_func=filter_imu_data if apply_filtering else None
    )
    
    metadata['filter_applied'] = apply_filtering
    
    print(f"Processing {len(samples)} samples...")
    
    # Convert to tensors; store labels and prev_labels as one-hot
    samples_tensor = [torch.tensor(s, dtype=torch.float32) for s in samples]
    labels_index = torch.tensor(labels, dtype=torch.long)
    prev_labels_index = torch.tensor(prev_labels, dtype=torch.long)
    labels_onehot = F.one_hot(labels_index, num_classes=NUM_CLASSES).float()
    prev_labels_onehot = torch.zeros(len(prev_labels), NUM_CLASSES, dtype=torch.float32)
    valid_prev = prev_labels_index >= 0
    prev_labels_onehot[valid_prev] = F.one_hot(prev_labels_index[valid_prev], num_classes=NUM_CLASSES).float()
    
    # Compute normalization stats
    mean = None
    std = None
    if normalize:
        all_data = torch.cat([s for s in samples_tensor], dim=0)
        # Replace any remaining NaN/Inf with 0 before computing stats
        all_data = torch.nan_to_num(all_data, nan=0.0, posinf=0.0, neginf=0.0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        std[std == 0] = 1.0
        # Also clean the samples tensors
        samples_tensor = [torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) for s in samples_tensor]
    
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
    
    # Compute class weights (from indices)
    class_counts = torch.zeros(NUM_CLASSES)
    for label in labels:
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    
    # Save to disk (labels and prev_labels stored as one-hot)
    save_dict = {
        'samples': samples_stacked,
        'labels': labels_onehot,
        'prev_labels': prev_labels_onehot,
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
    print(f"  - Samples: {metadata['num_samples']}")
    print(f"  - Input dim: {metadata['input_dim']}")
    print(f"  - Hands: {metadata.get('num_hands', (1 if metadata.get('has_right') else 0) + (1 if metadata.get('has_left') else 0))}")
    
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
    
    # Labels may be one-hot (N, num_classes) or indices (N,)
    def label_to_idx(lab, i):
        if lab.dim() == 2:
            return lab[i].argmax().item()
        return lab[i].item()
    
    # Create summary DataFrame
    summary_data = []
    for i in range(num_samples):
        sample = samples[i].numpy()
        label_idx = label_to_idx(labels, i)
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
            label_idx = label_to_idx(labels, i)
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
        labels = data['labels']
        prev_labels = data.get('prev_labels')
        num_classes = data['metadata']['num_classes']
        # Ensure one-hot: convert legacy index tensors (1D) to one-hot
        if labels.dim() == 1:
            self.labels = F.one_hot(labels, num_classes=num_classes).float()
        else:
            self.labels = labels
        if prev_labels is None:
            self.prev_labels = torch.zeros(len(self.samples), num_classes, dtype=torch.float32)
        elif prev_labels.dim() == 1:
            self.prev_labels = torch.zeros(len(self.samples), num_classes, dtype=torch.float32)
            valid = prev_labels >= 0
            self.prev_labels[valid] = F.one_hot(prev_labels[valid], num_classes=num_classes).float()
        else:
            self.prev_labels = prev_labels
        self.mean = data['mean']
        self.std = data['std']
        self.class_weights = data['class_weights']
        self.metadata = data['metadata']
        self.normalize = data['normalize']
        self.max_seq_length = data['max_seq_length']
        
        # Expose key properties (input_dim = feat + num_classes for prev-char)
        self._input_dim = self.metadata['input_dim']
        self.has_right = self.metadata.get('has_right', False)
        self.has_left = self.metadata.get('has_left', False)
        self.num_hands = self.metadata.get('num_hands', (1 if self.has_right else 0) + (1 if self.has_left else 0))
    
    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return self._input_dim
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample (with prev-char one-hot concat) and its one-hot label. Labels and prev_labels are one-hot."""
        sample = self.samples[idx].clone()
        if self.normalize and self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
        sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

        prev_onehot = self.prev_labels[idx].clone()
        prev_broadcast = prev_onehot.unsqueeze(0).expand(sample.size(0), -1)
        sample = torch.cat([sample, prev_broadcast], dim=-1)

        label = self.labels[idx].clone()
        return sample, label
    
    def get_class_weights(self) -> torch.Tensor:
        return self.class_weights
