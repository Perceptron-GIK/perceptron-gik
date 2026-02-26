"""
GIK Preprocessing Pipeline for Real-Time Inference

1. IMU signal filtering (using src/imu/main.py)
2. Data alignment between left and right IMU sensors/FSRs (using src/pipeline/align.py)
3. Dataset creation and export for inference

"""

import numpy as np
import pandas as pd
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Any, Union

# Custom imports
from src.imu.main import IMUTracker
from src.inference.align import AlignData

IMU_SAMPLING_RATE = 100.0
IMU_COLS = [0, 6, 13, 20, 27, 34]
IMU_IDX_TO_PART = {
    0: "base",
    6: "thumb",
    13: "index",
    20: "middle",
    27: "ring",
    34: "pinky"
}

def filter_imu_data(data: np.ndarray) -> np.ndarray:
    '''
    Apply IMU filtering to a single window of IMU data from one hand
    Returns the processed array
    '''
    timestamps = data[:, -1]
    time_rel = timestamps - timestamps[0]

    tracker = IMUTracker(sr=IMU_SAMPLING_RATE, use_mag=False)
    filtered_data = data.copy()

    for imu_col in IMU_COLS:
        imu_data = np.column_stack([time_rel, data[:, imu_col:imu_col+6]])

        try:
            init_tuple = tracker.initialise(imu_data)
            if imu_col == 0: # Use the base IMU as a reference for the keyboard frame
                R0_ref, a, *_ = tracker.track_attitude(imu_data, init_tuple)
            else:
                _, a, *_ = tracker.track_attitude(imu_data, init_tuple, R0_ref=R0_ref)

            a_p = tracker.remove_acc_drift(a, threshold=0.2, filter=True, cof=(0.1, 5))
            vel = tracker.zupt(a_p, threshold=0.2)
            pos = tracker.track_position(a, vel)

            a_adjusted = np.nan_to_num(a_p, nan=0.0, posinf=0.0, neginf=0.0)
            pos_adjusted = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

            filtered_data[:, imu_col:imu_col+3] = a_adjusted
            filtered_data = np.concatenate((filtered_data[:, :-1], pos_adjusted, filtered_data[:, -1:]), axis=1)

        except Exception as e:
            print(f"Warning: Filtering failed for {IMU_IDX_TO_PART[imu_col]} IMU")
            filtered_data = np.concatenate((filtered_data[:, :-1], np.zeros((data.shape[0], 3)), filtered_data[:, -1:]), axis=1)

    return filtered_data

def preprocess(
    output_path: str,
    left_data: Optional[np.ndarray] = None,
    right_data: Optional[np.ndarray] = None,
    max_seq_length: int=100,
    normalize: bool=True,
    apply_filtering: bool=True
):
    '''
    Preprocess a single window of data and save it to output_path
    '''

# IMU filter is applied to each hand individually, before timestamps are removed
# Alignment is done assuming timestamps are the last column of each array
# Therefore IMU filter needs to insert position predictions before the last column of timestamps
# Alignment doesn't need to save any timestamps

def preprocess_multiple_sources(
    data_dir: str,
    output_path: str,
    keyboard_files: List[str],
    left_files: Optional[List[str]] = None,
    right_files: Optional[List[str]] = None,
    max_seq_length: int = 100,
    normalize: bool = True,
    apply_filtering: bool = True
) -> Dict[str, Any]:
    """
    Preprocess data from multiple source files and combine them.
    Alignment outputs labels as characters. Labels are saved as characters;
    mapping to index/vector is done in the Dataset via char_to_index (default CHAR_TO_INDEX).
    """
    if len(keyboard_files) == 0:
        raise ValueError("At least one keyboard file must be provided")
    
    if left_files is None:
        left_files = []
    if right_files is None:
        right_files = []
    
    # Ensure file lists match in length (or one is empty)
    has_left = len(left_files) > 0
    has_right = len(right_files) > 0
    
    if has_left and len(left_files) != len(keyboard_files):
        raise ValueError(f"Number of left files ({len(left_files)}) must match keyboard files ({len(keyboard_files)})")
    if has_right and len(right_files) != len(keyboard_files):
        raise ValueError(f"Number of right files ({len(right_files)}) must match keyboard files ({len(keyboard_files)})")
    
    print(f"Loading data from {data_dir}...")
    print(f"  Keyboard files: {keyboard_files}")
    if has_left:
        print(f"  Left IMU files: {left_files}")
    if has_right:
        print(f"  Right IMU files: {right_files}")

    # fsr columns
    if has_left and has_right:
        fsr_idx = [12, 19, 26, 33, 40, 71, 78, 85, 92, 99]
    else:
        fsr_idx = [12, 19, 26, 33, 40]
    
    all_samples = []
    all_labels = []
    all_prev_labels = []
    total_skipped = {}
    
    # Process each file combination
    for i in range(len(keyboard_files)):
        keyboard_file = keyboard_files[i]
        left_file = left_files[i] if has_left else None
        right_file = right_files[i] if has_right else None
        
        print(f"\nProcessing source {i+1}/{len(keyboard_files)}: {keyboard_file}")
        
        preprocessor = Preprocessing(
            data_dir=data_dir,
            keyboard_file=keyboard_file,
            right_file=right_file,
            left_file=left_file,
        )
        
        samples, labels, prev_labels, metadata = preprocessor.align(
            max_seq_length=max_seq_length,
            filter_func=filter_imu_data if apply_filtering else None
        )
        
        all_samples.extend(samples)
        all_labels.extend(labels)
        all_prev_labels.extend(prev_labels)
        
        # Accumulate skipped chars
        for char, count in metadata.get('skipped_chars', {}).items():
            total_skipped[char] = total_skipped.get(char, 0) + count
        
        print(f"  Added {len(samples)} samples (total: {len(all_samples)})")
    
    if total_skipped:
        print(f"\nTotal skipped characters: {total_skipped}")
    
    print(f"\nProcessing {len(all_samples)} total samples...")

    samples_tensor = [torch.tensor(s, dtype=torch.float32) for s in all_samples]
    if normalize: # Only normalise non-FSR features and non padded samples
        F = samples_tensor[0].shape[1]

        valid_rows = []
        for s in samples_tensor:
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) 
            mask_valid = (s.abs().sum(dim=1) > 0)
            if mask_valid.any():
                valid_rows.append(s[mask_valid])

        if valid_rows:
            all_data = torch.cat(valid_rows, dim=0)  
            mean = all_data.mean(dim=0)           
            std = all_data.std(dim=0)              
            std[std == 0] = 1.0
        else:
            mean = torch.zeros(F)
            std = torch.ones(F)

        mask = torch.ones(F, dtype=torch.bool)
        mask[fsr_idx] = False
        nonfsr = mask
        fsr = ~mask

        norm_samples = []
        for s in samples_tensor:
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            s_norm = s.clone()
            mask_valid = (s.abs().sum(dim=1) > 0)
            s_norm[mask_valid][:, nonfsr] = (s[mask_valid][:, nonfsr] - mean[nonfsr]) / std[nonfsr] # the forgotten line :o
            s_norm[mask_valid][:, fsr] = s[mask_valid][:, fsr]
            s_norm[~mask_valid] = 0.0

            norm_samples.append(s_norm)

        samples_tensor = norm_samples
    else:
        mean = std = None

    samples_stacked = torch.stack(samples_tensor) if samples_tensor else torch.tensor([])

    combined_metadata = {
        'num_samples': len(all_samples),
        'num_hands': metadata['num_hands'],
        'has_right': metadata['has_right'],
        'has_left': metadata['has_left'],
        'feat_dim': metadata['feat_dim'],
        'features_per_hand': metadata['features_per_hand'],
        'max_seq_length': max_seq_length,
        'skipped_chars': total_skipped,
        'num_sources': len(keyboard_files),
        'keyboard_files': keyboard_files,
        'left_files': left_files if has_left else [],
        'right_files': right_files if has_right else [],
        'filter_applied': apply_filtering,
    }

    save_dict = {
        'samples': samples_stacked,
        'labels': all_labels,
        'prev_labels': all_prev_labels,
        'mean': mean,
        'std': std,
        'metadata': combined_metadata,
        'normalize': normalize,
        'max_seq_length': max_seq_length,
    }
    
    torch.save(save_dict, output_path)
    
    json_path = output_path.replace('.pt', '_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    print(f"\nSaved preprocessed dataset to {output_path}")
    print(f"Saved metadata to {json_path}")
    print(f"  - Total samples: {combined_metadata['num_samples']}")
    print(f"  - Feat dim: {combined_metadata['feat_dim']}")
    print(f"  - Sources combined: {combined_metadata['num_sources']}")
    
    return combined_metadata

def load_preprocessed_dataset(
    path: str,
    char_to_index: Optional[Dict[str, LabelType]] = None,
    is_one_hot_labels: bool = False,
    add_prev_char: bool = True,
) -> 'PreprocessedGIKDataset':
    """
    Load a preprocessed dataset from disk. Labels are stored as characters; char_to_index is applied in the Dataset.

    Args:
        path: Path to the .pt file
        char_to_index: char -> int or char -> vector. If None, uses CHAR_TO_INDEX from alignment.
        one_hot_labels: If True (index mode only), __getitem__ returns one-hot labels.

    Returns:
        PreprocessedGIKDataset instance
    """
    return PreprocessedGIKDataset(path, char_to_index=char_to_index, is_one_hot_labels=is_one_hot_labels, add_prev_char=add_prev_char)

def export_dataset_to_csv(
    pt_path: str,
    output_dir: str = None,
    include_features: bool = True,
    max_samples: int = None,
    add_prev_char: bool = True
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
    
    data = torch.load(pt_path, weights_only=False)
    samples = data['samples']
    prev_labels = data['prev_labels'] if add_prev_char else None  # list of str (characters)
    labels = data['labels']  # list of str (characters)
    metadata = data['metadata']

    if output_dir is None:
        output_dir = os.path.dirname(pt_path)
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(samples)
    if max_samples:
        num_samples = min(num_samples, max_samples)

    def char_display(c: str) -> str:
        return {' ': 'SPACE', '\n': 'ENTER', '\b': 'BACKSPACE', '\t': 'TAB'}.get(c, c)

#### SUMMARY CSV ######
    summary_data = []
    for i in range(num_samples):
        sample = samples[i].numpy()
        char = labels[i] if i < len(labels) else '?'
        if add_prev_char:
            prev_char = prev_labels[i] if i < len(prev_labels) else '?'
            summary_data.append({
                'sample_idx': i,
                'character': char_display(char),
                'seq_length': (sample.sum(axis=1) != 0).sum(),
                'feature_mean': sample.mean(),
                'feature_std': sample.std(),
                'feature_min': sample.min(),
                'feature_max': sample.max(),
                'prev_character': char_display(prev_char),
            })
        else:
            summary_data.append({
                    'sample_idx': i,
                    'character': char_display(char),
                    'seq_length': (sample.sum(axis=1) != 0).sum(),
                    'feature_mean': sample.mean(),
                    'feature_std': sample.std(),
                    'feature_min': sample.min(),
                    'feature_max': sample.max(),
                })
        
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'dataset_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
#### DETAIL CSV ######
    if include_features:
        features_data = []
        for i in range(num_samples):
            sample = samples[i].numpy()
            char = labels[i] if i < len(labels) else '?'
            if add_prev_char:
                prev_char = prev_labels[i] if i < len(prev_labels) else '?'
            for t in range(sample.shape[0]):
                if add_prev_char:
                    row = {'sample_idx': i, 'timestep': t, 'character': char_display(char), 'feature_1_prev_character': char_display(prev_char)} 
                else:
                    row = {'sample_idx': i, 'timestep': t, 'character': char_display(char)} 
                for f in range(sample.shape[1]):
                    row[f'feature_{f+1}'] = sample[t, f]
                features_data.append(row)
        features_df = pd.DataFrame(features_data)
        features_path = os.path.join(output_dir, 'dataset_features.csv')
        features_df.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")

    print(f"\nDataset Info:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Exported samples: {num_samples}")
    print(f"  Feat dim: {metadata['feat_dim']}")
    print(f"  Max seq length: {data['max_seq_length']}")
    print(f"\nClass Distribution:")
    for char, count in summary_df['character'].value_counts().head(15).items():
        print(f"  {char}: {count}")
    return summary_path

class PreprocessedGIKDataset(Dataset):
    """
    Loads preprocessed data; labels are stored as characters. char_to_index (default CHAR_TO_INDEX)
    is applied in __getitem__ to get index or vector. one_hot_labels only applies when mapping is int.
    """

    def __init__(
        self,
        path: str,
        char_to_index: Optional[Dict[str, LabelType]] = None,
        is_one_hot_labels: bool = False,
        add_prev_char: bool = True,
    ):
        """
        Args:
            path: Path to preprocessed .pt file
            char_to_index: char -> int or char -> tuple (vector). If None, uses CHAR_TO_INDEX (same as alignment filter).
            one_hot_labels: If True and mapping is int, __getitem__ returns one-hot label.
        """
        data = torch.load(path, weights_only=False)
        self.samples = data['samples']
        self._labels = data['labels']  # list of str
        self._prev_labels = data['prev_labels']  # list of str, '' for no previous
        meta = data['metadata']
        feat_dim = meta['feat_dim']

        self._char_to_index = dict(char_to_index) if char_to_index is not None else dict(CHAR_TO_INDEX)  # alignment filters by CHAR_TO_INDEX so saved labels are in this vocab
        self.is_one_hot_labels = is_one_hot_labels

        # Infer mode from first mapping value
        first_val = next(iter(self._char_to_index.values()), None) 
        self._is_vector = isinstance(first_val, tuple)
        self._num_classes = max(self._char_to_index.values(), default=-1) + 1
        if self._is_vector:
            self._label_dim = len(first_val)

        else:
            self._label_dim = 1

        self.add_prev_char = add_prev_char
        if self.add_prev_char:
            self._input_dim = feat_dim + (self._num_classes if self.is_one_hot_labels else self._label_dim)
        else: 
            self._input_dim = feat_dim
        self.mean = data['mean']
        self.std = data['std']
        self.metadata = meta
        self.normalize = data['normalize']
        self.max_seq_length = data['max_seq_length']

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def __len__(self) -> int:
        return len(self.samples)

    def _char_to_rep(self, char: str):
        """Return representation (int or vector) for char; '' or unknown -> None for prev."""
        if not char or char not in self._char_to_index:
            return None
        return self._char_to_index[char]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx].clone()
        ## Generating Previous Character's label
        if self.add_prev_char :
            prev_char = self._prev_labels[idx]
            prev_rep = self._char_to_rep(prev_char)
            prev_embed = None
            if self._is_vector:
                prev_embed = torch.tensor(prev_rep, dtype=torch.float32) if prev_rep is not None else torch.zeros(self._label_dim, dtype=torch.float32)
            else:
                prev_idx = prev_rep if prev_rep is not None else -1 # previous class from 1 to 40
                prev_embed = F.one_hot(torch.tensor(prev_idx, dtype=torch.long), self._num_classes).float() if prev_idx >= 0 else torch.zeros(self._num_classes, dtype=torch.float32)
            ## Main logic to stack prev charecter with the IMU and FSR features
            prev_broadcast = prev_embed.unsqueeze(0).expand(sample.size(0), -1)
            sample = torch.cat([sample, prev_broadcast], dim=-1)

        ## Generating Label from character
        char = self._labels[idx]
        rep = self._char_to_rep(char)
        if rep is None:
            raise KeyError(f"Character {repr(char)} not in char_to_index")
        if self._is_vector:
            label = torch.tensor(rep, dtype=torch.float32)
        elif self.is_one_hot_labels:
            label = F.one_hot(torch.tensor(rep, dtype=torch.long), self._num_classes).float()
        else:
            label = torch.tensor(rep, dtype=torch.long)
        return sample, label
