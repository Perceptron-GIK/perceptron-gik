"""
GIK Preprocessing Pipeline for Real-Time Inference

1. IMU signal filtering (using src/imu/main.py)
2. Data alignment between left and right IMU sensors/FSRs (using src/inference/align.py)
3. Dimensionality reduction (using src/pre_processing/reduce_dim.py)

"""

import numpy as np
import torch
from typing import Optional

# Custom imports
from src.imu.main import IMUTracker
from src.inference.align import AlignData
from src.pre_processing.reduce_dim import reduce_dim

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
    left_data: Optional[np.ndarray] = None,
    right_data: Optional[np.ndarray] = None,
    max_seq_length: int=100,
    normalize: bool=True,
    apply_filtering: bool=True,
    reduce_dim: bool=True,
    dim_red_method: Optional[str]="pca",
    dims_ratio: Optional[float]=0.4,
    root_dir: Optional[str]=None
):
    '''
    Preprocess a single window of data

    Args:
        left_data: Array of data from the left hand
        right_data: Array of data from the right hand
        max_seq_length: Window length for data alignment
        normalize: Whether to normalise the data
        apply_filtering: Whether to filter IMU data
        reduce_dim: Whether to apply dimensionality reduction
        dim_red_method: Method of dimensionality reduction (if applicable)
        dims_ratio: Proportion of dimensions to keep (for PCA only)
    
    Returns:
        A single PyTorch tensor of processed data from both hands
    '''
    preprocessor = AlignData(
        left_data=left_data,
        right_data=right_data
    )

    samples, metadata = preprocessor.align(
        max_seq_length=max_seq_length,
        filter_fn = filter_imu_data if apply_filtering else None
    )

    if left_data is not None and right_data is not None:
        fsr_idx = [12, 19, 26, 33, 40, 71, 78, 85, 92, 99]
    else:
        fsr_idx = [12, 19, 26, 33, 40]   

    samples_tensor = [w for w in torch.tensor(samples, dtype=torch.float32)] # List of tensors
    if normalize:
        F = samples_tensor[0].shape[1]

        valid_rows = []

        for s in samples_tensor:
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) 
            mask_valid = (s.abs().sum(dim=1) > 0)
            if mask_valid.any():
                valid_rows.append(s[mask_valid])

        if valid_rows:
            all_data = torch.cat(valid_rows, dim=0)  
            x_min = all_data.amin(dim=0) # (F, )
            x_max = all_data.amax(dim=0) # (F, )
            range_ = x_max - x_min
            range_[range_ == 0] = 1.0
        else:
            x_min = torch.zeros(F)
            range_ = torch.ones(F)

        mask = torch.ones(F, dtype=torch.bool)
        mask[fsr_idx] = False
        nonfsr = mask
        fsr = ~mask

        norm_samples = []
        for s in samples_tensor:
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            s_norm = s.clone()
            mask_valid = (s.abs().sum(dim=1) > 0)
            valid_idx = mask_valid.nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                s_norm[valid_idx[:, None], nonfsr] = 2.0 * (
                    (s[valid_idx[:, None], nonfsr] - x_min[nonfsr]) / range_[nonfsr]
                ) - 1.0
                s_norm[valid_idx[:, None], fsr] = s[valid_idx[:, None], fsr]
            norm_samples.append(s_norm)
        samples_tensor = norm_samples
    
    samples = torch.stack(samples_tensor) # 3D tensor

    if reduce_dim:
        output = reduce_dim(
            data_source=samples,
            method=dim_red_method,
            has_left=metadata["has_left"],
            has_right=metadata["has_right"],
            normalize=normalize,
            dims_ratio=dims_ratio,
            root_dir=root_dir
        )
        return output
    else:
        return samples
