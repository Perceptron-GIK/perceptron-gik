"""
GIK Preprocessing Pipeline for Real-Time Inference

1. IMU signal filtering (using src/imu/main.py)
2. Data alignment between left and right IMU sensors/FSRs (using src/inference/align.py)
3. Dimensionality reduction (using src/pre_processing/reduce_dim.py)
4. Expansion of data window with previous character prediction as a feature

"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Any

# Custom imports
from src.imu.main import IMUTracker
from src.inference.align import AlignData
from src.pre_processing.reduce_dim import reduce_dim
from src.Constants.char_to_key import (
    NUM_CLASSES, CHAR_TO_INDEX, INDEX_TO_CHAR, FULL_COORDS,
    NUM_CLASSES_4, CHAR_TO_INDEX_4, INDEX_TO_CHAR_4, FULL_COORDS_4,
    NUM_CLASSES_DIAGONAL, CHAR_TO_INDEX_DIAGONAL, INDEX_TO_CHAR_DIAGONAL, FULL_COORDS_DIAGONAL,
)

IMU_SAMPLING_RATE = 28.57
IMU_COLS = [0, 6, 13, 20, 27, 34]
IMU_IDX_TO_PART = {
    0: "base",
    6: "thumb",
    13: "index",
    20: "middle",
    27: "ring",
    34: "pinky"
}

def filter_imu_data(data: np.ndarray, part) -> np.ndarray:
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

            try:
                a_p = tracker.remove_acc_drift(a, threshold=0.2, filter=True, cof=(0.1, 5))
            except:
                a_p = tracker.remove_acc_drift(a, threshold=0.2, filter=False, cof=(0.1, 5))
            vel = tracker.zupt(a_p, threshold=0.2)
            pos = tracker.track_position(a, vel, IMU_IDX_TO_PART[imu_col] + part)

            a_adjusted = np.nan_to_num(a_p, nan=0.0, posinf=0.0, neginf=0.0)
            pos_adjusted = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

            filtered_data[:, imu_col:imu_col+3] = a_adjusted
            filtered_data = np.concatenate((filtered_data[:, :-1], pos_adjusted, filtered_data[:, -1:]), axis=1)

        except Exception as e:
            print(f"Warning: Filtering failed for {IMU_IDX_TO_PART[imu_col]} IMU, {e}")
            filtered_data = np.concatenate((filtered_data[:, :-1], np.zeros((data.shape[0], 3)), filtered_data[:, -1:]), axis=1)

    return filtered_data

def add_prev_char(data, prev_char, mode, num_classes=None, char_to_index=None, index_to_char=None, full_coords=None):
    '''
    Adds the previous character prediction to the current window of data

    Args:
        data: 3D PyTorch tensor containing a single window of preprocessed data
        prev_char: Index of the previous character prediction
        mode: "classification", "classification_4", "classification_diagonal", or "regression"
        num_classes, char_to_index, index_to_char, full_coords: optional overrides for 4-class mode
    '''
    n_classes = num_classes or NUM_CLASSES
    c2i = char_to_index or CHAR_TO_INDEX
    i2c = index_to_char or INDEX_TO_CHAR
    coords = full_coords or FULL_COORDS
    nRows = data.shape[1]
    if prev_char is None:
        if mode in ("classification", "classification_4", "classification_diagonal"):
            space = F.one_hot(torch.tensor([c2i[" "]]), num_classes=n_classes).float().unsqueeze(1).repeat(1, nRows, 1)
        else:
            space = torch.tensor([coords[" "]]).float().unsqueeze(1).repeat(1, nRows, 1)
        return torch.cat((data, space), dim=2)
    else:
        if mode in ("classification", "classification_4", "classification_diagonal"):
            prev_char = F.one_hot(torch.tensor([prev_char]), num_classes=n_classes).float().unsqueeze(1).repeat(1, nRows, 1)
        else:
            prev_char = torch.tensor([coords[i2c[prev_char]]]).float().unsqueeze(1).repeat(1, nRows, 1)
        return torch.cat((data, prev_char), dim=2)

def preprocess(
    left_data: Optional[np.ndarray] = None,
    right_data: Optional[np.ndarray] = None,
    left_pointer: int=None,
    right_pointer: int=None,
    prev_char: Any=None,
    mode: str="classification",
    max_seq_length: int=100,
    normalize: bool=True,
    apply_filtering: bool=True,
    apply_dim_reduction: bool=True,
    dim_red_method: Optional[str]="pca",
    dims_ratio: Optional[float]=0.4,
    root_dir: Optional[str]=None,
    training_dataset: str = None,
    append_prev_char: bool = True,
):
    '''
    Preprocess a single window of data for real-time inference.

    LM fusion support: When lm_fusion_inference is enabled in train config,
    the caller (inference_receiver) passes prev_char from the last prediction
    and builds history_chars for n-gram LM context. Set append_prev_char=True
    when the model was trained with append_prev_char_feature (required for
    simple models; optional for GIK models).

    Args:
        left_data: Array of data from the left hand
        right_data: Array of data from the right hand
        left_pointer: Index at which the current left hand inference window starts
        right_pointer: Index at which the current right hand inference window starts
        prev_char: Index of previous character prediction (for add_prev_char and LM history)
        mode: "classification" or "regression"
        max_seq_length: Window length for data alignment
        normalize: Whether to normalise the data
        apply_filtering: Whether to filter IMU data
        apply_dim_reduction: Whether to apply dimensionality reduction
        dim_red_method: Method of dimensionality reduction (if applicable)
        dims_ratio: Proportion of dimensions to keep (for PCA only)
        root_dir: Project root containing PCA params file
        training_dataset: File path to the processed dataset from training
        append_prev_char: If True, append prev_char one-hot/coords to input (match training)

    Returns:
        A single PyTorch tensor of processed data from both hands
    '''
    preprocessor = AlignData(
        left_data=left_data,
        right_data=right_data
    )

    samples, metadata = preprocessor.align(
        max_seq_length = max_seq_length,
        filter_fn = filter_imu_data if apply_filtering else None,
        left_pointer = left_pointer,
        right_pointer = right_pointer
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

    if apply_dim_reduction:
        output = reduce_dim(
            data_dir=training_dataset,
            inference_source=samples,
            method=dim_red_method,
            has_left=metadata["has_left"],
            has_right=metadata["has_right"],
            normalize=normalize,
            dims_ratio=dims_ratio,
            root_dir=root_dir
        )
        if append_prev_char:
            kwargs = {}
            if mode == "classification_4":
                kwargs = dict(num_classes=NUM_CLASSES_4, char_to_index=CHAR_TO_INDEX_4, index_to_char=INDEX_TO_CHAR_4, full_coords=FULL_COORDS_4)
            elif mode == "classification_diagonal":
                kwargs = dict(num_classes=NUM_CLASSES_DIAGONAL, char_to_index=CHAR_TO_INDEX_DIAGONAL, index_to_char=INDEX_TO_CHAR_DIAGONAL, full_coords=FULL_COORDS_DIAGONAL)
            return add_prev_char(output, prev_char, mode, **kwargs)
        return output
    else:
        if append_prev_char:
            kwargs = {}
            if mode == "classification_4":
                kwargs = dict(num_classes=NUM_CLASSES_4, char_to_index=CHAR_TO_INDEX_4, index_to_char=INDEX_TO_CHAR_4, full_coords=FULL_COORDS_4)
            elif mode == "classification_diagonal":
                kwargs = dict(num_classes=NUM_CLASSES_DIAGONAL, char_to_index=CHAR_TO_INDEX_DIAGONAL, index_to_char=INDEX_TO_CHAR_DIAGONAL, full_coords=FULL_COORDS_DIAGONAL)
            return add_prev_char(samples, prev_char, mode, **kwargs)
        return samples
