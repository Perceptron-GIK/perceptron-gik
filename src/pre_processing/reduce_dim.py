import torch
from typing import Dict, Optional

def reduce_dim(
        data_source,
        method: str,
        has_left: bool,
        has_right: bool,
        normalize: bool,
        output_path: Optional[str] = None,
        dims_ratio: Optional[float] = None
) -> Dict[str, int]:
    """
    Applies dimensionality reduction to the preprocessed dataset

    Args:
        data_source: Either the directory containing preprocessed dataset or a PyTorch tensor
        method: Method of dimensionality reduction (helper function must be defined below)
        has_left: Whether data from left hand is present
        has_right: Whether data from right hand is present
        normalize: Whether to normalise features
        output_path: Path of output file generated from dimensionality reduction (if applicable)
        dims_ratio: Proportion of dimensions to keep (for PCA only)

    Returns:
        If reading data from a PyTorch file:
            dims dictionary with feature dimension before and after dimensionality reduction
        Else:
            PyTorch tensor containing data after dimensionality reduction
            Retained dimensions (for PCA only)
    """

    if isinstance(data_source, str):
        data = torch.load(data_source)
    else:
        data = data_source

    if method == "active-imu":
        return active_imu_only(data, has_left, has_right, normalize, output_path)
    elif method == "pca":
        return pca(data, dims_ratio, output_path)
    else:
        raise Exception("Invalid dimensionality reduction method.")

# Reduces feature dimension by keeping only data from the active and base IMUs
def active_imu_only(data, has_left, has_right, normalize, output_path):
    if isinstance(data, dict):
        samples = data["samples"]
    else:
        samples = data
    
    if has_left and has_right:
        fsr_indices = torch.tensor([12, 19, 26, 33, 40, 71, 78, 85, 92, 99])
        base_indices = torch.tensor([0, 1, 2, 3, 4, 5, 41, 42, 43, 59, 60, 61, 62, 63, 64, 100, 101, 102])
        pos_ref = torch.tensor([44, 103]) # Starting index of predicted positions for finger IMUs
    else:
        fsr_indices = torch.tensor([12, 19, 26, 33, 40])
        base_indices = torch.tensor([0, 1, 2, 3, 4, 5, 41, 42, 43])
        pos_ref = torch.tensor([44])
    
    W, R, C = samples.shape # (nWindows, nRows, nCols)
    fsr_data = samples[:, :, fsr_indices] # (W, R, C), extract the columns with FSR data
    active_finger = (fsr_data != 0).sum(dim=1).argmax(dim=1) # (W, 1), identify the active finger which is an index from 0-9
    active_fsr = fsr_indices[active_finger].unsqueeze(1) # (W, 1), identify the feature index of the active FSR

    fsr_offsets = torch.arange(6).unsqueeze(0) # (1, 6)
    cols_to_keep = active_fsr - 6 + fsr_offsets # (W, 6), keep the 6 columns of active IMU data from each window
    cols_to_keep = cols_to_keep.unsqueeze(1).expand(-1, R, -1) # (W, R, 6), reshape to match data dimensions
    output = torch.gather(samples, dim=2, index=cols_to_keep) # (W, R, 6), gather the active IMU data from each window
    
    # If right hand only, shift the right hand finger indices from 0-4 to 5-9 
    finger_idx = (active_finger if (has_left and has_right) or (has_left and not has_right) else active_finger + 5).unsqueeze(1) # (W, 1)

    # Insert a feature indicating the active finger, which is an index from 0-9
    finger_feature = finger_idx.unsqueeze(1).expand(-1, R, 1) # (W, R, 1), reshape to match data dimensions
    output = torch.cat([finger_feature, output], dim=2) # (W, R, 7)
    
    # Insert data from base IMU
    base_data = samples[:, :, base_indices]
    output = torch.cat([output, base_data], dim=2)

    # If both hands, shift the right hand finger indices from 5-9 to 0-4
    active_finger = torch.where((has_left and has_right) & (active_finger >= 5), active_finger - 5, active_finger).unsqueeze(1) # (W, 1)

    # Insert predicted positions for active finger IMU
    pos_start = (active_finger*3).unsqueeze(1) # (W, 1, 1), starting index of predicted position for the active finger
    pos_offsets = torch.arange(3).unsqueeze(0).unsqueeze(0) # (1, 1, 3), pos xyz
    pos_ref = pos_ref.unsqueeze(0).unsqueeze(2) # (1, nHands, 1)
    pos_indices = pos_start + pos_ref + pos_offsets # (W, nHands, 3)
    pos_indices = pos_indices.reshape(W, -1) # (W, nHands*3)
    pos_indices = pos_indices.unsqueeze(1).expand(-1, R, -1) # (W, R, nHands*3)
    pos_data = torch.gather(samples, dim=2, index=pos_indices) # (W, R, nHands*3)
    output = torch.cat([output, pos_data], dim=2)

    dim_aft = output.shape[2]
    dims = {
        "dim_bef": C,
        "dim_aft": dim_aft
    }

    # Normalisation
    if normalize:
        samples_tensor = [w for w in output] # List of 2D tensors
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
            mean = all_data.mean(dim=0) # (F, )      
            std = all_data.std(dim=0) # (F, )

            range_ = x_max - x_min
            range_[range_ == 0] = 1.0
        else:
            x_min = torch.zeros(F)
            range_ = torch.ones(F)

        mask = torch.ones(F, dtype=torch.bool)
        mask[0] = False
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
        output = torch.stack(norm_samples)
    else:
        mean = std = None

    if isinstance(data, dict):
        data["samples"] = output
        data["metadata"]["feat_dim"] = output.shape[2]
        data["normalize"] = normalize
        data["mean"] = mean
        data["std"] = std
        torch.save(data, output_path)
        return dims
    else:
        return output

def pca(data, dims_ratio, output_path):
    if isinstance(data, dict):
        samples = data["samples"]
    else:
        samples = data

    W, R, C = samples.shape # (nWindows, nRows, nCols)

    dim_bef = C
    dim_aft = int(dim_bef*dims_ratio)
    dims = {
        "dim_bef": dim_bef,
        "dim_aft": dim_aft
    }

    # PCA
    all_samples = torch.cat([w for w in samples], dim=0)
    U, S, V = torch.svd(all_samples)
    output = torch.matmul(all_samples, V[:, :dim_aft])
    output = output.reshape(W, R, dim_aft)

    if isinstance(data, dict):
        data["samples"] = output
        data["metadata"]["feat_dim"] = dim_aft
        data["normalize"] = False # Avoid normalising again downstream  
        torch.save(data, output_path)
        return dims
    else:
        return output