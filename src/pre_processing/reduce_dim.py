import torch
from typing import Dict, Optional

def reduce_dim(
        data_dir: str,
        method: str,
        has_left: bool,
        has_right: bool,
        normalise: bool,
        output_path: str,
        dims_ratio: Optional[float] = None
) -> Dict[str, int]:
    """
    Applies dimensionality reduction to the preprocessed dataset

    Args:
        data_dir: Directory containing preprocessed dataset
        method: Method of dimensionality reduction (helper function must be defined below)
        has_left: Whether data from left hand is present
        has_right: Whether data from right hand is present
        normalise: Whether to normalise features
        output_path: Path of output file generated from dimensionality reduction
        dims_ratio: Proportion of dimensions to keep for PCA

    Returns:
        dims dictionary with feature dimension before and after dimensionality reduction
    """

    if method == "active-imu":
        dim_bef, dim_aft = active_imu_only(data_dir, has_left, has_right, normalise, output_path)
    elif method == "pca":
        dim_bef, dim_aft = pca(data_dir, dims_ratio, output_path)
    else:
        raise Exception("Invalid dimensionality reduction method.")

    dims = {
        "dim_bef": dim_bef,
        "dim_aft": dim_aft
    }

    return dims

# Reduces feature dimension by keeping only data from the active and base IMUs
def active_imu_only(data_dir, has_left, has_right, normalise, output_path):
    data = torch.load(data_dir)
    samples = data["samples"]
    
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

    # Normalisation
    mean, std = None, None
    if normalise:
        all_data = torch.cat([w for w in output], dim=0)
        mean = all_data.men(dim=0)
        std = all_data.std(dim=0)
        std[std == 0] = 1.0
    output = (output - mean) / std

    # Update .pt file
    data["samples"] = output
    data["metadata"]["feat_dim"] = output.shape[2]
    data["normalize"] = normalise
    data["mean"] = mean
    data["std"] = std
    torch.save(data, output_path)

    # Return feature dimension before and after dimensionality reduction
    return C, output.shape[2]

def pca(data_dir, dims_ratio, output_path):
    data = torch.load(data_dir)
    samples = data["samples"]

    W, R, C = samples.shape # (nWindows, nRows, nCols)

    dims_bef = C
    dims_aft = int(dims_bef*dims_ratio)

    # PCA
    all_samples = torch.cat([w for w in samples], dim=0)
    U, S, V = torch.svd(all_samples)
    output = torch.matmul(all_samples, V[:, :dims_aft])
    output = output.reshape(W, R, dims_aft)

    # Update .pt file
    data["samples"] = output
    data["metadata"]["feat_dim"] = dims_aft
    data["normalize"] = False # Avoid normalising again downstream  
    torch.save(data, output_path)

    # Return feature dimension before and after dimensionality reduction
    return dims_bef, dims_aft