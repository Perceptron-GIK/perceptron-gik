import torch
from typing import Dict

def reduce_dim(
        data_dir: str,
        method: str,
        has_left: bool,
        has_right: bool,
        normalise: bool,
        output_path: str
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

    Returns:
        dims dictionary with feature dimension before and after dimensionality reduction
    """

    if method == "active-imu":
        dim_bef, dim_aft = active_imu_only(data_dir, has_left, has_right, normalise, output_path)
    else: # Add other dimensionality reduction methods here
        pass

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
    else:
        fsr_indices = torch.tensor([12, 19, 26, 33, 40])
        base_indices = torch.tensor([0, 1, 2, 3, 4, 5, 41, 42, 43])
    
    W, R, C = samples.shape # (nWindows, nRows, nCols), for reference only
    fsr_data = samples[:, :, fsr_indices] # (W, R, C), extract the columns with FSR data
    active_finger = (fsr_data != 0).sum(dim=1).argmax(dim=1) # (W, 1), identify the active finger which is an index from 0-9
    active_fsr = fsr_indices[active_finger].unsqueeze(1) # (W, 1), identify the feature index of the active FSR

    offsets = torch.arange(6).unsqueeze(0) # (1, 6)
    cols_to_keep = active_fsr - 6 + offsets # (W, 6), keep the 6 columns of active IMU data from each window
    cols_to_keep = cols_to_keep.unsqueeze(1).expand(-1, R, -1) # (W, R, 6), reshape to match data dimensions
    output = torch.gather(samples, dim=2, index=cols_to_keep) # (W, R, 6), gather the active IMU data from each window
    
    # Shift the right hand finger indices from 0-4 to 5-9 
    finger_idx = (active_finger if (has_left and has_right) or (has_left and not has_right) else active_finger + 5).unsqueeze(1) # (W, 1)

    # Insert a feature indicating the active finger, which is an index from 0-9
    finger_feature = finger_idx.unsqueeze(1).expand(-1, R, 1) # (W, R, 1), reshape to match data dimensions
    output = torch.cat([finger_feature, output], dim=2) # (W, R, 7)
    
    # Insert data from base IMU
    base_data = samples[:, :, base_indices]
    output = torch.cat([output, base_data], dim=2)

    # Compute normalisation stats
    mean, std = None, None
    if normalise:
        all_data = torch.cat([w for w in output], dim=0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        std[std == 0] = 1.0

    # Update .pt file
    data["samples"] = output
    data["metadata"]["feat_dim"] = output.shape[2]
    data["metadata"]["input_dim"] = output.shape[2] + 40
    data["normalise"] = normalise
    data["mean"] = mean
    data["std"] = std
    torch.save(data, output_path)

    # Return feature dimension before and after dimensionality reduction
    return C, output.shape[2]