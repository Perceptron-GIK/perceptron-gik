import torch
from typing import Dict, Optional


def _index_map(col_names):
    return {name: i for i, name in enumerate(col_names)}


def _get_part_from_fsr_col(col_name: str) -> str:
    # f_thumbL -> thumbL
    return col_name.split("f_", 1)[1]


def _split_finger_and_side(part: str):
    if part.endswith(("L", "R")):
        return part[:-1], part[-1]
    return part, None

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
    meta = data.get("metadata", {})
    col_names = meta.get("combined_col_names", [])
    if not col_names:
        raise ValueError("metadata['combined_col_names'] is required for active-imu reduction")
    # Prefer saved metadata over caller flags to avoid mismatch errors.
    has_left = bool(meta.get("has_left", has_left))
    has_right = bool(meta.get("has_right", has_right))

    idx = _index_map(col_names)

    fsr_cols = [c for c in col_names if c.startswith("f_")]
    fsr_indices = torch.tensor([idx[c] for c in fsr_cols], dtype=torch.long)
    if fsr_indices.numel() == 0:
        raise ValueError("No FSR columns found in dataset metadata")

    base_parts = []
    if has_left:
        base_parts.append("baseL")
    if has_right:
        base_parts.append("baseR")

    base_indices = []
    for part in base_parts:
        for axis in ("ax", "ay", "az", "gx", "gy", "gz", "x", "y", "z"):
            name = f"{axis}_{part}"
            if name in idx:
                base_indices.append(idx[name])
    base_indices = torch.tensor(base_indices, dtype=torch.long)

    # For each FSR/finger part, gather ax..gz columns (active IMU motion streams).
    imu6_table = []
    finger_parts = []
    for fsr_col in fsr_cols:
        part = _get_part_from_fsr_col(fsr_col)
        finger_parts.append(part)
        imu6 = [idx[f"{axis}_{part}"] for axis in ("ax", "ay", "az", "gx", "gy", "gz")]
        imu6_table.append(imu6)
    imu6_table = torch.tensor(imu6_table, dtype=torch.long)

    # Position indices for active finger(s).
    pos3_table = torch.tensor(
        [[idx[f"x_{part}"], idx[f"y_{part}"], idx[f"z_{part}"]] for part in finger_parts],
        dtype=torch.long,
    )
    part_to_fsr_idx = {part: i for i, part in enumerate(finger_parts)}
    opposite_hand_idx = []
    for part in finger_parts:
        finger, side = _split_finger_and_side(part)
        if side == "L":
            opposite_part = f"{finger}R"
        elif side == "R":
            opposite_part = f"{finger}L"
        else:
            opposite_part = part
        opposite_hand_idx.append(part_to_fsr_idx.get(opposite_part, part_to_fsr_idx[part]))
    opposite_hand_idx = torch.tensor(opposite_hand_idx, dtype=torch.long)
    
    W, R, C = samples.shape # (nWindows, nRows, nCols)
    fsr_data = samples[:, :, fsr_indices] # (W, R, C), extract the columns with FSR data
    active_finger = (fsr_data != 0).sum(dim=1).argmax(dim=1) # (W, 1), identify the active finger which is an index from 0-9
    cols_to_keep = imu6_table[active_finger]  # (W, 6), active finger ax..gz
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

    # Insert predicted positions for active finger IMU
    if has_left and has_right:
        # Keep behavior of returning both-hand positions for corresponding finger family.
        active_pos = pos3_table[active_finger]
        paired_pos = pos3_table[opposite_hand_idx[active_finger]]
        pos_indices = torch.cat([active_pos, paired_pos], dim=1)  # (W, 6)
    else:
        pos_indices = pos3_table[active_finger]  # (W, 3)

    pos_indices = pos_indices.unsqueeze(1).expand(-1, R, -1) # (W, R, nHands*3)
    pos_data = torch.gather(samples, dim=2, index=pos_indices) # (W, R, nHands*3)
    output = torch.cat([output, pos_data], dim=2)

    # Normalisation
    mean, std = None, None
    if normalise:
        all_data = torch.cat([w for w in output], dim=0)
        mean = all_data.mean(dim=0)
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