import torch
import os
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
        inference_source: torch.Tensor = None,
        method: str = "pca",
        has_left: bool = "False",
        has_right: bool = "False",
        normalize: bool = "False",
        output_path: Optional[str] = None,
        dims_ratio: Optional[float] = None,
        root_dir: Optional[str] = None
) -> Dict[str, int]:
    """
    Applies dimensionality reduction to the preprocessed dataset

    Args:
        data_dir: Directory containing the processed dataset from training
        inference_source: PyTorch tensor containing a single window of data for inference (if available)
        method: Method of dimensionality reduction (helper function must be defined below)
        has_left: Whether data from left hand is present
        has_right: Whether data from right hand is present
        normalize: Whether to normalise features
        output_path: Path of output file generated from dimensionality reduction (during training only)
        dims_ratio: Proportion of dimensions to keep (for PCA only)
        root_dir: Project root (for PCA only)

    If reading data from a PyTorch file:
        Returns: dims dictionary with feature dimension before and after dimensionality reduction
        Post-DR dataset is saved to output_path
        PCA params is saved to root_dir (if applicable)
    Else:
        Returns: Post-DR dataset as a PyTorch tensor
    """

    if inference_source:
        data = inference_source
    else:
        data = torch.load(data_dir)

    if method == "active-imu":
        return active_imu_only(data_dir, inference_source, has_left, has_right, normalize, output_path)
    elif method == "pca":
        return pca(data, dims_ratio, output_path, root_dir)
    else:
        raise Exception("Invalid dimensionality reduction method.")

# Reduces feature dimension by keeping only data from the active and base IMUs
def active_imu_only(data_dir, inference_source, has_left, has_right, normalize, output_path):
    data = torch.load(data_dir)
    meta = data.get("metadata", {})
    col_names = meta.get("combined_col_names", [])
    if not col_names:
        raise ValueError("metadata['combined_col_names'] is required for active-imu reduction")

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

    if inference_source:
        samples = inference_source
    else:
        samples = data["samples"]
    
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

    if not inference_source:
        data["samples"] = output
        data["metadata"]["feat_dim"] = output.shape[2]
        data["normalize"] = normalize
        data["mean"] = mean
        data["std"] = std
        torch.save(data, output_path)
        return dims
    else:
        return output

def pca(data, dims_ratio, output_path, root_dir):
    if isinstance(data, dict):
        samples = data["samples"]

        W, R, C = samples.shape # (nWindows, nRows, nCols)

        dim_bef = C
        dim_aft = int(dim_bef*dims_ratio)
        dims = {
            "dim_bef": dim_bef,
            "dim_aft": dim_aft
        }

        all_samples = torch.cat([w for w in samples], dim=0)
        mean = all_samples.mean(dim=0, keepdim=True)
        U, S, V = torch.svd(all_samples)
        components = V[:, :dim_aft]

        pca_params = {
            "mean": mean,
            "components": components
        }
        torch.save(pca_params, os.path.join(root_dir, "pca_params.pt"))

        output = torch.matmul(all_samples, components)
        output = output.reshape(W, R, dim_aft)

        data["samples"] = output
        data["metadata"]["feat_dim"] = dim_aft
        data["normalize"] = False # Avoid normalising again downstream  
        torch.save(data, output_path)
        return dims
    else:
        W, R, C = data.shape
        samples = torch.cat([w for w in data], dim=0)

        params_path = os.path.join(root_dir, "pca_params.pt")
        if not os.path.exists(params_path):
            raise FileNotFoundError("PCA parameters file not found in root directory.")
        
        pca_params = torch.load(os.path.join(root_dir, "pca_params.pt"), weights_only=False)
        mean = pca_params["mean"]
        components = pca_params["components"]
        samples_centered = samples - mean
        output = torch.matmul(samples_centered, components)

        return output.reshape(W, R, components.shape[1])   
    