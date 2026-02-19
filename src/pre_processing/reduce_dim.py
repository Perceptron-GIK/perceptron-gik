import torch
from typing import Dict

def reduce_dim(
        data_dir: str,
        method: str,
        has_left: bool,
        has_right: bool
) -> Dict[str, int]:
    """
    Applies dimensionality reduction to the preprocessed dataset

    Args:
        data_dir: Directory containing preprocessed dataset
        method: Method of dimensionality reduction (helper function must be defined below)
        has_left: Whether data from left hand is present
        has_right: Whether data from right hand is present

    Returns:
        dims dictionary with feature dimension before and after dimensionality reduction
    """

    return

# Reduces feature dimension by keeping only data from the active and base IMUs
def active_imu_only(data_dir, has_left, has_right):
    data = torch.load(data_dir)
    samples = data["samples"]
    
    if has_left and has_right:
        fsr_indices = torch.tensor([12, 19, 26, 33, 40, 71, 78, 85, 92, 99])
    else:
        fsr_indices = torch.tensor([12, 19, 26, 33, 40])
    
    fsr_data = samples[:, :, fsr_indices]
    for window in fsr_data:
        fsr = torch.argmax((window != 0).sum(dim=0)) # Identify the FSR that was pressed