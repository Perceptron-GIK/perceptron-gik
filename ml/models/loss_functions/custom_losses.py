"""Loss functions for GIK Training
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance. Best performance with gamma=2.0."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
    
class CoordinateLoss(nn.Module):
    """Focal Loss for handling class imbalance. Best performance with gamma=2.0."""
    
    def __init__(self, h_v_ratio):
        """Params for the Coordinate Loss that takes two continuous logits as inputs and measurers their coordinate differences to the target

        Args:
            h_v_ratio (float): ratio of the weights given to horizontal correctness vs vertical correctness. 
            i.e, 1.5 means horizontal correctness matters 1.5 times more than vertical correctness
        """
        super().__init__()
        self.ratio = h_v_ratio
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.relu(inputs)
        loss = F.mse_loss(x, targets, reduction="none") # reduction prevents aggregation of the last axis
        weights = loss.new_tensor([self.ratio, 1]) # first logit is horizontal
        weighted_loss = weights * loss
        
        return weighted_loss.mean()
    
    
class CoordinateLossClassification(nn.Module):
    """Focal Loss for handling class imbalance. Best performance with gamma=2.0."""
    
    def __init__(self, h_v_ratio):
        """Params for the Coordinate Loss that takes two continuous logits as inputs and measurers their coordinate differences to the target

        Args:
            h_v_ratio (float): ratio of the weights given to horizontal correctness vs vertical correctness. 
            i.e, 1.5 means horizontal correctness matters 1.5 times more than vertical correctness
        """
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.ratio = h_v_ratio
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = 9 *  self.sigmoid(inputs)
        x = torch.floor(x)
        loss = F.mse_loss(x, targets, reduction="none") # reduction prevents aggregation of the last axis
        weights = loss.new_tensor([self.ratio, 1]) # first logit is horizontal
        weighted_loss = weights * loss
        
        return weighted_loss.mean()