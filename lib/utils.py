# kp_loss, oks
# oks, look at https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional

class Oks(nn.Module):
    """
    Oks: torch module to compute OKS
    
    Args:
        weights (torch[K]): iterable of weights to weight each keypoint
        normalize (bool): If true, the l2 distances are normalized by the area, 
            otherwise not.
    """
    def __init__(self, weights: Tensor, normalize: bool):
        super().__init__()
        self.weights = weights
        self.normalize = normalize
    
    def forward(
        self, 
        input_poses: Tensor, 
        target_poses: Tensor, 
        areas: Optional[Tensor]=None
    ):
        """
        Oks.forward: forward pass
        
        Args:
            input_poses (Tensor[N, K, 3]): input poses
            target_poses (Tensor[N, K, 3]): target poses
            (optional) areas (Tensor[N]): areas to normalize with, required if
                self.normalize=True
        
        Returns:
            oks (Tensor[N]): oks values for each pose
        """
        dists = (input_poses[:,:,0:2] - target_poses[:,:,0:2])**2        
        dists = dists.sum(dim=-1)        
        if self.normalize:
            assert areas is not None, f'`areas` is required when `normalize=True`'
            if len(areas.shape) < 2: areas = areas.unsqueeze(-1)
            dists /= areas
        if len(self.weights.shape) < 2: self.weights = self.weights.unsqueeze(0)
        dists /= 2*self.weights
        oks = torch.exp(-dists).mean(1)
        return oks
