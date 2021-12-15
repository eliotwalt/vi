# kp_loss, oks
# oks, look at https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional
from torchvision.models.resnet import BasicBlock

class Oks(nn.Module):
    """
    Oks: torch module to compute OKS
    
    Args:
        weights (torch[K]): iterable of weights to weight each keypoint
        normalize (bool): If true, the l2 distances are normalized by the area, 
            otherwise not.
    """
    def __init__(self, weights: Tensor):
        super().__init__()
        if len(self.weights.shape) < 2: weights = weights.unsqueeze(0)
        self.weights = weights
    
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
            areas (Tensor[N]): areas to normalize with, required if
                self.normalize=True
        
        Returns:
            oks (Tensor[N]): oks values for each pose
        """
        target_v = target_poses[:,:,2]
        dists = (input_poses[:,:,0:2] - target_poses[:,:,0:2])**2        
        dists = dists.sum(dim=-1)
        if len(areas.shape) < 2: areas = areas.unsqueeze(-1)
        dists = dists / (2*areas*self.weights**2)
        # filter visibilities
        oks = torch.exp(-dists)*target_v
        oks = oks.mean(1)
        return oks

class KeypointSignedError(nn.Module):
    """
    SignedError: torch module to compute signed error between arbitrary tensors
    """
    def __init__(self): 
        super().__init__()

    def forward(
        self,
        input_tensor: Tensor,
        targets_poses: Tensor,
        *args, **kwargs,
    ):
        """
        Oks.forward: forward pass
        
        Args:
            input_tensor (Tensor[N, K, 3]): input_tensor
            target_tensor (Tensor[N, K, 3]): target_tensor
            *args, **kwargs (Any): for compatibility
        
        Returns:
            signed_error (Tensor[N, *]): signed_error values for each position
        """
        target_v = target_poses[:,:,2]
        signed_error = target_tensor - input_tensor
        signed_error *= target_v
        return signed_error

class FeedbackResnet(nn.Module):
    """
    FeedbackResnet: Simple resnet for feedback prediction

    Args:
        (optional) in_channels (int): number of input channels (256+28=284)
        (optional) out_channels (int): number of output channels (1). When equal
            to 1, interpreted as energy ascent and therefore we produce output of
            size [N, 1].
        (optional) stride (int): stride of all convolutions
        (optional) num_blocks (int): number of torchvision.models.resnet.BasicBlock
        (optional) features_shape (int): input size of the feature maps
    """
    def __init__(
        self, 
        in_channels: Optional[int]=307,
        out_channels: Optional[int]=1,
        stride: Optional[int]=1, 
        num_blocks: Optional[int]=2,
        features_dim: Optional[int]=7
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *(BasicBlock(in_channels, in_channels, stride)
            for _ in range(num_blocks))
        )
        if out_channels == 1: # i.e energy ascent
            self.energy_ascent = True
            self.blocks.add_module(
                "output_conv", 
                nn.Conv2d(in_channels, out_channels, kernel_size=features_dim))
        else:
            self.energy_ascent = False
            self.blocks.add_module(
                "output_conv", 
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
    
    def forward(self, x: Tensor):
        x = self.blocks(x)
        if self.energy_ascent:
            x = F.sigmoid(x.reshape(-1,1))
        return x

