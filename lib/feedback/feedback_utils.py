import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_iou
from typing import Tuple, List, Dict, Optional, Union, Any

def pose_interpolation(init_poses: Tensor, target_poses: Tensor, num_iterations: int):
    """
    pose_interpolation: perform num_iteration linear interpolations between init_poses and target_poses

    Args:
        init_poses (List[Tensor[L, K, 3]]): list of M initial poses tensor
        target_poses (List[Tensor[L', K, 3]]): list of M target_poses tensor

    Returns:
        intermediary_poses (List[List[Tensor[L, L', K, 3]]]): list of num_iterations lists of M tensors
        of linear interpolations. i.e intermediary_poses[n][i][j][k] is the interpolation between  init
        pose j and target pose k at iteration n for image i.
    """
    intermediary_poses = [[] for numit in range(num_iterations)]
    for numit in range(num_iterations):
        intermediary_poses[numit] = [None for nimg in range(len(init_poses))]
        w = numit/(num_iterations+1)
        for nimg in range(len(init_poses)):
            L, K, _ = init_poses[nimg].size()
            Lp, _, _ = target_poses[nimg].size()
            init_p = init_poses[nimg]
            targ_p = target_poses[nimg]
            intermediary_poses[numit][nimg] = torch.empty(L,Lp,K,3)
            for pred_idx in range(L):
                for targ_idx in range(Lp):
                    tmp = torch.lerp(input=init_p[pred_idx][:,0:2], end=targ_p[targ_idx][:,0:2], weight=w)
                    tmp = torch.cat([tmp, targ_p[targ_idx][:,2].unsqueeze(-1)], dim=-1)
                    intermediary_poses[numit][nimg][pred_idx][targ_idx] = tmp
    return intermediary_poses

def compute_ious(pred_boxes, target_boxes):
    """
    compute_ious: ious between list of detections

    Args:
        pred_boxes (List[Tensor[L, 4]]): list of predicted boxes for each image
        target_boxes (List[Tensor[L, 4]]): list of target boxes for each image
    
    Returns:
        ious (List[Tensor[L,L']]): list of image-wise pairwise ious
    """
    ious = []
    for nimg in range(len(pred_boxes)): 
        p_boxes = pred_boxes[nimg]
        t_boxes = target_boxes[nimg]
        ious.append(box_iou(p_boxes, t_boxes))
    return ious