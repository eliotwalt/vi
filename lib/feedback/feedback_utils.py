import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional, Union, Any

def pose_interpolation(init_poses: Tensor, target_poses: Tensor, num_iterations: int):
    """
    pose_interpolation: perform num_iteration linear interpolations between init_poses and target_poses

    Args:
        init_poses (Tensor[batch_size, K, 3]): initial poses tensor
        target_poses (Tensor[batch_size, K, 3]): target_poses tensor

    Returns:
        intermediary_poses (List[Tensor[batch_size, K, 3]]): list of num_iterations linear interpolations

    Note:
        The visibility bit is always given by target_poses as we can't predict when it is supposed to appear
            along the interpolation trajectory.
    """
    visibilities = target_poses[:,:,-1].unsqueeze(-1)
    intermediary_poses = []
    for it in range(1, num_iterations+1):
        w = it/(num_iterations+1)
        print(init_poses.shape, init_poses[:,:,0:2].shape)
        print(target_poses.shape, target_poses[:,:,0:2].shape)
        tmp = torch.lerp(input=init_poses[:,:,0:2], end=target_poses[:,:,0:2], weight=w)
        intermediary_poses.append(
            torch.cat([tmp, visibilities], dim=-1)
        )
    return intermediary_poses
    