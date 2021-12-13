# BaseIterativeNet, AdditiveIterativeNet, EnergyAscentIterativeNet
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional, Union, Any
from feedback_utils import pose_interpolation

class BaseIterativeNet(nn.Module):
    """
    BaseIterativeNet: Iterative pose estimation from RCNN RoI
        
    Args:
        net (nn.Module): iterative network
        feedback_rate (float): feedback step size
        feedback_loss_fn (Any): a callable to compute the loss function on the target and 
            predicted signals. kwargs must be "input" and "target" for predicted and target
            feedback respectively. 
        (optional) interpolate_pose (bool): if true, the intermediary poses are linearly
            interpolated between initial and target poses, default=True
        (optional) feedback_fn (Any): a callable to compute the target feedback value. Only
            used when interpolate_poses=True
    """
    def __init__(
        self, 
        net: nn.Module, 
        feedback_rate: float, 
        feedback_loss_fn: Any, 
        interpolate_pose: Optional[bool]=True,
        feedback_fn: Optional[Any]=None
    ):
        assert not (feedback_fn==None and interpolate_pose), f'feedback_fn required when interpolate_pose=True'
        super().__init__()
        self.net = net
        self.feedback_rate = feedback_rate
        self.feedback_loss_fn = feedback_loss_fn
        self.interpolate_pose = interpolate_pose
        self.feedback_fn = feedback_fn

    def batch_from_list(self, tensors: List[Tensor]):
        """
        BaseIterativeNet.batch_from_list: makes a batch from list of tensors and return list of shapes[0]
            to invert operation with Tensor.split(pos_list)

        Args:
            tensors (List[Tensor]): list of tensors to batch
        
        Returns:
            batch (Tensor): batched tensors
            pos_list (List[int]): list of shapes[0] to reconstruct list
        """
        pos_list = [t.shape[0] for t in tensors]
        batch = torch.cat(tensors, dim=0)
        return batch, pos_list

    def batch_pose(self, pose: Union[Tensor, List[Tensor]], batch_size: int):
        """
        BaseIterativeNet.batch_pose: makes a batch from list of tensors and return list of shapes[0]
            to invert operation with Tensor.split(pos_list)

        Args:
            pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): pose
            batch_size (int): batch_size
        
        Returns:
            batch (Tensor[batch_size, K, 3]): batch of poses
        """
        if isinstance(pose, list):
            assert len(pose) == batch_size, f'if `pose` is list, it must have length {batch_size}, not {len(pose)}'
            batch = torch.cat(pose, dim=0).reshape(batch_size, pose[0].shape[1], -1)
        elif isinstance(pose, Tensor):
            batch = torch.repeat(pose).reshape(batch_size, pose[0].shape[1], -1)
        else:
            raise TypeError(f'`pose` must be Tensor or List[Tensor], not {type(initial_pose)}.')
        return batch

    def preprocess_poses(self, poses: Tensor, features_dim: int):
        """
        BaseIterativeNet.preprocess_poses: transform poses tensor into a tensor of poses feature
            feature maps to concatenate with CNN features

        Args:
            poses (Tensor[batch_size, K, 3]): tensor of poses to preprocess
            features_dim (int): dimension of feature maps

        Returns:
            features_poses (Tensor[batch_size, 2K, features_dim, features_dim]): preprocessed poses
            visibilities (Tensor[batch_size, K, 1]): visibility component for each keypoints
        """
        batch_size, K, _ = poses.shape
        features_poses = []
        visibilities = []
        for keypoint in poses:
            x, y = keypoint[:,0], keypoint[:,1]
            visibilities.append(keypoint[:,2].unsqueeze(0))
            features = []
            for x_val, y_val in zip(x, y):
                features.append(torch.full((1,1,features_dim,features_dim), x_val))
                features.append(torch.full((1,1,features_dim,features_dim), y_val))
            features = torch.cat(features, dim=1)
            features_poses.append(features)
        features_poses = torch.cat(features_poses, dim=0)
        visibilities = torch.cat(visibilities, dim=0).unsqueeze(-1)
        return features_poses, visibilities

    def forward_iteration(self, features_batch, poses_batch):
        raise NotImplementedError, f'No implementation of `forward_iteration` found.'

    def compute_loss(
        self, feedbacks: Tensor, 
        poses_batch: Tensor, 
        target_poses: Tensor, 
        it: Optional[int]=None, 
        areas: Optional[List[Tensor]]=None
    ):
        """
        BaseIterativeNet.compute_loss: compute feedback loss

        Args:
            feedback (Tensor[batch_size]): Tensor of feedback estimates
            poses_batch (Tensor[batch_size, K, 3]): poses estimates
            target_poses (Tensor[batch_size, K, 3]): poses targets
            (optional) it (int): iteration number
            (optional) areas (List[Tensor[L]]): List of M tensors containing the area of the 
                bounding box of each detection

        Returns:
            loss (Tensor[1]): loss
        """
        if self.interpolate_pose:
            intermediary_pose = self.intermediary_poses[it]
            target_feedback = self.feeback_fn(intermediary_poses, target_poses, areas)
            target_poses = intermediary_poses
        else:
            target_feedback = self.default_feedback*torch.ones_like(feedback)
        loss = self.feedback_loss_fn(feedback, target_feedback)
        return loss


    def forward(
        self, 
        features: List[Tensor], 
        init_pose: Union[Tensor, List[Tensor]], 
        num_iterations: int, 
        targets: Optional[List[Tensor]]=None,
        areas: Optional[List[Tensor]]=None,
    ):
        """
        BaseIterativeNet.forward: forward pass

        Args:
            features (List[Tensor[L, 256, D, D]]): list of M tensors corresponding to the feature 
                maps associated to M images
            init_pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): initial pose. If Tensor, it 
                represent a generic pose to be with all features. If List[Tensor], it contains a
                specific pose for each element of each features
            num_iterations (int): number of feedback iterations
            (optional) targets (List[Tensor[L, K, 3]]): List of M tensors containing the target
                poses for each objects.
            (optional) areas (List[Tensor[L]]): List of M tensors containing the area of the 
                bounding box of each detection
            
        Returns:
            poses (List[Tensor[L, K, 3]]): List of `num_iterations` tensors containing pose
                estimates at each iteration.
            feedbacks (List[Tensor[feedback_dim]]): List of `num_iterations` tensors containing
                feedback signal estimates at each iteration.
            (optional) losses (List[Tensor[*]]): List of `num_iterations` tensors containing
		        losses
        """
        # init
        poses = []
        feedbacks = []
        if not self.training:
            target_poses, _ = self.batch_from_list(targets)
            losses = []
        if areas is not None:
            areas = torch.cat(areas) # shape must be [batch_size]
        # batch features into Tensor[L*M, 256, D, D]
        features_batch, pos_list = self.batch_from_list(features)
        batch_size = features_batch.shape[0] # = L*M
        # turn init_pose into Tensor[L*M, K, 3]
        poses_batch = self.batch_pose(init_pose, batch_size)
        # create intermediary poses if interpolation required
        if self.interpolate_pose:
            self.intermediary_poses = pose_interpolation(poses_batch, target_poses, num_iterations)
        # iterative refinement
        for it in range(num_iterations):
            poses_batch, feedback = self.forward_iteration(features_batch, poses_batch)
            poses.append(poses_batch)
            feedbacks.append(feedback)
            if self.training:
                loss = self.compute_loss(feedback, poses_batch, target_poses, it, areas)
                losses.append(loss)
        # reindex
        poses = list(poses.split(pos_list, 0))
        feedbacks = list(feedbacks.split(pos_list, 0))
        # return
        if self.training:
            return poses, feedbacks, losses
        else:
            return poses, feedbacks

class EnergyAscentIterativeNet(BaseIterativeNet):
    """
    EnergyAscentIterativeNet: Iterative pose estimation from RCNN RoI using energy ascent
        
    Args:
        net (nn.Module): iterative network
        feedback_rate (float): feedback step size
        feedback_loss_fn (Any): a callable to compute the loss function on the target and 
            predicted signals. kwargs must be "input" and "target" for predicted and target
            feedback respectively
        (optional) interpolate_pose (bool): if true, the intermediary poses are linearly
            interpolated between initial and target poses, default=True
        (optional) feedback_fn (Any): a callable to compute the target feedback value. Only
            used when interpolate_poses=True
        (optional) default_feedback (int): value of default feedback to use when target pose
            is not interpolated, default=1.
    """
    def __init__(
        self, 
        net: nn.Module, 
        feedback_rate: float, 
        feedback_loss_fn: Any, 
        interpolate_pose: Optional[bool]=True,
        feedback_fn: Optional[Any]=None,
        default_feedback: Optional[float]=1.
    ):
        super().__init__(net, feedback_rate, feedback_loss_fn, interpolate_pose, feedback_fn)
        self.default_feedback = default_feedback
    
    def forward_ieration(self, features_batch, poses_batch):
        """
        EnergyAscentIterativeNet.forward_iteration: single iteration of energy ascent forward pass

        Args:
            features_batch (Tensor[batch_size, 256, D, D]): Tensor of features for each detection
            poses_batch (Tensor[batch_size, K, 3]): Tensor of current poses estimates

        Returns:
            next_poses (Tensor[batch_size, K, 3]): Tensor of next poses estimates
            feedback (Tensor[batch_size]): Tensor of feedback estimates
        """
        poses_features, visibilities = self.preprocess_poses(poses_batch, features_batch.shape[-1])
        poses_features.requires_grad_(True)
        inp = torch.cat([features_batch, poses_features])
        feedback = self.net(inp)
        grad = torch.autograd.grad(feedback, poses_features)
        poses_features += self.feedback_rate*grad
        pooled_poses = F.avg_pool2d(poses_features, kernel_size=poses_features.shape[-1])
        x = torch.cat([pooled_poses[:,i] for i in range(0, pooled_poses[1], 2)], dim=1)
        y = torch.cat([pooled_poses[:,i] for i in range(1, pooled_poses[1]+1, 2)], dim=1)
        next_poses = torch.cat([x, y, visibilities], dim=-1)
        return next_poses, feedback

class AdditiveIterativeNet(BaseIterativeNet):
    """
    AdditiveIterativeNet: Iterative pose estimation from RCNN RoI using energy ascent
        
    Args:
        net (nn.Module): iterative network
        feedback_rate (float): feedback step size
        feedback_loss_fn (Any): a callable to compute the loss function on the target and 
            predicted signals. kwargs must be "input" and "target" for predicted and target
            feedback respectively
        (optional) interpolate_pose (bool): if true, the intermediary poses are linearly
            interpolated between initial and target poses, default=True
        (optional) feedback_fn (Any): a callable to compute the target feedback value. Only
            used when interpolate_poses=True
        (optional) default_feedback (int): value of default feedback to use when target pose
            is not interpolated, default=0.
    """
    def __init__(
        self, 
        net: nn.Module, 
        feedback_rate: float, 
        feedback_loss_fn: Any, 
        interpolate_pose: Optional[bool]=True,
        feedback_fn: Optional[Any]=None,
        default_feedback: Optional[float]=0.
    ):
        super().__init__(net, feedback_rate, feedback_loss_fn, interpolate_pose, feedback_fn)
        self.default_feedback = default_feedback

    def forward_ieration(self, features_batch, poses_batch):
        """
        AdditiveIterativeNet.forward_iteration: single iteration of energy ascent forward pass

        Args:
            features_batch (Tensor[batch_size, 256, D, D]): Tensor of features for each detection
            poses_batch (Tensor[batch_size, K, 3]): Tensor of current poses estimates

        Returns:
            next_poses (Tensor[batch_size, K, 3]): Tensor of next poses estimates
            feedback (Tensor[batch_size]): Tensor of feedback estimates
        """
        poses_features, visibilities = self.preprocess_poses(poses_batch, features_batch.shape[-1])
        inp = torch.cat([features_batch, poses_features])
        feedback = self.net(inp)
        poses_features += self.feedback_rate*feedback
        pooled_poses = F.avg_pool2d(poses_features, kernel_size=poses_features.shape[-1])
        x = torch.cat([pooled_poses[:,i] for i in range(0, pooled_poses[1], 2)], dim=1)
        y = torch.cat([pooled_poses[:,i] for i in range(1, pooled_poses[1]+1, 2)], dim=1)
        next_poses = torch.cat([x, y, visibilities], dim=-1)
        return next_poses, feedback