# BaseIterativeNet, AdditiveIterativeNet, EnergyAscentIterativeNet
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional, Union, Any
from torchvision.ops import box_iou
from .feedback_utils import pose_interpolation

class BaseIterativeNet(nn.Module):
    """
    BaseIterativeNet: Iterative pose estimation from RCNN RoI
        
    Args:
        net (nn.Module): iterative network
        feedback_rate (float): feedback step size
        feedback_loss_fn (Any): a callable to compute the loss function on the target and 
            predicted signals. kwargs must be "input" and "target" for predicted and target
            feedback respectively. 
        (optional) interpolate_poses (bool): if true, the intermediary poses are linearly
            interpolated between initial and target poses, default=True
        (optional) feedback_fn (Any): a callable to compute the target feedback value. Only
            used when interpolate_poses=True
    """
    def __init__(
        self, 
        net: nn.Module, 
        feedback_rate: float, 
        feedback_loss_fn: Any, 
        interpolate_poses: Optional[bool]=True,
        feedback_fn: Optional[Any]=None
    ):
        assert not (feedback_fn==None and interpolate_poses), f'feedback_fn required when interpolate_poses=True'
        super().__init__()
        self.net = net
        self.feedback_rate = feedback_rate
        self.feedback_loss_fn = feedback_loss_fn
        self.interpolate_poses = interpolate_poses
        self.feedback_fn = feedback_fn

    def batch_from_list(self, tensors: List[Tensor]):
        """
        BaseIterativeNet.batch_from_list: makes a batch from list of tensors and return list of shapes[0]
            to invert operation with Tensor.split(idx_list)

        Args:
            tensors (List[Tensor]): list of tensors to batch
        
        Returns:
            batch (Tensor): batched tensors
            idx_list (List[int]): list of shapes[0] to reconstruct list
        """
        idx_list = [t.shape[0] for t in tensors]
        batch = torch.cat(tensors, dim=0)
        return batch, idx_list

    def preprocess_poses(self, poses: Tensor, features_dim: int):
        """
        BaseIterativeNet.preprocess_poses: transform poses tensor into a tensor of poses feature
            feature maps to concatenate with CNN features

        Args:
            poses (Tensor[batch_size, K, 3]): tensor of poses to preprocess
            features_dim (int): dimension of feature maps

        Returns:
            features_poses (Tensor[batch_size, 3K, features_dim, features_dim]): preprocessed poses
        """
        batch_size, K, _ = poses.shape
        features_poses = []
        visibilities = []
        for keypoint in poses:
            x, y, v = keypoint[:,0], keypoint[:,1], keypoint[:,2]
            features = []
            for x_val, y_val, v_val in zip(x, y, v):
                features.append(torch.full((1,1,features_dim,features_dim), x_val))
                features.append(torch.full((1,1,features_dim,features_dim), y_val))
                features.append(torch.full((1,1,features_dim,features_dim), v_val))
            features = torch.cat(features, dim=1)
            features_poses.append(features)
        features_poses = torch.cat(features_poses, dim=0)
        return features_poses

    def forward_iteration(self, features_batch, poses_batch):
        raise NotImplementedError(f'No implementation of `forward_iteration` found.')

    def compute_loss(
        self, 
        feedback: List[Tensor], 
        target_feedback: List[Tensor],
        poses: List[Tensor],
        ious: List[Tensor]
    ):
        """
        BaseIterativeNet.compute_loss: compute iou-weighted feedback loss

        Args:
            feedback (List[Tensor[L]]): list of predicted feedbacks (requires grad !)
            target_feedback (List[Tensor[L']]): list of target feedbacks
            poses (List[Tensor[L, K, 3]]): list of predicted poses (requires grad !)
            target_poses (List[Tensor[L', K, 3]]): list of target poses
            ious (List[Tensor[L, L']]): pairwise ious between predictions and targets

        Returns:
            loss (Tensor[1]): average feedback loss
        """
        for i, (pred_feedback, pred_pose) in enumerate(feedback, poses):
            pred_visibility = pred_pose[:,:,2]
            for j, (true_feedback, true_pose) in enumerate(target_feedback, target_poses):
                true_visibility = true_pose[:,:,2]
                loss = ious[i,j] * (self.feedback_loss_fn(pred_feedback, true_feedback) + F.binary_cross_entropy(pred_visibility, true_visibility))
                losses.append(loss)
        loss = losses.mean()
        return loss

    def forward(
        self, 
        detections: List[Dict[str, Tensor]], 
        num_iterations: int, 
        targets: Optional[List[Dict[str, Tensor]]]=None
    ):
        """
        BaseIterativeNet.forward: forward pass

        Args:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[L, 4]): coordinates of L bbox, formatted as [x0, y0, x1, y1]
                    - features (Tensor[L, 256, D, D]): RoI feature maps
                    - keypoints (Tensor[L, K, 3]): For each one of the L 
                        objects, it contains the K keypoints in [x, y, visibility] format, defining the object.
                    - others
            num_iterations (int): number of feedback iterations
            (optional) targets (List[Dict[str, Tensor]]): List of dictionaries containing target 
                attributes:
                    - boxes (Tensor[L', 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - keypoints (List[Tensor[L', K, 3]]): For each one of the L' objects, it 
                        contains the K keypoints in [x, y, visibility] format, defining the object.
                    - area (List[Tensor[1]]): area of bounding boxes
                    - others
        
        Returns:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - keypoints (List[List[Tensor[L, K, 3]]]): List of `num_iterations+1` lists of L 
                        keypoints estimates
                    - feedbacks (List[List[Tensor[L]]]): list of `num_iterations` lists of L feedback 
                        estimates
                    - others
            (optional) losses (List[Tensor[1]]): List of `num_iterations` averaged over batch losses
        """
        # parse and initialize
        features = [d['features'] for d in detections] # List[Tensor[L,256,D,D]]
        poses = [[d['keypoints'] for d in detections]] # List[List[Tensor[L,K,3]]]
        feedbacks = [] # List[List[Tensor[1]]]
        if not self.training:
            target_poses = [t['keypoints'] for t in targets] # List[Tensor[L',K,3]]
            target_areas = [t['area'] for t in targets] # List[Tensor[1]]
            losses = [] # List[Tensor[1]]
            boxes = [d['boxes'] for d in detections] # List[Tensor[L, 4]]
            target_boxes = [t['boxes'] for t in detections] # List[Tensor[L', 4]]
            ious = []
            for pred_boxes in boxes:
                for targ_boxes in target_boxes:
                    ious.append(box_iou(pred_boxes, target_boxes)) # List[Tensor[L, L']]
        # make batches
        features_batch, idx_list = self.batch_from_list(features) # Tensor[L*M, 256, D, D]
        poses_batch, _ = self.batch_from_list(poses[0]) # Tensor[L*M, K, 3]
        if not self.training:
            target_poses_batch, target_idx_list = self.batch_from_list(target_poses) # Tensor[L'*M, K, 3]
        # create intermediary poses if necessary
        if not self.training and self.interpolate_poses:
            intermediary_poses = pose_interpolation(poses_batch, target_poses_batch, num_iterations)
        # iterative refinement
        for iteration in range(num_iterations):
            poses_batch, feedback_batch = self.forward_iteration(features_batch, poses_batch)
            feedback_batch.squeeze(-1) # Tensor[L*M, 1] -> Tensor[L*M]
            poses.append(poses_batch.split(idx_list, 0)) # Tensor[L*M, 256, D, D] -> List[Tensor[L, 256, D, D]]
            feedbacks.append(feedback_batch.split(idx_list, 0)) # Tensor[L*M, K, 3] -> List[Tensor[L, K, 3]]
            if self.training:
                # compute target feedback
                if self.interpolate_poses:
                    target_feedback_batch = self.feeback_fn(intermediary_poses[iteration], target_poses_batch, target_areas).detach()
                    intermediary_poses = intermediary_poses[iteration].split(target_idx_list, 0)
                else:
                    target_feedback_batch = self.default_feedback*torch.ones_like(feedback_batch).detach()
                    intermediary_poses = target_poses_batch.split(target_idx_list, 0)
                target_feedback = target_feedback_batch.split(target_idx_list, 0)
                loss = self.compute_loss(feedbacks[-1], target_feedback, poses[-1], intermediary_poses, ious)
                losses.append(loss)
        # complete detections
        for detection in detections:
            detection['keypoints'] = poses
            detection['feedbacks'] = feedbacks
        # return
        if self.training:
            return detections, losses
        else:
            return detections

class EnergyAscentIterativeNet(BaseIterativeNet):
    """
    EnergyAscentIterativeNet: Iterative pose estimation from RCNN RoI using energy ascent
        
    Args:
        net (nn.Module): iterative network
        feedback_rate (float): feedback step size
        feedback_loss_fn (Any): a callable to compute the loss function on the target and 
            predicted signals. kwargs must be "input" and "target" for predicted and target
            feedback respectively
        (optional) interpolate_poses (bool): if true, the intermediary poses are linearly
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
        interpolate_poses: Optional[bool]=True,
        feedback_fn: Optional[Any]=None,
        default_feedback: Optional[float]=1.
    ):
        super().__init__(net, feedback_rate, feedback_loss_fn, interpolate_poses, feedback_fn)
        self.default_feedback = default_feedback
    
    def forward_iteration(self, features_batch: Tensor, poses_batch: Tensor):
        """
        EnergyAscentIterativeNet.forward_iteration: single iteration of energy ascent forward pass

        Args:
            features_batch (Tensor[batch_size, 256, D, D]): Tensor of features for each detection
            poses_batch (Tensor[batch_size, K, 3]): Tensor of current poses estimates

        Returns:
            next_poses (Tensor[batch_size, K, 3]): Tensor of next poses estimates
            feedback (Tensor[batch_size]): Tensor of feedback estimates
        """
        poses_features = self.preprocess_poses(poses_batch, features_batch.shape[-1])
        poses_features.requires_grad_(True)
        inp = torch.cat([features_batch, poses_features])
        feedback = self.net(inp)
        grad = torch.autograd.grad(feedback, poses_features)
        poses_features += self.feedback_rate*grad
        pooled_poses = F.avg_pool2d(poses_features, kernel_size=poses_features.shape[-1])
        x = F.relu(torch.cat([pooled_poses[:,i] for i in range(0, pooled_poses[1], 3)], dim=1))
        y = F.relu(torch.cat([pooled_poses[:,i] for i in range(1, pooled_poses[1]+1, 3)], dim=1))
        v = F.sigmoid(torch.cat([pooled_poses[:,i] for i in range(2, pooled_poses[1]+1, 3)], dim=1))
        next_poses = torch.cat([x, y, v], dim=-1)
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
        (optional) interpolate_poses (bool): if true, the intermediary poses are linearly
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
        interpolate_poses: Optional[bool]=True,
        feedback_fn: Optional[Any]=None,
        default_feedback: Optional[float]=0.
    ):
        super().__init__(net, feedback_rate, feedback_loss_fn, interpolate_poses, feedback_fn)
        self.default_feedback = default_feedback

    def forward_iteration(self, features_batch: Tensor, poses_batch: Tensor):
        """
        AdditiveIterativeNet.forward_iteration: single iteration of energy ascent forward pass

        Args:
            features_batch (Tensor[batch_size, 256, D, D]): Tensor of features for each detection
            poses_batch (Tensor[batch_size, K, 3]): Tensor of current poses estimates

        Returns:
            next_poses (Tensor[batch_size, K, 3]): Tensor of next poses estimates
            feedback (Tensor[batch_size]): Tensor of feedback estimates
        """
        poses_features = self.preprocess_poses(poses_batch, features_batch.shape[-1])
        inp = torch.cat([features_batch, poses_features])
        feedback = self.net(inp)
        poses_features += self.feedback_rate*feedback
        pooled_poses = F.avg_pool2d(poses_features, kernel_size=poses_features.shape[-1])
        x = F.relu(torch.cat([pooled_poses[:,i] for i in range(0, pooled_poses[1], 3)], dim=1))
        y = F.relu(torch.cat([pooled_poses[:,i] for i in range(1, pooled_poses[1]+1, 3)], dim=1))
        v = F.sigmoid(torch.cat([pooled_poses[:,i] for i in range(2, pooled_poses[1]+1, 3)], dim=1))
        next_poses = torch.cat([x, y, v], dim=-1)
        return next_poses, feedback