# BaseIterativeNet, AdditiveIterativeNet, EnergyAscentIterativeNet
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional, Union, Any
from .feedback_utils import pose_interpolation, compute_ious

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
        # print('theerror input poses', poses.__class__, len(poses))
        for keypoint in poses:
            x, y, v = keypoint[:,0], keypoint[:,1], keypoint[:,2]
            features = []
            for x_val, y_val, v_val in zip(x, y, v):
                features.append(torch.full((1,1,features_dim,features_dim), x_val.item()))
                features.append(torch.full((1,1,features_dim,features_dim), y_val.item()))
                features.append(torch.full((1,1,features_dim,features_dim), v_val.item()))
            features = torch.cat(features, dim=1)
            features_poses.append(features)
        # print('theerror before cat', features_poses.__class__, len(features_poses))
        features_poses = torch.cat(features_poses, dim=0)
        return features_poses

    def forward_iteration(self, features_batch, poses_batch):
        raise NotImplementedError(f'No implementation of `forward_iteration` found.')

    def compute_loss(
        self, 
        pred_poses: List[Tensor],
        target_poses: List[Tensor],
        pred_feedbacks: List[Tensor],
        ious: List[Tensor],
        target_areas: Optional[List[Tensor]],
        intermediary_poses: Optional[List[List[Tensor]]]=None 
    ):
        """
        BaseIterativeNet.compute_loss: compute iou-weighted feedback loss

        Args:
            pred_poses (List[Tensor[L, K, 3]]): list of predicted poses (requires grad !)
            target_poses (List[Tensor[L', K, 3]]): list of target poses
            pred_feedbacks (List[Tensor[L]]): list of predicted feedbacks (requires grad !)
            ious (List[Tensor[L, L']]): pairwise ious between predictions and targets
            target_areas (List[Tensor[L']]): list of target boxes areas
            (optional) intermediary_poses (List[Tensor[L,L',K,3]]): interpolated poses if necessary

        Returns:
            loss (Tensor[]): average feedback loss
        """
        loss = []
        for nimg in range(len(pred_poses)): # loop on input images
            device = pred_poses[nimg].device
            if intermediary_poses is not None:
                intermediary_poses[nimg] = intermediary_poses[nimg].to(device)
            target_poses[nimg] = target_poses[nimg].to(device)
            target_areas[nimg] = target_areas[nimg].to(device)
            ious[nimg] = ious[nimg].to(device)
            for i in range(pred_poses[nimg].shape[0]): # loop on predictions
                for j in range(target_poses[nimg].shape[0]): # loop on targets
                    pred_pose = pred_poses[nimg][i] # (17, 3)
                    targ_pose = target_poses[nimg][j] # (17, 3)
                    pred_feedback = pred_feedbacks[nimg][i] # (feedback_shape)
                    if intermediary_poses is not None:
                        interm_pose = intermediary_poses[nimg][i][j]
                        targ_feedback = self.feedback_fn(pred_pose, interm_pose, target_areas[nimg][j])
                    else:
                        targ_feedback = self.feedback_fn(pred_pose, targ_pose, target_areas[nimg][j])
                    loss_ij = ious[nimg][i][j] * (self.feedback_loss_fn(pred_feedback, targ_feedback) + \
                                                  F.binary_cross_entropy(pred_pose[:,2], targ_pose[:,2]))
                    loss.append(loss_ij)
        loss = sum(loss)
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
        M = len(detections)
        # parse and initialize
        features = [d['features'] for d in detections] # List[Tensor[L,256,D,D]]
        poses = [[detections[i]['keypoints']] for i in range(M)]
        feedbacks = [[] for i in range(M)]
        if self.training:
            target_poses = [t['keypoints'] for t in targets] # List[Tensor[L',K,3]]
            target_areas = [t['area'] for t in targets] # List[Tensor[1]]
            losses = [] # List[Tensor[1]]
            boxes = [d['boxes'].detach() for d in detections] # List[Tensor[L, 4]]
            target_boxes = [t['boxes'] for t in targets] # List[Tensor[L', 4]]
            ious = compute_ious(boxes, target_boxes)
        # make batches
        features_batch, idx_list = self.batch_from_list(features) # Tensor[L*M, 256, D, D]
        poses_batch, _ = self.batch_from_list([d['keypoints'] for d in detections]) # Tensor[L*M, K, 3]
        if self.training:
            target_poses_batch, target_idx_list = self.batch_from_list(target_poses) # Tensor[L'*M, K, 3]
        # create intermediary poses if necessary
        if self.training and self.interpolate_poses:
            intermediary_poses = pose_interpolation([d['keypoints'] for d in detections], target_poses, num_iterations) # List[List[Tensor[L, L', K, 3]]]
        # iterative refinement
        for iteration in range(num_iterations):
            poses_batch, feedback_batch = self.forward_iteration(features_batch, poses_batch)
            feedback_batch.squeeze(-1) # Tensor[L*M, 1] -> Tensor[L*M]
            poses_list = list(poses_batch.split(idx_list, 0))
            feedbacks_list = list(feedback_batch.split(idx_list, 0))
            for i in range(M):
                poses[i].append(poses_list[i])
                feedbacks[i].append(feedbacks_list[i])
            if self.training:
                loss_kwargs = {'pred_poses': poses_list, 'target_poses': target_poses, 
                               'pred_feedbacks': feedbacks_list, 'ious': ious,
                               'target_areas': target_areas}
                if self.interpolate_poses:
                    loss_kwargs['intermediary_poses'] = intermediary_poses[iteration]
                loss = self.compute_loss(**loss_kwargs)
                losses.append(loss)
        # complete detections
        for i in range(M):
            detections[i]['keypoints'] = poses[i]
            detections[i]['feedbacks'] = feedbacks[i]
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
        poses_features = self.preprocess_poses(poses_batch, features_batch.shape[-1]).to(features_batch.device)
        poses_features.requires_grad_(True)
        inp = torch.cat([features_batch, poses_features], dim=1)
        feedback = self.net(inp)
        grad = torch.autograd.grad(feedback.mean(), poses_features, retain_graph=True)[0]
        poses_features = poses_features + self.feedback_rate*grad.detach()
        pooled_poses = F.avg_pool2d(poses_features, kernel_size=poses_features.shape[-1])
        x = F.relu(torch.cat([pooled_poses[:,i] for i in range(0, pooled_poses.shape[1], 3)], dim=1))
        y = F.relu(torch.cat([pooled_poses[:,i] for i in range(1, pooled_poses.shape[1]+1, 3)], dim=1))
        v = F.sigmoid(torch.cat([pooled_poses[:,i] for i in range(2, pooled_poses.shape[1]+1, 3)], dim=1))
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
        poses_features = self.preprocess_poses(poses_batch, features_batch.shape[-1]).to(features_batch.device)
        inp = torch.cat([features_batch, poses_features], dim=1)
        feedback = self.net(inp)
        poses_features += self.feedback_rate*feedback
        pooled_poses = F.avg_pool2d(poses_features, kernel_size=poses_features.shape[-1])
        x = F.relu(torch.cat([pooled_poses[:,i] for i in range(0, pooled_poses.shape[1], 3)], dim=1))
        y = F.relu(torch.cat([pooled_poses[:,i] for i in range(1, pooled_poses.shape[1]+1, 3)], dim=1))
        v = F.sigmoid(torch.cat([pooled_poses[:,i] for i in range(2, pooled_poses.shape[1]+1, 3)], dim=1))
        next_poses = torch.cat([x, y, v], dim=-1)
        return next_poses, feedback