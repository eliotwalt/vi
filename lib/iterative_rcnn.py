# IterativeGeneralizedRCNN, IterativeKeypointRCNN, IterativeFasterRCNN
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_area
from typing import Tuple, List, Dict, Optional, Union, Any
from .rcnn.generalized_rcnn import FmGeneralizedRCNN
from .rcnn.faster_rcnn import FmFasterRCNN, fasterrcnn_resnet_fpn
from .rcnn.keypoint_rcnn import FmKeypointRCNN, keypointrcnn_resnet_fpn
from .feedback.iterative_net import BaseIterativeNet, AdditiveIterativeNet, EnergyAscentIterativeNet
from .ops.selector import ObjectSelector
from .ops.normalizer import normalize, inverse_normalize
from .utils import Oks, KeypointSignedError, FeedbackResnet

# ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
#  'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
coco_weights = Tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

class IterativeGeneralizedRCNN(nn.Module):
    """
    IterativeGeneralizedRCNN: Base class for iterative refinement from RCNN models predictions

    Args:
        rcnn (Union[FmFasterRCNN, FmKeypointRCNN]): rcnn network
        iter_net (BaseIterativeNet): iterative network
        selector (ObjectSelector): selector to filter rcnn predictions
    """
    def __init__(
        self, 
        rcnn: Union[FmFasterRCNN, FmKeypointRCNN], 
        iter_net: BaseIterativeNet, 
        selector: ObjectSelector, 
    ):
        super().__init__()
        self.rcnn = rcnn
        self.iter_net = iter_net
        self.selector = selector
        if type(rcnn)==FmFasterRCNN:
            self.is_keypoint_rcnn = False
        elif type(rcnn)==FmKeypointRCNN:
            self.is_keypoint_rcnn = True
        else:
            raise AttributeError(f'rcnn must be Union[FmFasterRCNN, FmKeypointRCNN]')

    def detection_initial_pose(self, detections: List[Dict[str, Tensor]], init_pose: Union[Tensor, List[Tensor]]):
        """
        IterativeGeneralizedRCNN.detection_initial_pose: parse init_pose and add it to detction dictionaries

        Args:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[L, 4]): coordinates of L bbox, formatted as [x0, y0, x1, y1]
                    - others
            init_pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): initial pose. If Tensor, it 
                represent a generic pose to be with all features. If List[Tensor], it contains a
                specific pose for each element of each features

        Returns:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - keypoints (List[Tensor[L, K, 3]]): For each one of the L objects, it 
                        contains the K keypoints in [x, y, visibility] format, defining the object.
                    - others
        """
        if isinstance(pose, list):
            assert len(detections)==len(init_pose), f'detections and init_pose must have the same length when init_pose is a list.'
        elif isinstance(pose, Tensor):
            assert len(init_pose.shape==2), f'init_pose must be of shape (K,3)'
            init_pose = init_pose.unsqueeze(0)
            init_pose = [init_pose.repeat(d['boxes'].shape[0], 1, 1) for d in detections]
        else:
            raise TypeError(f'`init_pose` must be Tensor or List[Tensor], not {type(init_pose)}.')
        for ipose, detection in zip(init_pose, detections):
            assert detection['boxes'].shape[0]==ipose.shape[0], 'dimension mismatch between boxes ({}) and initial pose ({}'.format(
                detection['boxes'].shape[0], ipose.shape[0])
            detection['keypoints'] = ipose
        return detections

    def detection_area(self, detections):
        for detection in detections:
            detection['area'] = box_area(detection['boxes'])
        return detections

    def forward(
        self, 
        images: List[Tensor],
        num_iterations: int,
        init_pose: Union[Tensor, List[Tensor]]=None, # MUST ALREADY BE NORMALIZED and of shape (K,3) if single
        targets: Optional[List[Dict[str, Tensor]]]=None,
    ):
        """
        IterativeGeneralizedRCNN.forward: forward pass

        Args:
            images (List[Tensor]): List of M Tensors representing images
            num_iterations (int): number of feedback iterations
            (optional) init_pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): initial pose. If Tensor, it 
                represent a generic pose to be with all features. If List[Tensor], it contains a
                specific pose for each element of each features
            (optional) targets (List[Dict[str, Tensor]]): List of dictionaries containing target 
                attributes:
                    - boxes (Tensor[L', 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Tensor[L']): label for each bbox. 0 is background
                    - image_id (Tensor[1]): an image identifier. It should be unique between 
                        all the images in the dataset and is used during evaluation
                    - iscrowd (Tensor[L']): instances with iscrowd=True will be ignored during 
                        evaluation
        
        Returns:
            detections (List[List[Dict[str, Tensor]]]): list of num_iterations+1 lists of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - features (Tensor[N, 256, D, D]): RoI feature maps
                    - (optional) keypoints (Union[List[Tensor[N, K, 3]], Tensor[N, K, 3]]): For each one of the N 
                        objects, it contains the K keypoints in [x, y, visibility] format, defining the object. 			
            	        visibility=0 means that the keypoint is not visible. Note that for data 
                        augmentation, the notion of flipping a keypoint is dependent on the data 
                        representation, and you should probably adapt references/detection/transforms.py 
                        for your new keypoint representation
            (optional) losses (Dict[str, Tensor]): Dictionary of losses
        """
        # assertions
        assert num_iterations >= 0, f'`num_iterations` must be >= 0.'
        if not self.is_keypoint_rcnn:
            assert init_pose is not None, f'Must pass an initial pose when using faster RCNN.'
        else:
            if init_pose is not None: 
                warnings.warn('init_pose was passed but keypoint_rcnn used therefore it will be ignored.')
        # rcnn initial detections
        if self.training:
            assert targets is not None, f'Must pass targets in training mode.'
            detections, losses = self.rcnn(images, targets)
        else:
            detections = self.rcnn(images)
        # return now if num_iterations is 0 (for incremental training)
        if num_iterations == 0:
            if self.training:
                return detections, losses
            else:
                return detections
        # filtering to reduce batch size
        # detections = self.selector(detections)
        i = 0
        print(detections[0])
        while i < len(detections):
            if detections[i]['boxes'].size(0) <= 1:
                detections.pop(i)
            else: i += 1
        if len(detections)==0:
            if self.training:
                losses['feedback'] = {str(i): None for i in range(num_iterations)}
                return detections, losses
            else:
                return detections
        # compute prediction areas
        detections = self.detection_area(detections)
        # normalize keypoints (only if is_keypoint_rcnn)
        if self.is_keypoint_rcnn:
            detections = normalize(detections)
        # preprocess init_pose
        if not self.is_keypoint_rcnn:
            detections = self.detection_initial_pose(detections, init_pose)
        # iterative refinement
        if self.training:
            detections, feedback_losses = self.iter_net(detections, num_iterations, targets)
            losses['feedback'] = {str(i): feedback_losses[i] for i in range(num_iterations)}
        else:
            detections = self.iter_net(detections, num_iterations)
        # inverse normalize
        detections = inverse_normalize(detections)
        # reshape
        detections = self.reshape_detections(detections, num_iterations)
        if self.training:
            return detections, losses
        else:
            return detections

    def reshape_detections(self, detections: List[Dict[str, List[Tensor]]], num_iterations: int) -> List[List[Dict[str, Tensor]]]:
        M = len(detections)
        N = num_iterations
        reshaped_dets = [[{k: v for k,v in detections[i].items()} for i in range(M)] for _ in range(N)]
        for i in range(M):
            dets = detections[i]
            for k, v in dets.items():
                for j in range(N):
                    reshaped_dets[j][i][k] = v[j]
        return reshaped_dets
            
        
        

def get_iter_kprcnn_resnet18_oks(
    keep_labels, 
    iou_thresh, 
    feedback_rate, 
    feedback_loss_fn,
    interpolate_poses,
    num_conv_blocks_feedback,
    features_dim=7
):
    rcnn = keypointrcnn_resnet_fpn('resnet18')
    selector = ObjectSelector(keep_labels, iou_thresh)
    feedback_fn = Oks(coco_weights)
    feedback_net = FeedbackResnet(out_channels=1, features_dim=features_dim, num_blocks=num_conv_blocks_feedback)
    iter_net = EnergyAscentIterativeNet(feedback_net, feedback_rate, feedback_loss_fn, interpolate_poses, feedback_fn)
    return IterativeGeneralizedRCNN(rcnn, iter_net, selector)

def get_iter_kprcnn_resnet18_ief(
    keep_labels, 
    iou_thresh, 
    feedback_rate, 
    feedback_loss_fn,
    interpolate_poses,
    num_conv_blocks_feedback,
    features_dim=7
):
    rcnn = keypointrcnn_resnet_fpn('resnet18')
    selector = ObjectSelector(keep_labels, iou_thresh)
    feedback_fn = KeypointSignedError()
    feedback_net = FeedbackResnet(out_channels=len(coco_weights)*3, features_dim=features_dim, num_blocks=num_conv_blocks_feedback)
    iter_net = AdditiveIterativeNet(feedback_net, feedback_rate, feedback_loss_fn, interpolate_poses, feedback_fn)
    return IterativeGeneralizedRCNN(rcnn, iter_net, selector)

def get_kprcnn_resnet50():
    return keypointrcnn_resnet_fpn('resnet50')
