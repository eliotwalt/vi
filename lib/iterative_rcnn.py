# IterativeRCNN, IterativeKeypointRCNN, IterativeFasterRCNN
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Dict, Optional, Union, Any
from rcnn.generalized_rcnn import FmGeneralizedRCNN
from rcnn.faster_rcnn import FmFasterRCNN
from rcnn.keypoint_rcnn import FmKeypointRCNN
from feedback.iterative_net import BaseIterativeNet, AdditiveIterativeNet, EnergyAscentIterativeNet
from selector import BoxSelector
from ops.normalizer import KeypointNormalizer

class IterativeRCNN(nn.Module):
    """
    IterativeRCNN: Base class for iterative refinement from RCNN models predictions

    Args:
        rcnn (FmGeneralizedRCNN): rcnn network
        iter_net (BaseIterativeNet): iterative network
        selector (BoxSelector): selector to filter rcnn predictions
    """
    def __init__(self, rcnn: FmGeneralizedRCNN, iter_net: BaseIterativeNet, selector: BoxSelector):
        super().__init__()
        self.rcnn = rcnn
        self.iter_net = iter_net
        self.selector = selector

    def forward(self, )
    '''
    rcnn: images, (targets) -> features, detection, (losses)
    selector: detections, features -> detections, features
    normalizer(areas): detections -> detections
    iter_net: features, init_pose, num_iterations, targets(only keypoints), areas -> poses, feedbacks, (losses)

    MAYBE PASS TARGETS DICTIONARIES TO ITER_NET TO MAKE IT CLEARER 
    '''
