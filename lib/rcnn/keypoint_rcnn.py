import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import *
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor, model_urls
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from typing import Tuple, List, Dict, Optional, Union
import warnings
from .roi_heads import FmRoIHeads
from .faster_rcnn import FmFasterRCNN

class FmKeypointRCNN(FmFasterRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=None,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # keypoint parameters
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        num_keypoints=17,
    ):

        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        if min_size is None:
            min_size = (640, 672, 704, 736, 768, 800)

        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")

        out_channels = backbone.out_channels

        if keypoint_roi_pool is None:
            keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if keypoint_head is None:
            keypoint_layers = tuple(512 for _ in range(8))
            keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

        if keypoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)

        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
        )

        self.roi_heads.keypoint_roi_pool = keypoint_roi_pool
        self.roi_heads.keypoint_head = keypoint_head
        self.roi_heads.keypoint_predictor = keypoint_predictor

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        """
        FmKeypointRCNN.forward: forward pass

        Args:
            images (List[Tensor]): List of M Tensors representing images
            (train mode) targets (List[Dict[str, Tensor]]): List of dictionaries containing target 
                attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Tensor[N]): label for each bbox. 0 is background
                    - image_id (Tensor[1]): an image identifier. It should be unique between 
                        all the images in the dataset and is used during evaluation
                    - iscrowd (Tensor[N]): instances with iscrowd=True will be ignored during 
                        evaluation
                    - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                        K keypoints in [x, y, visibility] format, defining the object. 			
                        visibility=0 means that the keypoint is not visible. 
                        Note that for data augmentation, the notion of flipping a keypoint 
                        is dependent on the data representation, and you should probably adapt 
                        references/detection/transforms.py for your new 
                        keypoint representation

        Returns:
            features (List[Tensor]): list of backbone features corresponding to detections 
                (RoIHeads outputs)
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                        K keypoints in [x, y, visibility] format, defining the object.
            (train mode) losses (Dict[str, Tensor]): Dictonary of losses
        """
        return super().forward(images, targets)


def keypointrcnn_resnet_fpn(backbone_arch, pretrained=True, num_classes=91, trainable_backbone_layers=None, progress=True, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained, trainable_backbone_layers, 5, 3)
    backbone = resnet_fpn_backbone('resnet18', pretrained, trainable_layers=trainable_backbone_layers)
    model = FmKeypointRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        key = None
        for modkey in model_urls.keys():
            if modkey.__contains__(backbone_arch):
                key = modkey
                break
        if key is not None:
            state_dict = load_state_dict_from_url(model_urls[key], progress=progress)
            model.load_state_dict(state_dict)
            overwrite_eps(model, 0.0)
        else:
            warnings.warn(f'No backbone pretrained on COCO could be found. An imagenet pretrained one was pulled.')
    return model
