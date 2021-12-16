# FmFasterRCNN
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import *
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor, model_urls
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from typing import Tuple, List, Dict, Optional, Union
from .roi_heads import FmRoIHeads
from .generalized_rcnn import FmGeneralizedRCNN
import warnings

class FmFasterRCNN(FmGeneralizedRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
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
    ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")
        out_channels = backbone.out_channels
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        roi_heads = FmRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super().__init__(backbone, rpn, roi_heads, transform)

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        """
        FmFasterRCNN.forward: forward pass

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

        Returns:
            features (List[Tensor]): list of backbone features corresponding to detections 
                (RoIHeads outputs)
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
            (train mode) losses (Dict[str, Tensor]): Dictonary of losses
        """
        return super().forward(images, targets)

def fasterrcnn_resnet_fpn(backbone_arch, pretrained=True, num_classes=91, trainable_backbone_layers=None, progress=True, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained, trainable_backbone_layers, 5, 3)
    backbone = resnet_fpn_backbone(backbone_arch, pretrained, trainable_layers=trainable_backbone_layers)
    model = FmFasterRCNN(backbone, num_classes, **kwargs)
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
            warnings.warn(f'No model pretrained on COCO could be found. An imagenet pretrained one was pulled.')
    return model
