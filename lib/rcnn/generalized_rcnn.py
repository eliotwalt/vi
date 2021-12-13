# FmGeneralizedRCNN
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
import torch
from torch import nn, Tensor
from .roi_heads import FmRoIHeads

class FmGeneralizedRCNN(nn.Module):
    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module):
        super().__init__()
        assert isinstance(roi_heads, FmRoIHeads)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
    
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
                    - (optional) keypoints (Tensor[N, K, 3]): For each one of the N objects, it 
                        contains the K keypoints in [x, y, visibility] format, defining the object. 			
            	        visibility=0 means that the keypoint is not visible. Note that for data 
                        augmentation, the notion of flipping a keypoint is dependent on the data 
                        representation, and you should probably adapt references/detection/transforms.py 
                        for your new keypoint representation
            (train mode) losses (Dict[str, Tensor]): Dictonary of losses
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        bckb_features = self.backbone(images.tensors)
        if isinstance(bckb_features, torch.Tensor):
            bckb_features = OrderedDict([("0", bckb_features)])
        proposals, proposal_losses = self.rpn(images, bckb_features, targets)
        detections, detector_losses, features = self.roi_heads(bckb_features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return features, detections, losses
        else:
            return features, detections
