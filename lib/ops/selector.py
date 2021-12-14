# ObjectSelector
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import nms 
from typing import Tuple, List, Dict, Optional, Union

class ObjectSelector(object):
    """
    ObjectSelector: Module to select RCNN detections and (optionally) features from a list of labels
        and a hard IOU threshold

    Args:
        keep_labels (List[int]): List of labels to keep in the filtering process, default [1]
        iou_thresh (float): IoU threshold to keep bounding boxes, default .3
    """
    def __init__(self, keep_labels: List[int]=[1], iou_thresh: float=.3):
        self.keep_labels = keep_labels
        self.iou_thresh = iou_thresh
    
    def select(self, detections: List[Dict[str, Tensor]], features: Optional[List[Tensor]]=None):
        """
        ObjectSelector.select: apply selection

        Args:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - features (Tensor[N, 256, D, D]): RoI feature maps


        Returns:
            slct_detections (List[Dict[str, Tensor]]): List of M dictionaries containing selected 
                detections attributes:
                    - boxes (Tensor[L, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[L]): the predicted labels for each detection
                    - scores (Tensor[L]): the scores of each detection
                    - (if keypoint=True) keypoints (Tensor[L, K, 3]): For each one of the L 
                        objects, it contains the K keypoints in [x, y, visibility] format,
                        defining the object.
        """
        slct_detections = [{k: [] for k in det.keys()} for det in detections]
        if features is not None: slct_features = [[] for _ in features]
        for i in range(len(detections)):
            idx_keep = nms(detections[i]['boxes'], detections[i]['scores'], self.iou_thresh)
            mask_keep = torch.full((detections[i]['labels'].shape[0],), False)
            mask_keep[idx_keep] = True
            for label in self.keep_labels:
                mask_keep = torch.bitwise_and(mask_keep, detections[i]['labels']==label)
            for key in slct_detections[i].keys():
                slct_detections[i][key] = detections[i][key][mask_keep]
        return slct_detections
        