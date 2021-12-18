# KeypointNormalizer
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union

def normalize(detections: List[Dict[str, Tensor]]):
    """
    normalize: apply normalization

    Args:
        detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
            attributes:
                - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                    K keypoints in [x, y, visibility] format, defining the object.
                - area (Tensor[N]): area of bounding boxes
                - others

    Returns:
        normalized_detections (List[Dict[str, Tensor]]): List of M dictionaries containing 			
            detections normalized keypoints along with initial attributes:
                - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                    K normalized keypoints in [x, y, visibility] format, defining the object.
                - area (Tensor[N]): area of bounding boxes
                - others
    """
    normalized_detections = detections
    for normalized_detection in normalized_detections:
        area = normalized_detection['area']
        if len(area.shape) == 1: area = area.reshape(-1,1)
        for i in range(2):
            normalized_detection['keypoints'][:,:,i] = normalized_detection['keypoints'][:,:,i]/area
    return normalized_detections

def inverse_normalize(normalized_detections: List[Dict[str, Tensor]]):
    """
    inverse_normalize: apply inverse normalization

    Args:
        normalized_detections (List[Dict[str, List[Tensor]]]): List of M dictionaries containing 			
            list of detections normalized keypoints along with initial attributes:
                - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                    K normalized keypoints in [x, y, visibility] format, defining the object.
                - area (Tensor[N]): area of bounding boxes
                - others

    Returns:
        detections (List[Dict[str, List[Tensor]]]): List of M dictionaries containing detections 
            attributes:
                - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                    K keypoints in [x, y, visibility] format, defining the object.
                - area (Tensor[N]): area of bounding boxes
                - others
    """
    for k, normalized_detection in enumerate(normalized_detections):
        for numit in range(len(normalized_detection['keypoints'])):
            area = normalized_detection['area'][numit]
            kps = normalized_detection['keypoints'][numit].clone()
            if len(area.shape) == 1: area.reshape(-1,1)
            for i in range(2):
                kps[:,:,i] = kps[:,:,i] * area
            normalized_detection['keypoints'][numit] = kps
    detections = normalized_detections
    return detections