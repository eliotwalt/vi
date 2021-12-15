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
    for area, normalized_detection in normalized_detections:
        area = normalized_detection['area']
        for i in range(2):
            normalized_detection['keypoints'][:,:,i] = normalized_detection['keypoints'][:,:,i]/area
    return normalized_detections

def inverse_normalize(detections: List[Dict[str, Tensor]]):
    """
    inverse_normalize: apply inverse normalization

    Args:
        normalized_detections (List[Dict[str, Tensor]]): List of M dictionaries containing 			
            detections normalized keypoints along with initial attributes:
                - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                    K normalized keypoints in [x, y, visibility] format, defining the object.
                - area (Tensor[N]): area of bounding boxes
                - others

    Returns:
        detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
            attributes:
                - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                    K keypoints in [x, y, visibility] format, defining the object.
                - area (Tensor[N]): area of bounding boxes
                - others
    """
    detections = normalized_detections
    for detection in detections:
        area = detection['area']
        for i in range(2):
            detection['keypoints'][:,:,i] = detection['keypoints'][:,:,i]*area
    return detections