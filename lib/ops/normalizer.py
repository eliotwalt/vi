# KeypointNormalizer
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union

class KeypointNormalizer(object):
    """
    KeypointNormalizer: class to normalize and inverse normalize detection keypoints such that all 
        their coordinates are in [0, 1] by dividing by the area of the box
    
    Args:
        areas (List[int]): list of areas to normalize keypoints with
    """
    def __init__(self, areas: List[int]):
        self.areas = areas

    def normalize(self, detections: List[Dict[str, Tensor]]):
        """
        KeypointNormalizer.normalize: apply normalization

        Args:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                        K keypoints in [x, y, visibility] format, defining the object.

        Returns:
            normalized_detections (List[Dict[str, Tensor]]): List of M dictionaries containing 			
                detections normalized keypoints along with initial attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                        K keypoints in [x', y', visibility] format, defining the object.
        """
        normalized_detections = detections
        for area, normalized_detection in zip(self.areas, normalized_detections):
            for i in range(2):
                normalized_detection['keypoints'][:,:,i] = normalized_detection['keypoints'][:,:,i]/area
        return normalized_detections

    def inverse_normalize(self, normalized_detections: List[Dict[str, Tensor]]):
        """
        KeypointNormalizer.inverse_normalize: apply inverse normalization

        Args:
            normalized_detections (List[Dict[str, Tensor]]): List of M dictionaries containing 			
                detections with normalized keypoints along with initial attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                        K keypoints in [x', y', visibility] format, defining the object.

        Returns:
            detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
                with inverse normalized keypoints along with initial attributes:
                    - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
                    - labels (Int64Tensor[N]): the predicted labels for each detection
                    - scores (Tensor[N]): the scores of each detection
                    - keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
                        K keypoints in [x, y, visibility] format, defining the object.
        """
        detections = normalized_detections
        for area, detection in zip(self.areas, detections):
            for i in range(2): 
                detection['keypoints'][:,:,i] = detection['keypoints'][:,:,i]*area
        return detections