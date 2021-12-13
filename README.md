# Iterative Human Pose Refinement via Energy Ascent

blabla

## Currently doing

- Intermediary feature maps: It seems to be working currently. We use the features used to perform box detection, of shape (N, 256, 7, 7). However, it may be a better to use the features used to perform keypoint estimation if available... **TOTEST** In that case, we could replace lines 206 and 207 of `lib.rcnn.roi_heads.py` by:
```python
images_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
keypoint_features_flat = self.keypoint_head(images_features)
```

## Data stuff

- install `pycocotools`: `pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI`
  - oks and stuff ? 
- Data set example at <a href='https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html'>this link</a>
- need `area`  and `keypoints_weights`

## Useful stuff

- visualization: https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html?highlight=_utils

## Modules

`M`: number of images

`N`: number of detections in each image (may not be the same for each image)

`L`: number of detections selected by the selector in each image (L<=N)

`K`: num_keypoints

- [x] **lib.rcnn.roi_heads.FmRoIHeads**
  - [x] `keypoint_roi_pool` and`box_roi_pool` ?
- [x] **lib.rcnn.generalized_rcnn.FmGeneralizedRCNN**
- [x] **lib.rcnn.faster_rcnn.FmFasterRCNN**
- [x] **lib.rcnn.keypoint_rcnn.FmKeypointRCNN**
  - [x] Adapt from FmFasterRCNN
- [x] **lib.ops.selector.ObjectSelector**
  - [x] as object not nn.Module
  - [x] Rename
  - [x] Re-factor output
- [x] **lib.ops.normalizer.KeypointNormalizer**
  - [x] kp/sqrt(area) -> as for oks ! (double check)
- [x] **lib.feedback.iterative_net.BaseIterativeNet**
- [x] **lib.feedback.iterative_net.AdditiveIterativeNet**
  - [x] Rename
- [x] **lib.feedback.iterative_net.EnergyAscentIterativeNet**
  - [x] Rename
- [ ] **lib.iterative_rcnn.IterativeRCNN**
- [ ] **lib.iterative_rcnn.IterativeKeypointRCNN**
  - [ ] quick builders (rcnn part <a href='https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html'>this link</a>)
- [ ] **lib.iterative_rcnn.IterativeFasterRCNN**
  - [ ] quick builders (rcnn part <a href='https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html'>this link</a>)
- [x] **lib.feedback.feedback_utils.pose_interpolation(current_pose(req grad!), target_pose, it)**
- [x] **lib.utils.oks(input(req grad!), target)**
  - [x] add "normalize" kwarg to CONSTRUCTOR because in iter_net already normalized !!
  - [ ] not sure how to deal with visibility. Ignored for the moment ...
- [ ] **lib.utils.kp_loss**
  - [ ] WHAT IS IT ???????????
  - [ ] losses, how exactly do we decide this output is this detection ? i.e how do we ensure that the order of the detections in detections\[i\]\['keypoints'\] is the same as in targets\[i\]\['keypoints'\] ????
- [ ] **lib.utils.ap50**
  - [ ] the metric they use in the papers ?
  - [ ] pycocotools ?

**FmRoIHeads** [RoIHead]

```python
"""
forward

Returns:
	result ?
	losses ?
	box_features ? / keypoint_features ?
"""
```

Questions:

- L.133: we have a `keypoint_roi_pool` just as `box_roi_pool`. Shouldn't we return that one for FmKeypointRCNN ? Is it also (256, 7, 7) ? No, it is  (256, 14, 14)

**FmFasterRCNN** [FmGeneralizedRCNN]

```python
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
```

Questions:

- Why return `proposals` ?
- `losses` keys ?

**FmKeypointRCNN** [FmFasterRCNN]

```python
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
```

Questions:

- Why return `proposals` ?
- `losses` keys ?
- Augmentation: How to augment keypoints ?

**ObjectSelector** [object]

  ```python
  """
  ObjectSelector: Module to select RCNN detections and (optionally) features from a list of labels
      and a hard IOU threshold
  
  Args:
      keep_labels (List[int]): List of labels to keep in the filtering process, default [1]
      iou_thresh (float): IoU threshold to keep bounding boxes, default .3
  """
  """
  ObjectSelector.select: apply selection
  
  Args:
      features (List[Tensor[N, *]]): list of M backbone features corresponding to detections 
          (RoIHeads outputs)
      detections (List[Dict[str, Tensor]]): List of M dictionaries containing detections 
          attributes:
              - boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
              - labels (Int64Tensor[N]): the predicted labels for each detection
              - scores (Tensor[N]): the scores of each detection
  
  Returns:
      slct_features (List[Tensor[L, *]]): list of M backbone features corresponding to detections 
          (RoIHeads outputs)
      slct_detections (List[Dict[str, Tensor]]): List of M dictionaries containing selected 
          detections attributes:
              - boxes (Tensor[L, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
              - labels (Int64Tensor[L]): the predicted labels for each detection
              - scores (Tensor[L]): the scores of each detection
              - (if keypoint=True) keypoints (Tensor[L, K, 3]): For each one of the L 
                  objects, it contains the K keypoints in [x, y, visibility] format,
                  defining the object.
  """
  ```

**KeypointNormalizer** [object]

```python
"""
KeypointNormalizer: class to normalize and inverse normalize detection keypoints such that 
	all their coordinates are in [0, 1] by x' = (x-x0)/(x1-x0) and y' = (y-y0)/(y1-y0)
	
Args:
	None
"""
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
	normalized_detections (List[Dict[str, Tensor]]): List of M dictionaries containing 			detections normalized keypoints along with initial attributes:
			- boxes (Tensor[N, 4]): coordinates of N bbox, formatted as [x0, y0, x1, y1]
			- labels (Int64Tensor[N]): the predicted labels for each detection
        	- scores (Tensor[N]): the scores of each detection
        	- keypoints (Tensor[N, K, 3]): For each one of the N objects, it contains the
            	K keypoints in [x', y', visibility] format, defining the object.
"""
"""
KeypointNormalizer.inverse_normalize: apply inverse normalization

Args:
	normalized_detections (List[Dict[str, Tensor]]): List of M dictionaries containing 			detections with normalized keypoints along with initial attributes:
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
```

**BaseIterativeNet** [nn.Module]

```python
"""
BaseIterativeNet: Iterative pose estimation from RCNN RoI
    
Args:
    net (nn.Module): iterative network
    feedback_rate (float): feedback step size
    feedback_loss_fn (Any): a callable to compute the loss function on the target and 
        predicted signals. kwargs must be "input" and "target" for predicted and target
        feedback respectively
    (optional) interpolate_pose (bool): if true, the intermediary poses are linearly
    	interpolated between initial and target poses, default=True
    (optional) feedback_fn (Any): a callable to compute the target feedback value. Only
    	used when interpolate_poses=True
"""
"""
BaseIterativeNet.forward: forward pass

Args:
	features (List[Tensor[L, 256, D, D]]): list of M tensors corresponding to the feature 
		maps associated to M images
	init_pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): initial pose. If Tensor, it 
		represent a generic pose to be with all features. If List[Tensor], it contains a
    	specific pose for each element of each features
    num_iterations (int): number of feedback iterations
    (train mode) targets (List[Tensor[L, K, 3]]): List of M tensors containing the target
    	poses for each objects.
    
Returns:
	poses (List[Tensor[L, K, 3]]): List of `num_iterations` tensors containing pose
    	estimates at each iteration.
	feedbacks (List[Tensor[feedback_dim]]): List of `num_iterations` tensors containing
		feedback signal estimates at each iteration.
	(train mode) losses (List[Tensor[1]]): List of `num_iterations` tensors containing
		losses averaged over batch
"""
```

**AdditiveIterativeNet** [BaseIterativeNet]

```python
"""
AdditiveIterativeNet: Iterative pose estimation from RCNN RoI additive dense feedback
	
Args:
	net (nn.Module): iterative network
	feedback_rate (float): feedback step size
	feedback_loss_fn (Any): a callable to compute the loss function on the target and 
        predicted signals. kwargs must be "input" and "target" for predicted and target
        feedback respectively
	interpolate_pose (bool): if true, the intermediary poses are linearly interpolated
		between initial and target poses
	input_shape (Tuple[int, int, int]): shape of input features
"""
"""
AdditiveIterativeNet.forward: forward pass

Args:
	features (List[Tensor[L, 256, D, D]]): list of M tensors corresponding to the feature 
		maps associated to M images
	init_pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): initial pose. If Tensor, it 
		represent a generic pose to be with all features. If List[Tensor], it contains a
    	specific pose for each element of each features
    num_iterations (int): number of feedback iterations
    (train mode) target (List[Tensor[L, K, 3]]): target pose
    
Returns:
	poses (List[Tensor[L, K, 3]]): List of `num_iterations` tensors containing pose
    	estimates at each iteration.
	feedbacks (List[Tensor[L, K, 2]]): List of `num_iterations` tensors containing
		feedback signal estimates at each iteration. For additive feedback,the feedback
		dimension is the same as the keypoint dimensions (without visibility)
	(train mode) losses (List[Tensor[1]]): List of `num_iterations` tensors containing
		losses averaged over batch
"""
```

**EnergyAscentIterativeNet** [BaseIterativeNet]

```python
"""
EnergyAscentIterativeNet: Iterative pose estimation from RCNN RoI using energy ascent
	
Args:
	net (nn.Module): iterative network
	feedback_rate (float): feedback step size
	feedback_loss_fn (Any): a callable to compute the loss function on the target and 
        predicted signals. kwargs must be "input" and "target" for predicted and target
        feedback respectively
	interpolate_pose (bool): if true, the intermediary poses are linearly interpolated
		between initial and target poses
	input_shape (Tuple[int, int, int]): shape of input features
"""
"""
EnergyAscentIterativeNet.forward: forward pass

Args:
	features (List[Tensor[L, 256, D, D]]): list of M tensors corresponding to the feature 
		maps associated to M images
	init_pose (Union[Tensor[K, 3], List[Tensor[L, K, 3]]]): initial pose. If Tensor, it 
		represent a generic pose to be with all features. If List[Tensor], it contains a
    	specific pose for each element of each features
    num_iterations (int): number of feedback iterations
    (train mode) target (List[Tensor[L, K, 3]]): target pose
    
Returns:
	poses (List[List[Tensor[L, K, 3]]]): List of M lists of `num_iterations` tensors 			containing pose estimates at each iteration.
	feedbacks (List[List[Tensor[L, 1]]]): List of M lists of `num_iterations` tensors 
		containing feedback signal estimates at each iteration. For Energy ascent 
		feedback, the feedback dimension is 1.
	(train mode) losses (List[Tensor[1]]): List of `num_iterations` tensors containing
		losses averaged over batch
"""
```

**IterativeRCNN**

```python
"""
IterativeRCNN: Base class for iterative refinement from RCNN models predictions

Args:
	rcnn (FmGeneralizedRCNN): rcnn network
	iter_net (BaseIterativeNet): iterative network
		-> input_shape: (256, 7, 7) if faster else (256, 14, 14)
	selector (BoxSelector): selector to filter rcnn predictions
"""
"""
IterativeRCNN.forward: forward pass

Args:
	images
	init_pose
	num_iterations
	(train mode) target

Returns:
	poses
	feedbacks
	(train mode) losses: rcnn and iter format as rcnn one AND kp loss (whatever it means)
		at all iterations
"""

# Check that the output is not "empty" after selector
```

**IterativeKeypointRCNN** [IterativeRCNN]
**IterativeFasterRCNN** [IterativeRCNN]

**feedback_utils.intepolate_pose**

**feedback_utils.oks**

**eval_utils.kp_loss**



## Alternative 2: FmKeypointRCNN

`M`: number of images

`N`: number of detections in each image (may not be the same for each image)

- 