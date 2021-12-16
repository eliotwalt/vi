import torch
from torch import Tensor
from torch import nn

def nested_detach(losses):
    if isinstance(losses, dict):
        return {k: nested_detach(l) for (k,l) in losses.items()}
    elif isinstance(losses, list):
        return nested_detach([loss_elt for loss_elt in losses])
    elif isinstance(losses, Tensor):
        return losses.detach().item()
    else:
        return None
    
def process_dataloader_output(tup):
    images = []
    targets = []
    for x, y in tup:
        images.append(x)
        targets.append(y)
    return images, targets

def train_model(train_dataloader, model, device, num_iterations):
    tr_losses = []
    for tup in train_dataloader:
        train_images, train_targets = process_dataloader_output(tup)
        # to device
        train_images = [image.to(device) for image in train_images]
        # get initial pose
        if not model.is_keypoint_rcnn:
            init_pose = train_dataloader.mean_pose
        else:
            init_pose = None
        # prediction and loss
        detections, losses = model(images=train_images, 
                                   targets=train_targets, 
                                   num_iterations=num_iterations,
                                   init_pose=init_pose)
        # Backward pass
        step = False
        optimizer.zero_grad()
        if num_iterations == 0: # iteration 0 -> train RCNN
            step = True
            [loss.backward() for (k, loss) in losses.items() if k != 'feedbacks'] 
        else: # iteration >= 1 -> train last feedback iteration (if present)
            if 'feedbacks' in loss.keys(): # could not be the case if the selector return empty list (some imgs have 0 keypoints for some reason)
                step = True
                loss['feebacks'][-1].backward()
        if step:
            optimizer.step()
        # detach and get float and append all losses (rcnn+feedback)
        tr_losses.append(nested_detach(losses))
    return losses

def validate_model(val_dataloader, model, device, num_iterations):
    losses = []
    for tup in val_dataloader:
        val_images, val_targets = process_dataloader_output(tup)
        # to device
        val_images = [image.to(device) for image in val_images]
        # get initial pose
        if not model.is_keypoint_rcnn:
            init_pose = val_dataloader.mean_pose
        else:
            init_pose = None
        # prediction and loss
        detections, losses = model(images=val_images, 
                                   targets=val_targets, 
                                   num_iterations=num_iterations,
                                   init_pose=init_pose)
        # detach and get float and append all losses (rcnn+feedback)
        val_losses.append(nested_detach(losses))
    return val_losses