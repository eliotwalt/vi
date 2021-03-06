import torch
from torch import Tensor
from torch import nn
from tqdm import tqdm
from time import time

def nested_detach(losses):
    if isinstance(losses, dict):
        return {k: nested_detach(l) for (k,l) in losses.items()}
    elif isinstance(losses, list):
        return nested_detach([loss_elt for loss_elt in losses])
    elif isinstance(losses, Tensor):
        return losses.detach().cpu().item()
    else:
        return None

def train_model(train_dataloader, model, device, num_iterations, optimizer, max_ds_size):
    tr_losses = []
    tr_times = {'forward': [], 'backward': []}
    empties = 0
    print('Training ...')
    pbar = tqdm(train_dataloader)
    i = 0
    for train_images, train_targets in pbar:
        # to device
        train_images = [image.to(device) for image in train_images]
        train_targets = [{k: v.to(device) for k, v in train_target.items()} for train_target in train_targets]
        [t['keypoints'].to(torch.float32) for t in train_targets]
        # get initial pose
        if not model.is_keypoint_rcnn:
            init_pose = train_dataloader.mean_pose
        else:
            init_pose = None
        # prediction and loss
        t0 = time()
        detections, losses = model(images=train_images, 
                                   targets=train_targets, 
                                   num_iterations=num_iterations)
        t1 = time()
        tr_times['forward'].append(t1-t0)
        # Backward pass
        step = False
        t0 = time()
        optimizer.zero_grad()
        if num_iterations == 0: # iteration 0 -> train RCNN
            step = True
            loss = sum(loss for loss in losses.values())
            loss.backward()
        else: # iteration >= 1 -> train last feedback iteration (if present)
            if losses['feedback'][str(num_iterations)] is not None: # could not be the case if the selector return list of None (some imgs have 0 keypoints for some reason)
                step = True
                loss = losses['feedback'][str(num_iterations)]
                loss.backward()
        if step:
            optimizer.step()
        t1 = time()
        tr_times['backward'].append(t1-t0)
        # detach and get float and append all losses (rcnn+feedback)
        losses = nested_detach(losses)
        pbar_kwargs = {'i': f'{i+1}/{max_ds_size}'}
        for k,v in losses.items():
            if num_iterations == 0:
                if k != 'feedback':
                    pbar_kwargs[k]=v 
            else:
                if k == 'feedback':
                    pbar_kwargs[k]=v
        tr_losses.append(losses)        
        pbar.set_postfix(pbar_kwargs)
        if num_iterations == 0:
            for det in detections:
                if len(det['boxes'] == 0):
                    empties += 1
        i += 1
        if i == max_ds_size:
            break
    return tr_losses, tr_times, empties/len(train_dataloader)/train_dataloader.batch_size

def validate_model(val_dataloader, model, device, num_iterations):
    val_losses = []
    val_times = []
    print('Validating ...')
    pbar = tqdm(val_dataloader)
    for val_images, val_targets in pbar:
        # to device
        val_images = [image.to(device) for image in val_images]
        val_targets = [{k: v.to(device) for k, v in val_target.items()} for val_target in val_targets]
        [t['keypoints'].to(torch.float32) for t in val_targets]
        # get initial pose
        if not model.is_keypoint_rcnn:
            init_pose = val_dataloader.mean_pose
        else:
            init_pose = None
        # prediction and loss
        t0 = time()
        detections, losses = model(images=val_images, 
                                   targets=val_targets, 
                                   num_iterations=num_iterations,
                                   init_pose=init_pose)
        t1 = time()
        val_times.append(t1-t0)
        # detach and get float and append all losses (rcnn+feedback)
        losses = nested_detach(losses)
        val_losses.append(losses)        
        pbar.set_postfix(losses)
    return val_losses, val_times