import argparse
import torch
from torch import Tensor
from torch import nn
import torchvision
import torchvision.transforms as T
import os
import pickle
from lib.iterative_rcnn import get_iter_kprcnn_resnet18_oks, get_iter_kprcnn_resnet18_ief, \
                               get_iter_kprcnn_resnet50_oks, get_iter_kprcnn_resnet50_ief, \
                               get_kprcnn_resnet50

def get_transform():
    return T.Compose([T.ToTensor()])

def get_target_transform():
    return None

def get_train_args():
    p = argparse.ArgumentParser()
    # Required
    p.add_argument(
        '--model',
        help='(required) model type: iter_kprcnn_resnet18_oks, iter_kprcnn_resnet18_ief, ff_kprcnn_resnet50',
        required=True
    )
    p.add_argument(
        '--batch_size',
        help="(required) batch size",
        required=True,
        type=int
    )
    p.add_argument(
        '--lr',
        help='(required) learning rate',
        required=True,
        type=float,
    )
    p.add_argument(
        '--num_iterations',
        help='(required) number of feedback iteration maximum',
        required=True,
        type=int,
    )
    p.add_argument(
        '--num_epochs',
        help='(required) number of training epochs',
        required=True,
        type=int,
    )
    p.add_argument(
        '--model_dir',
        help='(required) directory in which to save trained models',
        required=True
    )
    p.add_argument(
        '--model_name',
        help='(required) name for saved trained models'
    )
    # Optional
    p.add_argument(
        '--train_imgs',
        help="path to dir containing train images",
        required=False,
        default='data/images/train2017/'
    )
    p.add_argument(
        '--train_annots',
        help="path to dir containing train annotations",
        required=False,
        default='data/annotations/person_keypoints_train2017.json'
    )
    p.add_argument(
        '--val_imgs',
        help="path to dir containing val images",
        required=False,
        default='data/images/val2017'
    )
    p.add_argument(
        '--val_annots',
        help="path to dir containing val annotations",
        required=False,
        default='data/annotations/person_keypoints_val2017.json'
    )
    p.add_argument(
        '--mean_pose',
        help='path to pickled tensor of mean pose',
        required=False,
        default='data/mean_poses/train2017.pt'
    )
    p.add_argument(
        '--keep_labels',
        help='list of labels to keep in selection process',
        required=False,
        nargs='+',
        default=[1],
        type=int,
    )
    p.add_argument(
        '--iou_thresh',
        help='iou threshold in selection process',
        required=False,
        default=.6,
        type=float,
    )
    p.add_argument(
        '--feedback_loss_fn',
        help='loss function for feedback prediction, possible values are "l2", "l1" and "smooth_l1',
        required=False,
        default='smooth_l1'
    )
    p.add_argument(
        '--feedback_rate',
        help=' step size of feedback updates',
        required=False,
        default=.1,
        type=float,
    )
    p.add_argument(
        '--interpolate_poses',
        help='if specified, intermediary poses are interpolated',
        required=False,
        action='store_true'
    )
    p.add_argument(
        '--num_conv_blocks_feedback',
        help='number of convolutional blocks in iter_net',
        required=False,
        default=1,
        type=int,
    )
    p.add_argument(
        '--print_frequency',
        help='print frequency (in number of epochs)',
        required=False,
        default=10,
        type=int,
    )
    p.add_argument(
        '--max_ds_size',
        help='maximum dataset size',
        required=False,
        default=8000,
        type=int
    )
    # parse argv
    args = p.parse_args()
    # create directories
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.isdir(os.path.join(args.model_dir, args.model_name)):
        os.mkdir(os.path.join(args.model_dir, args.model_name))
    # load mean pose
    args.mean_pose = torch.load(args.mean_pose)
    # feedback loss parsing
    if args.feedback_loss_fn == 'l2':
        args.feedback_loss_fn = nn.MSELoss(reduction='mean')
    elif args.feedback_loss_fn == 'l1':
        args.feedback_loss_fn = nn.L1Loss(reduction='mean')
    elif args.feedback_loss_fn == 'smooth_l1':
        args.feedback_loss_fn = nn.SmoothL1Loss(reduction='mean')
    else:
        raise AttributeError(f'invalid feedback_loss_fn, possible values are "l2", "l1" and "smooth_l1')
    return args

def save_at_root(obj, path):
    with open(path, 'wb') as h:
        pickle.dump(obj, h)

def save_model(model, val_losses, path_, best_loss, num_iterations):
    '''If mean val_losses smaller than best_loss save model and return new best'''
    mean_loss = [v for v in val_losses.values()]
    mean_loss = sum(mean_loss)/len(mean_loss)
    if mean_loss < best_loss:
        path_ = os.path.join(path_, str(num_iterations))
        if not os.path.isdir(path_): os.mkdir(path_)
        path_ = os.path.join(path_, 'model.pt')
        print(f'Saving new best model at {path_}')
        torch.save(model.state_dict(), path_)
        return mean_loss
    return best_loss

def load_best_model(path_, num_iterations):
    with open(os.path.join(path_, 'args.pkl'), 'rb') as h:
        args = pickle.load(h)
    path_ = os.path.join(path_, str(num_iterations))
    if not os.path.isdir(path_): os.mkdir(path_)
    path_ = os.path.join(path_, 'model.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.model == 'iter_kprcnn_resnet18_oks':
        model = get_iter_kprcnn_resnet18_oks(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, 
                                             args.num_conv_blocks_feedback,7).to(device)
        first_iter = 0
    elif args.model == 'iter_kprcnn_resnet18_ief':
        model = get_iter_kprcnn_resnet18_ief(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, args.num_conv_blocks_feedback,7).to(device)
        first_iter = 0
    elif args.model == 'iter_kprcnn_resnet50_oks':
        model = get_iter_kprcnn_resnet50_oks(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, 
                                             args.num_conv_blocks_feedback,7).to(device)
        first_iter = 1
    elif args.model == 'iter_kprcnn_resnet50_ief':
        model = get_iter_kprcnn_resnet50_ief(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, args.num_conv_blocks_feedback,7).to(device)
        first_iter = 1
    else:
        raise AttributeError(f'`{args.model}` not recognized as a TRAINABLE model.')
    print('loading model from:', path_)
    _ = model.load_state_dict(torch.load(path_))
    return model
    
def compute_mean_loss(losses, iteration):
    if iteration == 0:
        keys = list(losses[0].keys())
        n = {k: len(losses) for k in keys}
    else:
        keys = [str(i) for i in range(1,iteration+1)]
        n = {k: 0 for k in keys}
    mean_losses = {k: 0 for k in keys}
    for loss in losses:
        for key in keys:
            if iteration == 0:
                mean_losses[key] += loss[key]
            else:
                if loss['feedback'][key] is not None:
                    mean_losses[key] += loss['feedback'][key]
                    n[key] += 1
    for k,v in mean_losses.items():
        if n[k] == 0: # all feedback losses were None
            mean_losses[k] = 1e7 # arbitrary recognizable large number
        else:
            mean_losses[k] /= n[k]
    return mean_losses

if __name__ == '__main__':
    args = get_train_args()
    print(args)


    