import argparse
import torch
from torch import Tensor
from torch import nn
import torchvision
import torchvision.transforms as T
import os
import pickle

def get_transform():
    return T.Compose([T.ToTensor()])

def get_target_transform():
    return None

# def get_train_args():
#     p = argparse.ArgumentParser()
#     # Required
#     p.add_argument(
#         '--backbone_arch', 
#         help="(required) resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50','resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'",
#         required=True
#     )
#     p.add_argument(
#         '--feedback',
#         help="(required) feedback signal. Possible values are 'oks', 'ief'",
#         required=True
#     )
#     p.add_argument(
#         '--batch_size',
#         help="(required) batch size",
#         required=True,
#         type=int
#     )
#     p.add_argument(
#         '--lr',
#         help='(required) learning rate',
#         required=True,
#         type=float,
#     )
#     p.add_argument(
#         '--num_iterations',
#         help='(required) number of feedback iteration maximum',
#         required=True,
#         type=int,
#     )
#     p.add_argument(
#         '--num_epochs',
#         help='(required) number of training epochs',
#         required=True,
#         type=int,
#     )
#     p.add_argument(
#         '--model_dir',
#         help='(required) directory in which to save trained models',
#         required=True
#     )
#     p.add_argument(
#         '--model_name',
#         help='(required) name for saved trained models'
#     )
#     # Optional
#     p.add_argument(
#         '--train_imgs',
#         help="path to dir containing train images",
#         required=False,
#         default='data/images/train2017/'
#     )
#     p.add_argument(
#         '--train_annots',
#         help="path to dir containing train annotations",
#         required=False,
#         default='data/annotations/person_keypoints_train2017.json'
#     )
#     p.add_argument(
#         '--val_imgs',
#         help="path to dir containing val images",
#         required=False,
#         default='data/images/val2017'
#     )
#     p.add_argument(
#         '--val_annots',
#         help="path to dir containing val annotations",
#         required=False,
#         default='data/annotations/person_keypoints_val2017.json'
#     )
#     p.add_argument(
#         '--mean_pose',
#         help='path to pickled tensor of mean pose',
#         required=False,
#         default='data/mean_poses/train2017.pt'
#     )
#     p.add_argument(
#         '--keypoint_rcnn',
#         help='if specified, use FmKeypointRCNN, otherwise FmFasterRCNN',
#         action='store_true',
#         required=False
#     )    
#     p.add_argument(
#         '--rcnn_pretrained',
#         help='if specified try to find pretrained rcnn for the architecture. If not found, a warning is printed and only the backbone is pretrained on Imagenet.',
#         required=False,
#         action='store_true'
#     )
#     p.add_argument(
#         '--num_classes',
#         help='number of classes for rcnn',
#         required=False,
#         default=2,
#         type=int,
#     )
#     p.add_argument(
#         '--train_box_head',
#         help='if specified the box head is trained, otherwise, it is frozen',
#         action='store_true',
#         required=False,
#     )
#     p.add_argument(
#         '--train_kp_head',
#         help='if specified the keypoint head is trained, otherwise, it is frozen',
#         action='store_true',
#         required=False,
#     )
#     p.add_argument(
#         '--trainable_backbone_layers',
#         help='number of backbone layer to let be trainable between 0 and . If 0, the backbone is frozen',
#         required=False,
#         default=0,
#         type=int
#     )
#     p.add_argument(
#         '--keep_labels',
#         help='list of labels to keep in selection process',
#         required=False,
#         nargs='+',
#         default=[1],
#         type=int,
#     )
#     p.add_argument(
#         '--iou_thresh',
#         help='iou threshold in selection process',
#         required=False,
#         default=.3,
#         type=float,
#     )
#     p.add_argument(
#         '--feedback_loss_fn',
#         help='loss function for feedback prediction, possible values are "l2", "l1" and "smooth_l1',
#         required=False,
#         default='smooth_l1'
#     )
#     p.add_argument(
#         '--feedback_rate',
#         help=' step size of feedback updates',
#         required=False,
#         default=.1,
#         type=float,
#     )
#     p.add_argument(
#         '--interpolate_poses',
#         help='if specified, intermediary poses are interpolated',
#         required=False,
#         action='store_true'
#     )
#     p.add_argument(
#         '--num_conv_blocks_feedback',
#         help='number of convolutional blocks in iter_net',
#         required=False,
#         default=1,
#         type=int,
#     )
#     p.add_argument(
#         '--print_frequency',
#         help='print frequency (in number of epochs)',
#         required=False,
#         default=10,
#         type=int,
#     )
#     # parse argv
#     args = p.parse_args()
#     # create directories
#     if not os.path.isdir(args.model_dir):
#         os.mkdir(args.model_dir)
#     if not os.path.isdir(os.path.join(args.model_dir, args.model_name)):
#         os.mkdir(os.path.join(args.model_dir, args.model_name))
#     # load mean pose
#     args.mean_pose = torch.load(args.mean_pose)
#     # feedback loss parsing
#     if args.feedback_loss_fn == 'l2':
#         args.feedback_loss_fn = nn.MSELoss(reduction='none')
#     elif args.feedback_loss_fn == 'l1':
#         args.feedback_loss_fn = nn.L1Loss(reduction='none')
#     elif args.feedback_loss_fn == 'smooth_l1':
#         args.feedback_loss_fn = nn.SmoothL1Loss(reduction='none')
#     else:
#         raise AttributeError(f'invalid feedback_loss_fn, possible values are "l2", "l1" and "smooth_l1')
#     return args

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
        default=.3,
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
        args.feedback_loss_fn = nn.MSELoss(reduction='none')
    elif args.feedback_loss_fn == 'l1':
        args.feedback_loss_fn = nn.L1Loss(reduction='none')
    elif args.feedback_loss_fn == 'smooth_l1':
        args.feedback_loss_fn = nn.SmoothL1Loss(reduction='none')
    else:
        raise AttributeError(f'invalid feedback_loss_fn, possible values are "l2", "l1" and "smooth_l1')
    return args

def save_at_root(obj, path):
    with open(path, 'wb') as h:
        pickle.dump(obj, h)

def save_model(model, val_losses, path_, best_loss, num_iterations):
    '''If mean val_losses smaller than best_loss save model and return new best'''
    mean_val_loss = sum(val_losses)/len(val_losses)
    if mean_val_loss < best_loss:
        path_ = os.path.join(path_, num_iterations)
        path_ = os.path,join(path_, 'model.pt')
        torch.save(model, path_)
        return mean_val_loss
    return best_loss

def load_best_model(path_, num_iterations):
    from lib.iterative_rcnn import get_iter_kprcnn_resnet18_oks, get_iter_kprcnn_resnet18_ief, get_kprcnn_resnet50
    with open(os.path.join(path_, 'args.pkl'), 'rb') as h:
        args = pickle.load(h)
    path_ = os.path.join(path_, num_iterations)
    path_ = os.path.join(path_, 'model.pt')
    if args.model == 'iter_kprcnn_resnet18_oks':
        model = get_iter_kprcnn_resnet18_oks(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, 
                                             args.num_conv_blocks_feedback,7).to(device)
    elif args.model == 'iter_kprcnn_resnet18_ief':
        model = get_iter_kprcnn_resnet18_ief(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, args.num_conv_blocks_feedback,7).to(device)
    else:
        raise AttributeError(f'`{args.model}` not recognized as a TRAINABLE model.')
    return model
    

if __name__ == '__main__':
    args = get_train_args()
    print(args)


    