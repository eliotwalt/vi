# train.py
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import Adam, lr_scheduler
from lib.iterative_rcnn import get_iter_kprcnn_resnet18_oks, get_iter_kprcnn_resnet18_ief, get_kprcnn_resnet50
from train_utils import get_train_args, get_transform, get_target_transform, save_model, save_at_root, load_best_model
from coco_utils import get_coco, collate_fn
from engine import train_model, validate_model
import os
import warnings
warnings.filterwarnings("ignore")

def print_header(epoch, num_epochs, iteration, num_iterations):
    print('*'*100)
    print('Iteration: {0}/{1}, Epoch {2}/{3}'.format(iteration+1, num_iterations, epoch+1, num_epochs))

def print_losses(val_loss, train_loss):
    print('|--- Training losses:')
    for k, l in train_loss.items():
        print('    |--- {0}: {1:.4f}'.format(k, l))
    print('|--- Validation losses:')
    for k, l in train_loss.items():
        print('    |--- {0}: {1:.4f}'.format(k, l))

def main():
    # Arguments and configuration
    args = get_train_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    root = os.path.join(args.model_dir, args.model_name)
    save_at_root(args, os.path.join(root, 'args.pkl'))

    # Build model
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

    # Build dataset and dataloader
    train_dataset = get_coco(args.train_imgs, args.train_annots, get_transform(), get_target_transform(), True)
    val_dataset = get_coco(args.val_imgs, args.val_annots, get_transform(), get_target_transform(), False)
    train_dataloader = DataLoader(train_dataset, num_workers=4, shuffle=True, 
                                  batch_size=args.batch_size, 
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, num_workers=4, shuffle=True, 
                                batch_size=args.batch_size, 
                                collate_fn=collate_fn)
    train_dataloader.mean_pose = args.mean_pose
    val_dataloader.mean_pose = args.mean_pose

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr)

    # Incremental training: num_epochs at depth num_iterations
    model.train()
    train_times = {str(iteration): {'forward': [], 'backward': []} for iteration in range(args.num_iterations+1)}
    val_times = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    train_losses = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    val_losses = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    for num_iterations in range(1, args.num_iterations+1):
        best_val_loss_for_iteration = float('inf')
        for epoch in range(args.num_epochs):
            print_header(epoch, args.num_epochs, num_iterations, args.num_iterations)
            # training
            tr_losses, tr_times = train_model(train_dataloader, model, device, num_iterations, optimizer)
            train_times[str(num_iterations)]['forward'].append(tr_times['forward'])
            train_times[str(num_iterations)]['backward'].append(tr_times['backward'])
            train_losses[str(num_iterations)].append(tr_losses)
            # validating
            vl_losses, vl_times = validate_model(val_dataloader, model, device, num_iterations)
            val_times[str(num_iterations)].append(vl_times)
            val_losses[str(num_iterations)].append(vl_losses)
            # print
            # if epoch % args.print_frequency == 0:
            print_losses(train_losses[str(num_iterations)][-1], 
                         val_losses[str(num_iterations)][-1])
            # save model
            best_val_loss_for_iteration = save_model(model, val_losses[str(num_iterations)], 
                                                    os.path.join(args.model_dir, args.model_name),
                                                    best_val_loss_for_iteration, num_iterations)
        model = load_best_model(os.path.join(args.model_dir, args.model_name), num_iterations)
    metrics = {'losses': {'train': train_losses, 'val': val_losses}, 'times': {'train': train_times, 'val': val_times}}

if __name__ == '__main__':
    main()
