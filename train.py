# train.py
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import Adam, lr_scheduler
from lib.iterative_rcnn import get_iter_kprcnn_resnet18_oks, get_iter_kprcnn_resnet18_ief, \
                               get_iter_kprcnn_resnet50_oks, get_iter_kprcnn_resnet50_ief, \
                               get_kprcnn_resnet50
from train_utils import get_train_args, get_transform, \
                        get_target_transform, save_model, \
                        save_at_root, load_best_model, \
                        compute_mean_loss
from coco_utils import get_coco, collate_fn
from engine import train_model, validate_model
import os
import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

def print_header(epoch, num_epochs, iteration, num_iterations):
    print('*'*100)
    print('Iteration: {0}/{1}, Epoch {2}/{3}'.format(iteration, num_iterations, epoch+1, num_epochs))

def print_losses(train_loss, val_loss):
    print('|--- Training losses):')
    for k, l in train_loss.items():
        print('    |--- {0}: {1:.4f}'.format(k, l))
    print('|--- Validation losses:')
    for k, l in val_loss.items():
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

    # Build dataset and dataloader
    train_dataset = get_coco(args.train_imgs, args.train_annots, get_transform(), get_target_transform(), True)
    val_dataset = get_coco(args.val_imgs, args.val_annots, get_transform(), get_target_transform(), True)
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
    mean_train_losses = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    mean_val_losses = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    for num_iterations in range(first_iter, args.num_iterations+1):
        best_val_loss_for_iteration = float('inf')
        epoch = 0
        while epoch < args.num_epochs:
            tr_done = False
            vl_done = False
            try:
                print_header(epoch, args.num_epochs, num_iterations, args.num_iterations)
                # training
                # with torch.autograd.set_detect_anomaly(True):
                tr_losses, tr_times, empty = train_model(train_dataloader, model, device, num_iterations, optimizer, args.max_ds_size)
                train_times[str(num_iterations)]['forward'].extend(tr_times['forward'])
                train_times[str(num_iterations)]['backward'].extend(tr_times['backward'])
                train_losses[str(num_iterations)].append(tr_losses)
                mean_tr_losses = compute_mean_loss(tr_losses, num_iterations)
                mean_train_losses[str(num_iterations)].append(mean_tr_losses)   
                tr_done = True
                # validating
                vl_losses, vl_times = validate_model(val_dataloader, model, device, num_iterations)
                val_times[str(num_iterations)].extend(vl_times)
                val_losses[str(num_iterations)].append(vl_losses)
                mean_vl_losses = compute_mean_loss(vl_losses, num_iterations)                
                mean_val_losses[str(num_iterations)].append(mean_vl_losses)
                vl_done = True
                # prints
                # if epoch % args.print_frequency == 0:
                print_losses(mean_tr_losses, mean_vl_losses)
                # save model
                best_val_loss_for_iteration = save_model(model, mean_vl_losses, 
                                                        os.path.join(args.model_dir, args.model_name),
                                                        best_val_loss_for_iteration, num_iterations)
                # incr epoch
                epoch += 1
                if empty > .4:
                    print('RCNN predicts nothing, going to next iteration')
                    epoch = args.num_epochs+1
                    break
            except RuntimeError: # Optimal batch size hard to predict. If it happens, take back at previous iteration
                print('Catched CUDA error: Restarting training epoch with smaller batch size ...')
                # create smaller data loaders
                args.batch_size -= 1
                train_dataloader = DataLoader(train_dataset, num_workers=4, shuffle=True, 
                                  batch_size=args.batch_size, 
                                  collate_fn=collate_fn)
                val_dataloader = DataLoader(val_dataset, num_workers=4, shuffle=True, 
                                            batch_size=args.batch_size, 
                                            collate_fn=collate_fn)
                # reload previous model
                if epoch == 0:
                    if num_iterations == 0: # there will be no model to load ...
                        model = get_iter_kprcnn_resnet18_oks(args.keep_labels, args.iou_thresh, 
                                             args.feedback_rate, args.feedback_loss_fn,
                                             args.interpolate_poses, 
                                             args.num_conv_blocks_feedback,7).to(device)
                    else: # load previous iteration best model
                        model = load_best_model(os.path.join(args.model_dir, args.model_name), num_iterations-1)
                else: # load last best model for that number of iterations
                    model = load_best_model(os.path.join(args.model_dir, args.model_name), num_iterations)
                # ensure metric lists are the right length
                for lst, flag in zip([train_times[str(num_iterations)]['forward'],
                                     train_times[str(num_iterations)]['backward'],
                                     train_losses[str(num_iterations)],
                                     val_times[str(num_iterations)],
                                     val_losses[str(num_iterations)],
                                     mean_train_losses[str(num_iterations)],
                                     mean_val_losses[str(num_iterations)]],
                                     [tr_done, tr_done, tr_done, vl_done,
                                      vl_done, tr_done, vl_done]):
                    if flag: # if happend then, make it disappear
                        lst = lst[:-1]             
        model = load_best_model(os.path.join(args.model_dir, args.model_name), num_iterations)
    metrics = {'losses': {'train': train_losses, 'val': val_losses, 'mean_train': mean_train_losses, 'mean_val': mean_val_losses}, 
               'times': {'train': train_times, 'val': val_times}}
    save_at_root(metrics, os.path.join(root, 'train_val_metrics.pkl'))

if __name__ == '__main__':
    main()
