# train.py
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import Adam
from lib.iterative_rcnn import get_iterative_rcnn
from utils import get_train_args, get_transform, get_target_transform, save_model
from engine import train_model, validate_model

def collate_identity(x): return x

def print_losses(val_loss, train_loss, epoch, num_epochs, iteration, num_iterations):
    print('*'*20)
    print('Iteration: {0}/{1}, Epoch {2}/{3}'.format(iteration+1, num_iterations, epoch+1, num_epochs))
    print('|---Training losses:')
    for k, l in train_loss.items():
        print('    |--- {0}: {1:.4f}'.format(k, l))
    print('Validation losses:')
    for k, l in train_loss.items():
        print('    |--- {0}: {1:.4f}'.format(k, l))

def main():
    # Arguments and configuration
    args = get_train_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Build model
    model = get_iterative_rcnn(args.backbone_arch, args.keypoint_rcnn, args.feedback, args.num_classes,
                               args.rcnn_pretrained, args.train_box_head, args.train_kp_head,
                               args.trainable_backbone_layers, args.keep_labels, args.iou_thresh,
                               args.feedback_loss_fn, args.feedback_rate, args.interpolate_poses,
                               args.num_conv_blocks_feedback).to(device)

    # Build dataset and dataloader
    train_dataset = CocoDetection(root=args.train_imgs, 
                                  annFile=args.train_annots, 
                                  transform=get_transform(),
                                  target_transform=get_target_transform())
    val_dataset = CocoDetection(root=args.val_imgs, 
                                  annFile=args.val_annots, 
                                  transform=get_transform(),
                                  target_transform=get_target_transform())
    train_dataloader = DataLoader(train_dataset, num_workers=4, shuffle=True, 
                                  batch_size=args.batch_size, 
                                  collate_fn=collate_identity)
    val_dataloader = DataLoader(val_dataset, num_workers=4, shuffle=True, 
                                batch_size=args.batch_size, 
                                collate_fn=collate_identity)
    train_dataloader.mean_pose = args.mean_pose
    val_dataloader.mean_pose = args.mean_pose

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr)

    # Incremental training: num_epochs at depth num_iterations
    model.train()
    train_losses = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    val_losses = {str(iteration): [] for iteration in range(args.num_iterations+1)}
    for num_iterations in range(args.num_iterations+1):
        best_val_loss_for_iteration = float('inf')
        for epoch in range(args.num_epochs):
            # training
            tr_losses = train_model(train_dataloader, model, device, num_iterations)
            train_losses[str(num_iterations)].append(tr_losses)
            # validating
            vl_losses = validate_model(val_dataloader, model, device, num_iterations)
            val_losses[str(num_iterations)].append(vl_losses)
            # print
            if epoch % args.print_frequency == 0:
                print_losses(train_losses[str(num_iterations)][-1], 
                            val_losses[str(num_iterations)][-1], 
                            epoch, args.num_epochs, num_iterations,
                            args.num_iterations)
        # save model
        best_val_loss_for_iteration = save_model(model, val_losses[str(num_iterations)], 
                                                 os.path.join(args.model_dir, args.model_name),
                                                 best_val_loss_for_iteration)

if __name__ == '__main__':
    main()
