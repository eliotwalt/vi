# train.py
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import Adam
from lib.iterative_rcnn import get_iterative_rcnn
from utils import get_train_args, get_train_transform, get_test_transform

def print_losses(val_loss, train_loss, epoch, num_epochs):
    print('this would be the loss print')

def main():
    # Arguments and configuration
    args = get_train_args()

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
    train_dataloader = DataLoader(train_dataset, num_workers=4, shuffle=True, batch_size=args.batch_size).to(device)
    val_dataloader = DataLoader(val_dataset, num_workers=4, shuffle=True, batch_size=args.batch_size).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr).to(device)

    # Incremental training: num_epochs at depth num_iterations
    model.train()
    train_losses = []
    val_losses = []
    for num_iterations in range(args.num_iterations+1):
        for epoch in range(args.num_epochs):
            # training
            for train_images, train_targets in train_dataloader:
                detections, losses = model(train_images, train_targets, num_iterations) # init_pose ??????
                optimizer.zero_grad()
                if num_iterations == 0: [loss.backward() for (k, loss) in losses if k != 'feedbacks'] # iteration 0 -> train RCNN
                else: loss['feebacks'][-1].backward() # iteration >= 1 -> train last feedback iteration
                optimizer.step()
                for k, loss in losses.items(): # add all losses to losses list
                    if not loss_key == 'feedbacks':
                        losses[k] = loss.detach().cpu()
                    else:
                        losses[k] = [l.detach().cpu() for l in loss]
                train_losses.append(losses)
            # validating
            for val_images, val_targets in val_dataloader:
                detections, losses = model(val_images, val_targets, num_iterations) # init_pose ??????
                for k, loss in losses.items(): # add all losses to losses list
                    if not loss_key == 'feedbacks':
                        losses[k] = loss.detach().cpu()
                    else:
                        losses[k] = [l.detach().cpu() for l in loss]
                val_losses.append(losses)
            # print
            print_losses(train_losses[-1], val_losses[-1], epoch, args.num_epochs)

if __name__ == '__main__':
    main()
