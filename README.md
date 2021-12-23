# Iterative Human Pose Refinement via Energy Ascent

This repository contains the source code of the project "Iterative Human Pose Refinement via Energy Ascent" realised for the CS-503 course at EPFL.

## Structure

The repository has the following structure.

- `data`
  - `compute_mean_pose.py` script to compute mean pose of COCO annotation file
  - `mean_poses`: contain such mean poses
- `env/requirements.txt` dependencies
- `lib` contains the source code
  - `feedback` contains the iterative feedback network implementation
  - `ops` contains non-parameteric operations
  - `rcnn` contains modified `torchvision` RCNNs and RoIHeads
  - `iterative_rcnn.py` contains the implementation of the full model
  - `utils.py` contains utilities, such as feedback functions
- `coco_utils.py` contains adaptations of `torchvision` helper functions to filter and reformat COCO datasets
- `train_utils.py`  helper functions used in the training script
- `train.py` training script
- `train_iter_kprcnn_resnet(50|18)_(oks|ief).sh` shell script running `train.py` with predefined arguments

## Usage

**Note** The code was tested with python 3.8 and CUDA 11.0.

After installing the dependencies in `requirements.txt`, the default shell scripts can be run directly. Custom arguments can also be applied to `train.py`:

```bash
$ python train.py --help
usage: train.py [-h] --model MODEL --batch_size BATCH_SIZE --lr LR --num_iterations NUM_ITERATIONS --num_epochs NUM_EPOCHS --model_dir MODEL_DIR
                [--model_name MODEL_NAME] [--train_imgs TRAIN_IMGS] [--train_annots TRAIN_ANNOTS] [--val_imgs VAL_IMGS] [--val_annots VAL_ANNOTS]
                [--mean_pose MEAN_POSE] [--keep_labels KEEP_LABELS [KEEP_LABELS ...]] [--iou_thresh IOU_THRESH] [--feedback_loss_fn FEEDBACK_LOSS_FN]
                [--feedback_rate FEEDBACK_RATE] [--interpolate_poses] [--num_conv_blocks_feedback NUM_CONV_BLOCKS_FEEDBACK]
                [--print_frequency PRINT_FREQUENCY] [--max_ds_size MAX_DS_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         (required) model type: iter_kprcnn_resnet18_oks,
                        iter_kprcnn_resnet18_ief, iter_kprcnn_resnet50_oks,
                        iter_kprcnn_resnet50_ief
  --batch_size BATCH_SIZE
                        (required) batch size
  --lr LR               (required) learning rate
  --num_iterations NUM_ITERATIONS
                        (required) number of feedback iteration maximum
  --num_epochs NUM_EPOCHS
                        (required) number of training epochs
  --model_dir MODEL_DIR
                        (required) directory in which to save trained models
  --model_name MODEL_NAME
                        (required) name for saved trained models
  --train_imgs TRAIN_IMGS
                        path to dir containing train images
  --train_annots TRAIN_ANNOTS
                        path to dir containing train annotations
  --val_imgs VAL_IMGS   path to dir containing val images
  --val_annots VAL_ANNOTS
                        path to dir containing val annotations
  --mean_pose MEAN_POSE
                        path to pickled tensor of mean pose
  --keep_labels KEEP_LABELS [KEEP_LABELS ...]
                        list of labels to keep in selection process
  --iou_thresh IOU_THRESH
                        iou threshold in selection process
  --feedback_loss_fn FEEDBACK_LOSS_FN
                        loss function for feedback prediction, possible values are "l2",
                        "l1" and "smooth_l1"
  --feedback_rate FEEDBACK_RATE
                        step size of feedback updates
  --interpolate_poses   if specified, intermediary poses are interpolated
  --num_conv_blocks_feedback NUM_CONV_BLOCKS_FEEDBACK
                        number of convolutional blocks in iter_net
  --print_frequency PRINT_FREQUENCY
                        print frequency (in number of epochs)
  --max_ds_size MAX_DS_SIZE
                        maximum dataset size
```