# resnet50 -> backbone resnet50
# frozen backbone -> 
# frozen heads
python train.py --backbone_arch resnet50 \
                --feedback ief \
                --batch_size 28 \
                --lr 0.01 \
                --num_iterations 10 \
                --num_epochs 15 \
                --model_dir saved_models \
                --model_name resnet50_frozen_ief \
                --rcnn_pretrained \
                --num_classes 91 \
                --interpolate_poses
                