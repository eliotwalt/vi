python train.py --model iter_kprcnn_resnet50_oks \
                --batch_size 3 \
                --lr 0.0002 \
                --num_iterations 5 \
                --num_epochs 3 \
                --model_dir saved_models \
                --model_name iter_kprcnn_resnet50_oks \
                --print_frequency 1 \
                --interpolate_poses \
                --feedback_loss_fn smooth_l1 \
                --num_conv_blocks_feedback 2