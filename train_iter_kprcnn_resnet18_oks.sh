python train.py --model iter_kprcnn_resnet18_oks \
                --batch_size 3 \
                --lr 0.001 \
                --num_iterations 5 \
                --num_epochs 5 \
                --model_dir saved_models \
                --model_name iter_kprcnn_resnet18_oks \
                --print_frequency 1 \
                --interpolate_poses \
                --feedback_loss_fn l2