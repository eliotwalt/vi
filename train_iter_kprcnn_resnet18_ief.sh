python train.py --model iter_kprcnn_resnet18_ief \
                --batch_size 4 \
                --lr 0.001 \
                --num_iterations 7 \
                --num_epochs 5 \
                --model_dir saved_models \
                --model_name iter_kprcnn_resnet18_ief \
                --print_frequency 1 \
                --feedback_loss_fn l2