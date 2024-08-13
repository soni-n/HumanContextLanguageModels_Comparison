echo $@
CUDA_VISIBLE_DEVICES=2,3 \
python -O EIHaRT/run_ft_hart.py \
    --weight_decay 0.01 \
    --model_name_or_path $1 \
    --task_type document \
    --task_name stance \
    --num_labels 3 \
    --do_predict \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --output_dir $2 \
    --add_history \
    --initial_history EIHaRT/initial_history/initialized_history_tensor.pt \
    --test_table $3 \
    --save_preds_labels \
    # --overwrite_output_dir \

    
    