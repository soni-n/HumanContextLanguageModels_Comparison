echo $@
CUDA_VISIBLE_DEVICES=0,1 \
python -O EIHaRT/run_ft_hart.py \
    --learning_rate 4.00540716638447e-05 \
    --early_stopping_patience 6 \
    --weight_decay 0.01 \
    --model_name_or_path $1 \
    --task_type document \
    --task_name stance \
    --num_labels 3 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 15 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --output_dir /chronos_data/nisoni/eihart_hartbase_outputs/Doc_FT_5e5aftertrials/stance/clim \
    --add_history \
    --initial_history EIHaRT/initial_history/initialized_history_tensor.pt \
    --train_table EIHaRT/data/datasets/stance/clim_train_all.pkl \
    --dev_table EIHaRT/data/datasets/stance/clim_dev_all.pkl \
    --test_table EIHaRT/data/datasets/stance/clim_test_all.pkl \
    # --overwrite_output_dir \

    
    