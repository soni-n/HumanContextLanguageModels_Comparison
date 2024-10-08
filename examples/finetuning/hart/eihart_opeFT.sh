echo $@
CUDA_VISIBLE_DEVICES=2,3 \
python /home/nisoni/eihart/EIHaRT/run_ft_hart.py \
    --learning_rate 5e-5 \
    --early_stopping_patience 10 \
    --model_name_or_path /chronos_data/nisoni/eihart_outputs/PT/5e5_eihart_100pc_8ep_hsc_ce_mse_hart_age/checkpoint-227720 \
    --task_type user \
    --task_name ope \
    --num_labels 1 \
    --use_history_output \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_r_pear \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 25 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 4 \
    --output_dir /chronos_data/nisoni/USER_FT_eihart_outputs/AgePT_to_ope_5e5LR \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --db HuLM \
    --hostname localhost \
    --train_table fb20lbp_upt50_en_non_oosmsgs \
    --dev_table fb20lbp_upt50_en_non_oosmsgs \
    --test_table fb20lbp_upt50_en_non_oosmsgs \
    # --overwrite_output_dir \

    
    