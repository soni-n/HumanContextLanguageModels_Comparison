echo $@
CUDA_VISIBLE_DEVICES=0 \
python -O /home/nisoni/eihart/EIHaRT/run_ft_eihart.py \
    --learning_rate 0.00021100689963260628 \
    --early_stopping_patience 6 \
    --model_name_or_path /chronos_data/nisoni/eihart_outputs/PT/509e7_eihart_100pc_8ep_hsc_ce_mse_hart_age \
    --task_type document \
    --num_labels 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 30 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --block_size 100 \
    --max_train_blocks 8 \
    --output_dir /chronos_data/nisoni/eihart_outputs/hart_base/TP/TP_TD2_32bs_30ep_100bls_bestHP \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --train_table age.united_states.2.2.category \
    --dev_table dummy \
    --test_table dummy \
    # --overwrite_output_dir \
