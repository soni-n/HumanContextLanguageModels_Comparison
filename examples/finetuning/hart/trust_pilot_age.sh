echo $@
CUDA_VISIBLE_DEVICES=2 \
python -O /home/nisoni/eihart/EIHaRT/run_ft_eihart.py \
    --learning_rate 5e-5 \
    --early_stopping_patience 6 \
    --model_name_or_path /home/nisoni/eihart/final8epoch_pt_model/checkpoint-227720 \
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
    --num_train_epochs 25 \
    --per_device_train_batch_size 60 \
    --per_device_eval_batch_size 60 \
    --block_size 200 \
    --max_train_blocks 8 \
    --output_dir /home/nisoni/eihart/EIHaRT/output/TP_FT/US_M_age_default \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --train_table dummy \
    --dev_table dummy \
    --test_table dummy \
    # --overwrite_output_dir \
