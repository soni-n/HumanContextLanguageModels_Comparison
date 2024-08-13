echo $@
CUDA_VISIBLE_DEVICES=1 \
python -O /home/nisoni/eihart/EIHaRT/run_ft_eihart.py \
    --model_name_or_path /chronos_data/nisoni/eihartOPE/TP_afterTrials/TP_TD2_32bs_30ep_100bls_bestHP \
    --task_type document \
    --num_labels 5 \
    --do_predict \
    --per_device_eval_batch_size 32 \
    --block_size 100 \
    --output_dir /chronos_data/nisoni/EIHaRT_Downstream_final/TP_eval_eihart_ope/TD2 \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --train_table age.united_states.2.2.category \
    --dev_table dummy \
    --test_table dummy \
    --save_preds_labels \
    # --overwrite_output_dir \
