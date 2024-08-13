echo $@
CUDA_VISIBLE_DEVICES=2,3 \
python -O /home/nisoni/eihart/EIHaRT/run_mtl_eihart.py \
    --label_names ac_labels \
    --ac_task_name ope \
    --model_name_or_path /chronos_data/nisoni/eihart_User_TL/eihart_age4blocks_to_ope4bl/continue_PT/checkpoint-56930 \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_predict \
    --output_dir /chronos_data/nisoni/eihart_Error_disparity_outputs/eihartAge4bltoOPE4bl/continue_PT_lbp_eval_30to45 \
    --per_device_eval_batch_size 20 \
    --max_val_blocks 63 \
    --block_size 1024 \
    --db HuLM \
    --hostname localhost \
    --test_table lbp_testset_created \
    --save_preds_labels \
    # --overwrite_output_dir \
    # --instantiate_hart \

    
    
