echo $@
CUDA_VISIBLE_DEVICES=2,1 \
python -O /home/nisoni/eihart/EIHaRT/run_mtl_eihart.py \
    --label_names ac_labels \
    --ac_task_name ope \
    --model_name_or_path /chronos_data/nisoni/eihart_outputs/PT/5e5_eihart_100pc_8ep_hsc_ce_mse_hart_age/checkpoint-227720 \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_eval \
    --output_dir /chronos_data/nisoni/eihart_outputs/PT_eval/5e5lastep_maxvalblock \
    --per_device_eval_batch_size 20 \
    --max_val_blocks 63 \
    --block_size 1024 \
    --db HuLM \
    --hostname localhost \
    --test_table lbp_testset_created \
    # --overwrite_output_dir \
    # --instantiate_hart \

    
    
