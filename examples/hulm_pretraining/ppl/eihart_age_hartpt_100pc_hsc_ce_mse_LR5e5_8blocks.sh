echo $@
CUDA_VISIBLE_DEVICES=0,1 \
python -O /home/nisoni/eihart/EIHaRT/run_mtl_eihart.py \
    --label_names ac_labels \
    --ac_task_name age \
    --model_name_or_path /chronos_data/nisoni/eihart_outputs/PT/5e5_eihart_100pc_8ep_hsc_ce_mse_hart_age/checkpoint-227720 \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_eval \
    --output_dir /chronos_data/nisoni/eihart_4blocks/PT_ppl/eihart_age_8blocks \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --db HuLM \
    --hostname localhost \
    --test_table fb20lbp_upt50_en_non_oosmsgs \
    --save_preds_labels \
    # --overwrite_output_dir \
    # --instantiate_hart \

    
    
