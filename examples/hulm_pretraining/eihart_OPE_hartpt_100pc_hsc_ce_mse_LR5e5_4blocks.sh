echo $@
sleep 1h
CUDA_VISIBLE_DEVICES=0,1 \
python -O /home/nisoni/eihart/EIHaRT/run_mtl_eihart.py \
    --label_names ac_labels \
    --ac_task_name ope \
    --learning_rate 5e-5 \
    --model_name_or_path /chronos_data/nisoni/HaRT_model \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir /chronos_data/nisoni/eihart_4blocks/PT/5e5_eihart_100pc_8ep_hsc_ce_mse_hart_OPE \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 4 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --db HuLM \
    --hostname localhost \
    --train_table fb20lbp_upt50_en_non_oosmsgs \
    --dev_table fb20lbp_upt50_en_non_oosmsgs \
    --test_table fb20lbp_upt50_en_non_oosmsgs \
    # --overwrite_output_dir \
    # --instantiate_hart \

    
    
