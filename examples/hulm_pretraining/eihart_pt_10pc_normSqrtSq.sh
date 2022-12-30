echo $@
# CUDA_VISIBLE_DEVICES=1,2 \
/home/nisoni/miniconda3/envs/hulm/bin/python -O /home/nisoni/eihart/EIHaRT/run_mtl_eihart.py \
    --label_names ac_labels \
    --ac_task_name age \
    --learning_rate 0.00024447089107483056 \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --add_history \
    --initial_history /home/nisoni/eihart/EIHaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir /home/nisoni/eihart/EIHaRT/output/eihart_10pc_5ep_SqrtSqScale \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 10 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --db HuLM \
    --hostname aiorion.ai.stonybrook.edu \
    --train_table fb20lbp_upt50_en_train_10pc \
    --dev_table fb20lbp_upt50_en_non_oosmsgs \
    --test_table fb20lbp_upt50_en_non_oosmsgs \
    # --overwrite_output_dir \

    
    
