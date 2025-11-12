# -*- coding: utf-8 -*-



####HY1
torchrun --nnodes 1 --nproc_per_node=8 --node_rank=0 --master_port=29900 train.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TRAIN.json" \
--datasets_fnspath "../../datasets/Data_lists_cleaned/" \
--datasets_fnsname "Families.json" \
--output_dir "./trainer_dir_paramS48HY1/" \
--model_path "/home/jovyan/h800fast/shared/models/Qwen/Qwen2_5/" \
--model_name "Qwen2.5-3B-Instruct" \
--deepspeed "" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 1 \
--max_time_len 2048 \
--kvgraph_dim 512 \
--hypergraph_dim 512 \
--hypergraph_layernum 1 \
--C_list "8" "16" "24" "32" "40" "48" \
--topK 168 \
--is_ae 1 \
--is_ac 1 \
--each_p 0.2 \
--batch_size 2 \
--accumulation_steps 2 \
--max_steps 30000 \
--learning_rate 1e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--save_strategy "steps" \
--save_steps 10000 \
--logging_steps 100



python test.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TEST.json" \
--output_dir "./test_dir_paramS48HY1/" \
--model_path "./trainer_dir_paramS48HY1/" \
--model_name "checkpoint-30000" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 0 \
--max_time_len 2048 \
--batch_size 32 \
--max_new_tokens 64 \
--do_sample 0 \
--flag "@30000"








####HY2
torchrun --nnodes 1 --nproc_per_node=8 --node_rank=0 --master_port=29900 train.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TRAIN.json" \
--datasets_fnspath "../../datasets/Data_lists_cleaned/" \
--datasets_fnsname "Families.json" \
--output_dir "./trainer_dir_paramS48HY2/" \
--model_path "/home/jovyan/h800fast/shared/models/Qwen/Qwen2_5/" \
--model_name "Qwen2.5-3B-Instruct" \
--deepspeed "" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 1 \
--max_time_len 2048 \
--kvgraph_dim 512 \
--hypergraph_dim 512 \
--hypergraph_layernum 2 \
--C_list "8" "16" "24" "32" "40" "48" \
--topK 168 \
--is_ae 1 \
--is_ac 1 \
--each_p 0.2 \
--batch_size 2 \
--accumulation_steps 2 \
--max_steps 30000 \
--learning_rate 1e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--save_strategy "steps" \
--save_steps 10000 \
--logging_steps 100



python test.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TEST.json" \
--output_dir "./test_dir_paramS48HY2/" \
--model_path "./trainer_dir_paramS48HY2/" \
--model_name "checkpoint-30000" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 0 \
--max_time_len 2048 \
--batch_size 32 \
--max_new_tokens 64 \
--do_sample 0 \
--flag "@30000"













####HY3
torchrun --nnodes 1 --nproc_per_node=8 --node_rank=0 --master_port=29900 train.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TRAIN.json" \
--datasets_fnspath "../../datasets/Data_lists_cleaned/" \
--datasets_fnsname "Families.json" \
--output_dir "./trainer_dir_paramS48HY3/" \
--model_path "/home/jovyan/h800fast/shared/models/Qwen/Qwen2_5/" \
--model_name "Qwen2.5-3B-Instruct" \
--deepspeed "" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 1 \
--max_time_len 2048 \
--kvgraph_dim 512 \
--hypergraph_dim 512 \
--hypergraph_layernum 3 \
--C_list "8" "16" "24" "32" "40" "48" \
--topK 168 \
--is_ae 1 \
--is_ac 1 \
--each_p 0.2 \
--batch_size 2 \
--accumulation_steps 2 \
--max_steps 30000 \
--learning_rate 1e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--save_strategy "steps" \
--save_steps 10000 \
--logging_steps 100



python test.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TEST.json" \
--output_dir "./test_dir_paramS48HY3/" \
--model_path "./trainer_dir_paramS48HY3/" \
--model_name "checkpoint-30000" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 0 \
--max_time_len 2048 \
--batch_size 32 \
--max_new_tokens 64 \
--do_sample 0 \
--flag "@30000"












####HY5
torchrun --nnodes 1 --nproc_per_node=8 --node_rank=0 --master_port=29900 train.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TRAIN.json" \
--datasets_fnspath "../../datasets/Data_lists_cleaned/" \
--datasets_fnsname "Families.json" \
--output_dir "./trainer_dir_paramS48HY5/" \
--model_path "/home/jovyan/h800fast/shared/models/Qwen/Qwen2_5/" \
--model_name "Qwen2.5-3B-Instruct" \
--deepspeed "" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 1 \
--max_time_len 2048 \
--kvgraph_dim 512 \
--hypergraph_dim 512 \
--hypergraph_layernum 5 \
--C_list "8" "16" "24" "32" "40" "48" \
--topK 168 \
--is_ae 1 \
--is_ac 1 \
--each_p 0.2 \
--batch_size 2 \
--accumulation_steps 2 \
--max_steps 30000 \
--learning_rate 1e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--save_strategy "steps" \
--save_steps 10000 \
--logging_steps 100



python test.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TEST.json" \
--output_dir "./test_dir_paramS48HY5/" \
--model_path "./trainer_dir_paramS48HY5/" \
--model_name "checkpoint-30000" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 0 \
--max_time_len 2048 \
--batch_size 32 \
--max_new_tokens 64 \
--do_sample 0 \
--flag "@30000"