# -*- coding: utf-8 -*-


python test.py \
--datasets_path "../../datasets/Data_lists_cleaned/" \
--datasets_name "TEST_forUnknownAttack.json" \
--output_dir "./test_dir_paramS48HY4/" \
--model_path "../../experiments_main/trainer_dir_paramS48HY4/" \
--model_name "checkpoint-30000" \
--system_prompt "" \
--template_prompt "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}" \
--is_training 0 \
--max_time_len 2048 \
--batch_size 32 \
--max_new_tokens 64 \
--do_sample 0 \
--flag "@30000forUnknownAttack"
