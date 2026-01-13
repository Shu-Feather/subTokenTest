#################################
# Experiment Logs
# Run in the root directory 
#################################

cd /gemini/code/tokenbench/tasks/adversarial_prompt

nohup python main.py \
  --model_type vllm \
  --model_path /gemini/space/pretrained_models/Qwen2.5-32B-Instruct\
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --num_samples 100 \
  --verbose \
  > logs/Qwen2.5-32B-Instruct_hard_1216.log 2>&1 &

nohup python main.py \
  --model_type vllm \
  --model_path /gemini/space/pretrained_models/DeepSeek-R1-Distill-Qwen-32B\
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --num_samples 100 \
  --verbose \
  > logs/DeepSeek-R1-Distill-Qwen-32B_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/biological_sequence

nohup python main.py \
  --config config/local_qwen_model.json \
  --models qwen-32b \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/qwen-32b_hard_1216.log 2>&1 &

nohup python main.py \
  --config config/local_qwen_model.json \
  --models r1-distill-qwen-32b \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/r1-distill-qwen-32b_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/aligned_table

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/Qwen2.5-32B-Instruct \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_Qwen2.5-32B-Instruct_1216.json \
  --save_responses \
  --verbose \
  > logs/Qwen2.5-32B-Instruct_hard_1216.log 2>&1 &

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_DeepSeek-R1-Distill-Qwen-32B_1216.json \
  --save_responses \
  --verbose \
  > logs/DeepSeek-R1-Distill-Qwen-32B_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/cipher_decipher

nohup python main.py \
  --config config/vllm_qwen32_config.yaml \
  --models qwen2.5-32b \
  --test-file datasets/hard_dataset.json \
  --samples 100 \
  --verbose \
  > logs/qwen2.5-32b_hard_1216.log 2>&1 &

nohup python main.py \
  --config config/vllm_qwen32_config.yaml \
  --models deepseek-r1-distill-qwen-32b \
  --test-file datasets/hard_dataset.json \
  --samples 100 \
  --verbose \
  > logs/deepseek-r1-distill-qwen-32b_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/context_aware_redaction

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/Qwen2.5-32B-Instruct \
  --dataset data/generated/dataset_hard_1120.json \
  --num_samples 100 \
  --verbose \
  > logs/Qwen2.5-32B-Instruct_hard_1216.log 2>&1 &

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --dataset data/generated/dataset_hard_1120.json \
  --num_samples 100 \
  --verbose \
  > logs/DeepSeek-R1-Distill-Qwen-32B_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/gomoku

nohup python main.py \
  --models qwen2.5-32b \
  --board-sizes 15 \
  --test-counts 100 \
  --test-file datasets/hard_test_data_15x15.json \
  --output-dir results \
  --save-response logs/qwen2.5-32b_responses_1216.json \
  --verbose \
  > logs/qwen2.5-32b_hard_1216.log 2>&1 &

nohup python main.py \
  --models deepseek-r1-distill-qwen-32b \
  --board-sizes 15 \
  --test-counts 100 \
  --test-file datasets/hard_test_data_15x15.json \
  --output-dir results \
  --save-response logs/deepseek-r1-distill-qwen-32b_responses_1216.json \
  --verbose \
  > logs/deepseek-r1-distill-qwen-32b_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/rsa_randomart

nohup python main.py \
  --model-type vllm \
  --model-path /gemini/space/pretrained_models/Qwen2.5-32B-Instruct \
  --input-data data/difficult_7_difference.json \
  --num-samples 100 \
  --output-results outputs/Qwen2.5-32B-Instruct_results_1216.json \
  --save-response outputs/Qwen2.5-32B-Instruct_responses_1216.json \
  --verbose \
  > logs/Qwen2.5-32B-Instruct_hard_1216.log 2>&1 &

nohup python main.py \
  --model-type vllm \
  --model-path /gemini/space/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --input-data data/difficult_7_difference.json \
  --num-samples 100 \
  --output-results outputs/DeepSeek-R1-Distill-Qwen-32B_results_1216.json \
  --save-response outputs/DeepSeek-R1-Distill-Qwen-32B_responses_1216.json \
  --verbose \
  > logs/DeepSeek-R1-Distill-Qwen-32B_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/tree

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/Qwen2.5-32B-Instruct \
  --task_type task1 \
  --num_samples 100 \
  --max_depth 4 \
  --tensor_parallel_size 2 \
  --test_file datasets/tree_task1_hard_1119.json \
  --verbose \
  > logs/Qwen2.5-32B-Instruct_task1_hard_1216.log 2>&1 &

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/Qwen2.5-32B-Instruct \
  --task_type task2 \
  --num_samples 100 \
  --max_depth 4 \
  --tensor_parallel_size 2 \
  --test_file datasets/tree_task2_hard_1119.json \
  --verbose \
  > logs/Qwen2.5-32B-Instruct_task2_hard_1216.log 2>&1 &

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --task_type task1 \
  --num_samples 100 \
  --max_depth 4 \
  --tensor_parallel_size 2 \
  --test_file datasets/tree_task1_hard_1119.json \
  --verbose \
  > logs/DeepSeek-R1-Distill-Qwen-32B_task1_hard_1216.log 2>&1 &

nohup python main.py \
  --model_type vllm \
  --model_name /gemini/space/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --task_type task2 \
  --num_samples 100 \
  --max_depth 4 \
  --tensor_parallel_size 2 \
  --test_file datasets/tree_task2_hard_1119.json \
  --verbose \
  > logs/DeepSeek-R1-Distill-Qwen-32B_task2_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/tasks/typewriter

nohup python main.py \
  --models qwen2.5-32b \
  --test-file datasets/test_cases_hard.json \
  --num-samples 100 \
  --prompt-types system few_shot \
  --use-vllm \
  --verbose \
  > logs/qwen2.5-32b_hard_1216.log 2>&1 &

nohup python main.py \
  --models deepseek-r1-distill-qwen-32b \
  --test-file datasets/test_cases_hard.json \
  --num-samples 100 \
  --prompt-types system few_shot \
  --use-vllm \
  --verbose \
  > logs/deepseek-r1-distill-qwen-32b_hard_1216.log 2>&1 &


cd /gemini/code/tokenbench/wordle

nohup python run_model_test.py \
  --config local_qwen32 \
  --output-dir results/local_qwen32 \
  --transcript-dir results/transcripts/local_qwen32 \
  --verbose \
  > logs/local_qwen32_1216.log 2>&1 &
