# Equivalent invocations of task main.py scripts using the unified cli.py runner.
# Run from repository root.

set -euo pipefail

# --- Adversarial Prompt ---
python cli.py run adversarial_prompt -- \
  --model_type openai \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --num_samples 2 \
  --verbose

python cli.py run adversarial_prompt -- \
  --model_type deepseek \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --num_samples 2 \
  --verbose

nohup python cli.py run adversarial_prompt -- \
  --model_type openai \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --restricted_reasoning \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python cli.py run adversarial_prompt -- \
  --model_type openai \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/gpt-5_hard_1205.log 2>&1 &

nohup python cli.py run adversarial_prompt -- \
  --model_type openai \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/gpt-4_hard_1203.log 2>&1 &

nohup python cli.py run adversarial_prompt -- \
  --model_type deepseek \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/deepseek-v3_hard_1205.log 2>&1 &

nohup python cli.py run adversarial_prompt -- \
  --model_type deepseek \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/deepseek-r1_hard_1205.log 2>&1 &

python cli.py run adversarial_prompt -- \
  --model_type vllm \
  --model_path "/path/to/vllm_model" \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --num_samples 3 \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --verbose

# --- Biological Sequence ---
nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/gpt_quick.json \
  --models gpt-4 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/gpt4_hard_1120.log 2>&1 &

nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/gpt_quick.json \
  --models gpt-5 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/gpt5_hard_1120.log 2>&1 &

nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/gpt_quick.json \
  --models o4-mini \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/gpt_quick.json \
  --models o4-mini \
  --load-data datasets/bio_seq_dataset_hard.json \
  --restricted-reasoning \
  --verbose \
  > logs/o4-mini-low_hard_restricted_1221.log 2>&1 &

nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/deepseek_quick.json \
  --models deepseek-v3 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/deepseek-v3_hard_1120.log 2>&1 &

nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/deepseek_quick.json \
  --models deepseek-r1 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/deepseek-r1_hard_1120.log 2>&1 &

nohup python cli.py run biological_sequence -- \
  --config configs/biological_sequence/deepseek_quick.json \
  --models deepseek-r1 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --restricted-reasoning \
  --verbose \
  > logs/deepseek-r1_hard_restricted_1220.log 2>&1 &

# --- Aligned Table ---
python cli.py run aligned_table -- \
  --model_type openai \
  --model_name o4-mini \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/medium_contexts_1203_cleaned.json \
  --format latex \
  --output ./results/medium_latex_o4-mini_results_1203.json \
  --save_responses \
  --verbose

python cli.py run aligned_table -- \
  --model_type deepseek \
  --model_name deepseek-reasoner \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/medium_contexts_1203_cleaned.json \
  --format latex \
  --output ./results/medium_latex_deepseek-r1_results_1203.json \
  --save_responses \
  --verbose

nohup python cli.py run aligned_table -- \
  --model_type openai \
  --model_name gpt-5 \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_gpt5_results_1203.json \
  --save_responses \
  --verbose \
  > logs/gpt5_hard_1203.log 2>&1 &

nohup python cli.py run aligned_table -- \
  --model_type openai \
  --model_name gpt-4 \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_gpt4_results_1203.json \
  --save_responses \
  --verbose \
  > logs/gpt4_hard_1203.log 2>&1 &

nohup python cli.py run aligned_table -- \
  --model_type openai \
  --model_name o4-mini \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_o4-mini-low_results_1220.json \
  --restricted-reasoning \
  --save_responses \
  --verbose \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python cli.py run aligned_table -- \
  --model_type deepseek \
  --model_name deepseek-reasoner \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_deepseek-r1_results_1209.json \
  --save_responses \
  --verbose \
  > logs/deepseek-r1_hard_1209.log 2>&1 &

nohup python cli.py run aligned_table -- \
  --model_type deepseek \
  --model_name deepseek-chat \
  --config configs/aligned_table/config.yaml \
  --test_file ./datasets/hard_contexts_1203_cleaned.json \
  --output ./results/hard_deepseek-v3_results.json \
  --save_responses \
  --verbose \
  > logs/deepseek-v3_hard_1203.log 2>&1 &

# --- Cipher-Decipher ---
nohup python cli.py run cipher_decipher -- \
  --test-file ./datasets/hard_dataset.json \
  --models openai \
  --config configs/cipher_decipher/gpt_config.yaml \
  --verbose \
  --restricted-reasoning \
  --save-responses \
  --response-log logs/o4-mini-low_experiment_1220.json \
  --samples 25 \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python cli.py run cipher_decipher -- \
  --test-file ./datasets/hard_dataset.json \
  --models deepseek \
  --config configs/cipher_decipher/deepseek_config.yaml \
  --verbose \
  --save-responses \
  --response-log logs/deepseek-r1_experiment_1209.json \
  --samples 25 \
  > logs/deepseek-r1_hard_1209.log 2>&1 &

# --- Context-Aware Redaction ---
python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name o4-mini \
  --config configs/context_aware_redaction/config.yaml \
  --api_key "$OPENAI_API_KEY" \
  --dataset datasets/dataset_easy_1120.json \
  --num_samples 2 \
  --verbose

python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name deepseek-chat \
  --config configs/context_aware_redaction/config.yaml \
  --api_key "$DEEPSEEK_API_KEY" \
  --base_url "https://api.deepseek.com/v1" \
  --dataset datasets/dataset_easy_1120.json \
  --num_samples 1 \
  --verbose

nohup python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name o4-mini \
  --config configs/context_aware_redaction/config.yaml \
  --api_key "$OPENAI_API_KEY" \
  --dataset datasets/dataset_hard_1120.json \
  --num_samples 100 \
  --verbose \
  --restricted-reasoning \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name o4-mini \
  --config configs/context_aware_redaction/config.yaml \
  --api_key "$OPENAI_API_KEY" \
  --dataset datasets/dataset_hard_1120.json \
  --num_samples 2 \
  --verbose \
  > logs/o4-mini-high_test.log 2>&1 &

nohup python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name gpt-4 \
  --config configs/context_aware_redaction/config.yaml \
  --api_key "$OPENAI_API_KEY" \
  --dataset datasets/dataset_hard_1120.json \
  --num_samples 100 \
  --verbose \
  > logs/gpt4_hard_1120.log 2>&1 &

nohup python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name deepseek-reasoner \
  --config configs/context_aware_redaction/config.yaml \
  --api_key "$DEEPSEEK_API_KEY" \
  --dataset datasets/dataset_hard_1120.json \
  --num_samples 100 \
  --verbose \
  > logs/deepseek-r1_hard_1210.log 2>&1 &

# --- Gomoku ---
python cli.py run gomoku -- \
  --models gpt-4 \
  --test-counts 10 \
  --board-sizes 9 \
  --verbose \
  --test-file datasets/hard_test_data_15x15.json

nohup python cli.py run gomoku -- \
  --models o4-mini \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_diagonal_test_data_h0v0dd50du50_d70-80_15x15_100.json \
  --save-response outputs/o4-mini_hard_dense_diagonal-responses_1211.json \
  > logs/o4-mini_hard_dense_diagonal_1211.log 2>&1 &

nohup python cli.py run gomoku -- \
  --models o4-mini \
  --test-counts 100 \
  --verbose \
  --restricted-reasoning \
  --test-file datasets/hard_dense_diagonal_test_data_h0v0dd50du50_d70-80_15x15_100.json \
  --save-response outputs/o4-mini-low_hard_dense_diagonal-responses_1220.json \
  > logs/o4-mini-low_hard_dense_diagonal_1220.log 2>&1 &

nohup python cli.py run gomoku -- \
  --models gpt-5 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_diagonal_test_data_h0v0dd50du50_d70-80_15x15_100.json \
  --save-response outputs/gpt5_hard_dense_diagonal-responses_1211.json \
  > logs/gpt5_hard_dense_diagonal_1211.log 2>&1 &

nohup python cli.py run gomoku -- \
  --models gpt-4 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/gpt4_hard_dense_linear-responses_1210.json \
  > logs/gpt4_hard_dense_linear_1210.log 2>&1 &

nohup python cli.py run gomoku -- \
  --models o4-mini \
  --test-counts 100 \
  --verbose \
  --restricted-reasoning \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/o4-mini-low_hard_dense_linear-responses_1220.json \
  > logs/o4-mini-low_hard_dense_linear_1220.log 2>&1 &

nohup python cli.py run gomoku -- \
  --models deepseek-r1 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/deepseek-r1_hard_dense_linear-responses_1210.json \
  > logs/deepseek-r1_hard_dense_linear_1210.log 2>&1 &

nohup python cli.py run gomoku -- \
  --models deepseek-v3 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/deepseek-v3_hard_dense_linear-responses_1210.json \
  > logs/deepseek-v3_hard_dense_linear_1210.log 2>&1 &

# --- Tree ---
python cli.py run tree -- \
  --model_type deepseek \
  --model_name deepseek-reasoner \
  --task_type task2 \
  --verbose \
  --api_key "$DEEPSEEK_API_KEY" \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 2

nohup python cli.py run tree -- \
  --model_type openai \
  --model_name gpt-5 \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  > logs/gpt-5_task2_hard_1120.log 2>&1 &

nohup python cli.py run tree -- \
  --model_type openai \
  --model_name gpt-4 \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  > logs/gpt-4_task2_hard_1120_2.log 2>&1 &

nohup python cli.py run tree -- \
  --model_type openai \
  --model_name o4-mini \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  > logs/o4-mini-high_task2_hard_1120.log 2>&1 &

nohup python cli.py run tree -- \
  --model_type openai \
  --model_name o4-mini \
  --task_type task1 \
  --verbose \
  --restricted_reasoning \
  --test_file ./datasets/tree_task1_hard_1119.json \
  --num_samples 100 \
  > logs/o4-mini-low_task1_hard_1220.log 2>&1 &

nohup python cli.py run tree -- \
  --model_type deepseek \
  --model_name deepseek-chat \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  --api_key "$DEEPSEEK_API_KEY" \
  > logs/deepseek-v3_task2_hard_1120.log 2>&1 &

nohup python cli.py run tree -- \
  --model_type deepseek \
  --model_name deepseek-reasoner \
  --task_type task1 \
  --verbose \
  --test_file ./datasets/tree_task1_hard_1119.json \
  --num_samples 100 \
  --api_key "$DEEPSEEK_API_KEY" \
  > logs/deepseek-r1_task1_hard_1211.log 2>&1 &

# --- Typewriter ---
nohup python cli.py run typewriter -- \
  --models gpt-5 \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --num-samples 100 \
  > logs/gpt-5_hard_1118.log 2>&1 &

nohup python cli.py run typewriter -- \
  --models o4-mini \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --num-samples 100 \
  > logs/o4-mini-high_hard_1118.log 2>&1 &

nohup python cli.py run typewriter -- \
  --models o4-mini \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --restricted-reasoning \
  --num-samples 100 \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python cli.py run typewriter -- \
  --models deepseek-v3 \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --num-samples 100 \
  > logs/deepseek-v3_hard_1118.log 2>&1 &

# --- RSA RandomArt ---
python cli.py run rsa_randomart -- \
  --generate-only \
  --num-samples 100 \
  --num-differences 9 \
  --pattern-width 40 \
  --pattern-height 23 \
  --key-size 2048 \
  --output-data datasets/difficult_9_difference.json

python cli.py run rsa_randomart -- \
  --model-type openai \
  --model-name gpt-4 \
  --num-samples 100 \
  --input-data datasets/difficult_7_difference.json \
  --save-response logs/gpt4_difficult_7_difference.log \
  --verbose \
  --output-results outputs/gpt4_results_difficult_7_difference.json

nohup python cli.py run rsa_randomart -- \
  --model-type openai \
  --model-name gpt-5 \
  --num-samples 100 \
  --input-data datasets/difficult_9_difference.json \
  --save-response logs/gpt-5_difficult_9_difference_1208.json \
  --output-results outputs/gpt-5_results_difficult_9_difference_1208.json \
  --verbose \
  > logs/gpt-5_results_difficult_9_difference_1208.log 2>&1 &

nohup python cli.py run rsa_randomart -- \
  --model-type openai \
  --model-name gpt-4 \
  --num-samples 100 \
  --input-data datasets/difficult_3_difference.json \
  --save-response logs/gpt-4_difficult_3_difference_1208.json \
  --output-results outputs/gpt-4_results_difficult_3_difference_1208.json \
  --verbose \
  > logs/gpt-4_results_difficult_3_difference_1208.log 2>&1 &

python cli.py run rsa_randomart -- \
  --model-type deepseek \
  --model-name deepseek-chat \
  --num-samples 100 \
  --input-data datasets/difficult_5_difference.json \
  --save-response logs/deepseek-v3_difficult_5_difference.json \
  --output-results outputs/deepseek-v3_results_difficult_5_difference.json

nohup python cli.py run rsa_randomart -- \
  --model-type deepseek \
  --model-name deepseek-reasoner \
  --num-samples 100 \
  --input-data datasets/difficult_7_difference.json \
  --save-response logs/deepseek-r1_difficult_7_difference_1208.json \
  --output-results outputs/deepseek-r1_results_difficult_7_difference_1208.json \
  --verbose \
  > logs/deepseek-r1_results_difficult_7_difference_1208.log 2>&1 &

python cli.py run rsa_randomart -- \
  --model-type deepseek \
  --model-name deepseek-reasoner \
  --num-samples 1 \
  --input-data datasets/difficult_7_difference.json \
  --save-response logs/deepseek-r1_difficult_7_difference_test.json \
  --output-results outputs/deepseek-r1_results_difficult_7_difference_test.json \
  --verbose

nohup python cli.py run rsa_randomart -- \
  --model-type deepseek \
  --model-name deepseek-chat \
  --num-samples 100 \
  --input-data datasets/difficult_9_difference.json \
  --save-response logs/deepseek-v3_difficult_9_difference_1208.json \
  --output-results outputs/deepseek-v3_results_difficult_9_difference_1208.json \
  --verbose \
  > logs/deepseek-v3_results_difficult_9_difference_1208.log 2>&1 &

nohup python cli.py run rsa_randomart -- \
  --model-type openai \
  --model-name o4-mini \
  --num-samples 100 \
  --input-data datasets/difficult_9_difference.json \
  --save-response logs/o4-mini_difficult_9_difference_1208.json \
  --output-results outputs/o4-mini_results_difficult_9_difference_1208.json \
  --verbose \
  > logs/o4-mini_results_difficult_9_difference_1208.log 2>&1 &

nohup python cli.py run rsa_randomart -- \
  --model-type openai \
  --model-name o4-mini \
  --num-samples 100 \
  --input-data datasets/difficult_7_difference.json \
  --save-response logs/o4-mini-low_difficult_7_difference_1220.json \
  --output-results outputs/o4-mini-low_results_difficult_7_difference_1220.json \
  --verbose \
  --restricted-reasoning \
  --reasoning-effort low \
  > logs/o4-mini-low_results_difficult_7_difference_1220.log 2>&1 &

# --- Map Navigation ---
python cli.py run map_navigation -- \
  --model deepseek-v3 \
  --data datasets/sokoban_test.json \
  --output results/deepseek-v3_sokoban_test.json \
  --verbose \
  --save-response results/deepseek-v3_sokoban_detailed_test.json

python cli.py run map_navigation -- \
  --model deepseek-r1 \
  --data datasets/sokoban_test.json \
  --output results/deepseek-r1_sokoban_test.json \
  --verbose \
  --save-response results/deepseek-r1_sokoban_detailed_test.json

python cli.py run map_navigation -- \
  --model o4-mini \
  --data datasets/sokoban_test.json \
  --output results/o4-mini_sokoban_test.json \
  --verbose \
  --save-response results/o4-mini_sokoban_detailed_test.json

nohup python cli.py run map_navigation -- \
  --model deepseek-v3 \
  --data datasets/sokoban_12x12.json \
  --output results/deepseek-v3_sokoban_12x12_1220.json \
  --verbose \
  --save-response results/deepseek-v3_sokoban_detailed_12x12_1220.json \
  > logs/deepseek-v3_sokoban_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model deepseek-r1 \
  --data datasets/sokoban_12x12.json \
  --output results/deepseek-r1_sokoban_12x12_1220.json \
  --verbose \
  --save-response results/deepseek-r1_sokoban_detailed_12x12_1220.json \
  > logs/deepseek-r1_sokoban_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model o4-mini \
  --data datasets/sokoban_12x12.json \
  --output results/o4-mini-high_sokoban_12x12_1220.json \
  --verbose \
  --save-response results/o4-mini-high_sokoban_detailed_12x12_1220.json \
  > logs/o4-mini-high_sokoban_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model o4-mini \
  --data datasets/sokoban_12x12.json \
  --output results/o4-mini-low_sokoban_12x12_1221.json \
  --verbose \
  --restricted-reasoning \
  --save-response results/o4-mini-low_sokoban_detailed_12x12_1221.json \
  > logs/o4-mini-low_sokoban_1221.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model gpt-4 \
  --data datasets/sokoban_12x12.json \
  --output results/gpt-4_sokoban_12x12_1220.json \
  --verbose \
  --save-response results/gpt-4_sokoban_detailed_12x12_1220.json \
  > logs/gpt-4_sokoban_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model gpt-5 \
  --data datasets/sokoban_12x12.json \
  --output results/gpt-5_sokoban_12x12_1220.json \
  --verbose \
  --save-response results/gpt-5_sokoban_detailed_12x12_1220.json \
  > logs/gpt-5_sokoban_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model deepseek-v3 \
  --data datasets/frozenlake_12x12.json \
  --output results/deepseek-v3_frozenlake_12x12_1220.json \
  --verbose \
  --save-response results/deepseek-v3_frozenlake_detailed_12x12_1220.json \
  > logs/deepseek-v3_frozenlake_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model deepseek-r1 \
  --data datasets/frozenlake_12x12.json \
  --output results/deepseek-r1_frozenlake_12x12_1220.json \
  --verbose \
  --save-response results/deepseek-r1_frozenlake_detailed_12x12_1220.json \
  > logs/deepseek-r1_frozenlake_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model o4-mini \
  --data datasets/frozenlake_12x12.json \
  --output results/o4-mini-high_frozenlake_12x12_1220.json \
  --verbose \
  --save-response results/o4-mini-high_frozenlake_detailed_12x12_1220.json \
  > logs/o4-mini-high_frozenlake_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model o4-mini \
  --data datasets/frozenlake_12x12.json \
  --output results/o4-mini-low_frozenlake_12x12_1221.json \
  --verbose \
  --restricted-reasoning \
  --save-response results/o4-mini-low_frozenlake_detailed_12x12_1221.json \
  > logs/o4-mini-low_frozenlake_1221.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model gpt-4 \
  --data datasets/frozenlake_12x12.json \
  --output results/gpt-4_frozenlake_12x12_1220.json \
  --verbose \
  --save-response results/gpt-4_frozenlake_detailed_12x12_1220.json \
  > logs/gpt-4_frozenlake_1220.log 2>&1 &

nohup python cli.py run map_navigation -- \
  --model gpt-5 \
  --data datasets/frozenlake_12x12.json \
  --output results/gpt-5_frozenlake_12x12_1220.json \
  --verbose \
  --save-response results/gpt-5_frozenlake_detailed_12x12_1220.json \
  > logs/gpt-5_frozenlake_1220.log 2>&1 &
