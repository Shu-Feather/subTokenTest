# Setting up environment

export OPENAI_API_KEY=
export DEEPSEEK_API_KEY=

export CUDA_VISIBLE_DEVICES=

#################################
# Experiment Logs
# Run in the root directory
#################################
# Tasks now live under tasks/<name>; cd into the desired task directory before running the commands below.

# Adversarial Contexts

python generate_contexts.py \
  --samples_per_difficulty 20 \
  --output datasets/hard_contexts.json \
  --difficulty_level hard \
  --cost

python generate_contexts.py \
  --samples_per_difficulty 10 \
  --output datasets/medium_contexts.json \
  --difficulty_level medium \
  --cost

python generate_contexts.py \
  --samples_per_difficulty 10 \
  --output datasets/easy_contexts.json \
  --difficulty_level easy \
  --cost

python main.py \
  --model_type openai \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --num_samples 2 \
  --verbose

python main.py \
  --model_type deepseek \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --num_samples 2 \
  --verbose

nohup python main.py \
  --model_type openai \
  --use_generated_contexts \
  --restricted_reasoning \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python main.py \
  --model_type openai \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/gpt-5_hard_1205.log 2>&1 &

nohup python main.py \
  --model_type openai \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/gpt-4_hard_1203.log 2>&1 &

nohup python main.py \
  --model_type deepseek \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/deepseek-v3_hard_1205.log 2>&1 &

nohup python main.py \
  --model_type deepseek \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts.json \
  --num_samples 100 \
  --verbose \
  > logs/deepseek-r1_hard_1205.log 2>&1 &

python main.py \
  --model_type vllm \
  --model_path "/data/mys/hf_models/Meta-Llama-3.1-8B-Instruct" \
  --num_samples 3 \
  --use_generated_contexts \
  --contexts_file datasets/hard_contexts_sample.json \
  --verbose

python scripts/analysis_results.py \
  --results outputs/deepseek-chat_20251102_172937/results.json

# Biological Sequences

python generate_datasets.py --num-cases 25 --min-length 50 --max-length 50 --output-dir ./datasets 


nohup python main.py \
  --config config/gpt_quick.json \
  --models gpt-4 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/gpt4_hard_1120.log 2>&1 &

nohup python main.py \
  --config config/gpt_quick.json \
  --models gpt-5 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/gpt5_hard_1120.log 2>&1 &

nohup python main.py \
  --config config/gpt_quick.json \
  --models o4-mini \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python main.py \
  --config config/gpt_quick.json \
  --models o4-mini \
  --load-data datasets/bio_seq_dataset_hard.json \
  --restricted-reasoning \
  --verbose \
  > logs/o4-mini-low_hard_restricted_1221.log 2>&1 &

nohup python main.py \
  --config config/gpt_quick.json \
  --models o4-mini \
  --load-data datasets/bio_seq_dataset_hard.json \
  --restricted-reasoning \
  --verbose \
  > logs/o4-mini-high_hard_restricted_1221.log 2>&1 &

nohup python main.py \
  --config config/deepseek_quick.json \
  --models deepseek-v3 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/deepseek-v3_hard_1120.log 2>&1 &

nohup python main.py \
  --config config/deepseek_quick.json \
  --models deepseek-r1 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --verbose \
  > logs/deepseek-r1_hard_1120.log 2>&1 &

nohup python main.py \
  --config config/deepseek_quick.json \
  --models deepseek-r1 \
  --load-data datasets/bio_seq_dataset_hard.json \
  --restricted-reasoning \
  --verbose \
  > logs/deepseek-r1_hard_restricted_1220.log 2>&1 &

# Aligned Table

python main.py \
    --model_type openai \
    --model_name o4-mini \
    --test_file ./datasets/medium_contexts_1203_cleaned.json \
    --format latex \
    --output ./results/medium_latex_o4-mini_results_1203.json \
    --save_responses \
    --verbose

python main.py \
    --model_type deepseek \
    --model_name deepseek-reasoner \
    --test_file ./datasets/medium_contexts_1203_cleaned.json \
    --format latex \
    --output ./results/medium_latex_deepseek-r1_results_1203.json \
    --save_responses \
    --verbose

nohup python main.py \
    --model_type openai \
    --model_name gpt-5 \
    --test_file ./datasets/hard_contexts_1203_cleaned.json \
    --output ./results/hard_gpt5_results_1203.json \
    --save_responses \
    --verbose \
    > logs/gpt5_hard_1203.log 2>&1 &
  
nohup python main.py \
    --model_type openai \
    --model_name gpt-4 \
    --test_file ./datasets/hard_contexts_1203_cleaned.json \
    --output ./results/hard_gpt4_results_1203.json \
    --save_responses \
    --verbose \
    > logs/gpt4_hard_1203.log 2>&1 &

nohup python main.py \
    --model_type openai \
    --model_name o4-mini \
    --test_file ./datasets/hard_contexts_1203_cleaned.json \
    --output ./results/hard_o4-mini-low_results_1220.json \
    --restricted-reasoning \
    --save_responses \
    --verbose \
    > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python main.py \
    --model_type deepseek \
    --model_name deepseek-reasoner \
    --test_file ./datasets/hard_contexts_1203_cleaned.json \
    --output ./results/hard_deepseek-r1_results_1209.json \
    --save_responses \
    --verbose \
    > logs/deepseek-r1_hard_1209.log 2>&1 &

nohup python main.py \
    --model_type deepseek \
    --model_name deepseek-chat \
    --test_file ./datasets/hard_contexts_1203_cleaned.json \
    --output ./results/hard_deepseek-v3_results.json \
    --save_responses \
    --verbose \
    > logs/deepseek-v3_hard_1203.log 2>&1 &

python scripts/analysis_scores.py \
  ./results/hard_gpt4_results.json \
  --output ./results/hard_gpt4_analysis_output

python scripts/generate_contexts.py \
    --model gpt-4 \
    --num_contexts 25 \
    --output ./datasets/simple_contexts_1203.json \
    --min_rows 3 \
    --max_rows 3 \
    --min_cols 3 \
    --max_cols 3

python scripts/generate_contexts.py \
    --model gpt-4 \
    --num_contexts 25 \
    --output ./datasets/medium_contexts_1203.json \
    --min_rows 5 \
    --max_rows 5 \
    --min_cols 5 \
    --max_cols 5

python scripts/generate_contexts.py \
    --model gpt-4 \
    --num_contexts 100 \
    --output ./datasets/hard_contexts_1203.json \
    --min_rows 8 \
    --max_rows 8 \
    --min_cols 8 \
    --max_cols 8


# Cipher-Decipher

python generate_dataset.py \
  --samples 3 \
  --batch-size 1 \
  --output ./datasets/sample_dataset.json

python generate_dataset.py \
  --samples 25 \
  --batch-size 5 \
  --output ./datasets/easy_dataset.json \
  --difficulties easy

python generate_dataset.py \
  --samples 25 \
  --batch-size 5 \
  --output ./datasets/medium_dataset.json \
  --difficulties medium

python generate_dataset.py \
  --samples 50 \
  --batch-size 5 \
  --output ./datasets/hard_dataset.json \
  --difficulties hard

nohup python main.py \
  --test-file ./datasets/hard_dataset.json \
  --models openai \
  --config config/gpt_config.yaml \
  --verbose \
  --restricted-reasoning \
  --save-responses \
  --response-log logs/o4-mini-low_experiment_1220.json \
  --samples 25 \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python main.py \
  --test-file ./datasets/hard_dataset.json \
  --models deepseek \
  --config config/deepseek_config.yaml \
  --verbose \
  --save-responses \
  --response-log logs/deepseek-r1_experiment_1209.json \
  --samples 25 \
    > logs/deepseek-r1_hard_1209.log 2>&1 &

python scripts/analyze_scores.py \
  logs/gpt-4_experiment.json \
  --output logs/gpt-4_analysis_results \
  -a

# context aware redaction
python generate_data.py \
    --num_samples 50 \
    --difficulty easy \
    --output dataset_easy_1120.json

python generate_data.py \
    --num_samples 50 \
    --difficulty medium \
    --output dataset_medium_1120.json

python generate_data.py \
    --num_samples 100 \
    --difficulty hard \
    --output dataset_hard_1120.json

python main.py \
    --model_type api \
    --model_name o4-mini \
    --api_key $OPENAI_API_KEY \
    --dataset data/generated/dataset_easy_1120.json \
    --num_samples 2 \
    --verbose

python main.py \
    --model_type api \
    --model_name deepseek-chat \
    --api_key $DEEPSEEK_API_KEY \
    --base_url "https://api.deepseek.com/v1" \
    --dataset data/generated/dataset_easy_1120.json \
    --num_samples 1 \
    --verbose

nohup python main.py \
    --model_type api \
    --model_name o4-mini \
    --api_key $OPENAI_API_KEY \
    --dataset data/generated/dataset_hard_1120.json \
    --num_samples 100 \
    --verbose \
    --restricted-reasoning \
    > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python main.py \
    --model_type api \
    --model_name o4-mini \
    --api_key $OPENAI_API_KEY \
    --dataset data/generated/dataset_hard_1120.json \
    --num_samples 2 \
    --verbose \
    > logs/o4-mini-high_test.log 2>&1 &
  
nohup python main.py \
    --model_type api \
    --model_name gpt-4 \
    --api_key $OPENAI_API_KEY \
    --dataset data/generated/dataset_hard_1120.json \
    --num_samples 100 \
    --verbose \
    > logs/gpt4_hard_1120.log 2>&1 &

nohup python main.py \
    --model_type api \
    --model_name deepseek-reasoner \
    --api_key $DEEPSEEK_API_KEY \
    --dataset data/generated/dataset_hard_1120.json \
    --num_samples 100 \
    --verbose \
    > logs/deepseek-r1_hard_1210.log 2>&1 &

# Gomoku Benchmark
python main.py \
  --models gpt-4 \
  --test-counts 10 \
  --board-sizes 9 \
  --verbose \

python generate_datasets.py \
  --board-sizes 15 \
  --test-counts 100 \
  --horizontal 0 \
  --vertical 0 \
  --diagonal-down 0.5 \
  --diagonal-up 0.5 \
  --min-density 0.7 \
  --max-density 0.8 \
  --name hard_dense_diagonal \
  --with-metadata \
  --seed 42

python generate_datasets.py \
  --board-sizes 15 \
  --test-counts 100 \
  --horizontal 0.5 \
  --vertical 0.5 \
  --diagonal-down 0 \
  --diagonal-up 0 \
  --min-density 0.7 \
  --max-density 0.8 \
  --name hard_dense_linear \
  --with-metadata \
  --seed 42

python generate_datasets.py \
  --verify-only \
  --dataset-path datasets/hard_dense_diagonal_*.json \
  --verbose

python main.py \
  --models gpt-4 \
  --test-counts 10 \
  --verbose \
  --test-file datasets/hard_test_data_15x15.json

nohup python main.py \
  --models o4-mini \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_diagonal_test_data_h0v0dd50du50_d70-80_15x15_100.json \
  --save-response outputs/o4-mini_hard_dense_diagonal-responses_1211.json \
  > logs/o4-mini_hard_dense_diagonal_1211.log 2>&1 &

nohup python main.py \
  --models o4-mini \
  --test-counts 100 \
  --verbose \
  --restricted-reasoning \
  --test-file datasets/hard_dense_diagonal_test_data_h0v0dd50du50_d70-80_15x15_100.json \
  --save-response outputs/o4-mini-low_hard_dense_diagonal-responses_1220.json \
  > logs/o4-mini-low_hard_dense_diagonal_1220.log 2>&1 &

nohup python main.py \
  --models gpt-5 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_diagonal_test_data_h0v0dd50du50_d70-80_15x15_100.json \
  --save-response outputs/gpt5_hard_dense_diagonal-responses_1211.json \
  > logs/gpt5_hard_dense_diagonal_1211.log 2>&1 &

nohup python main.py \
  --models gpt-4 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/gpt4_hard_dense_linear-responses_1210.json \
  > logs/gpt4_hard_dense_linear_1210.log 2>&1 &

nohup python main.py \
  --models o4-mini \
  --test-counts 100 \
  --verbose \
  --restricted-reasoning \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/o4-mini-low_hard_dense_linear-responses_1220.json \
  > logs/o4-mini-low_hard_dense_linear_1220.log 2>&1 &

nohup python main.py \
  --models deepseek-r1 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/deepseek-r1_hard_dense_linear-responses_1210.json \
  > logs/deepseek-r1_hard_dense_linear_1210.log 2>&1 &

nohup python main.py \
  --models deepseek-v3 \
  --test-counts 100 \
  --verbose \
  --test-file datasets/hard_dense_linear_test_data_h50v50dd0du0_d70-80_15x15_100.json \
  --save-response outputs/deepseek-v3_hard_dense_linear-responses_1210.json \
  > logs/deepseek-v3_hard_dense_linear_1210.log 2>&1 &

python scripts/results_analysis.py \
    --results_dir ./results \
    --output_dir analysis_output \
    --plots_subdir figures


# Tree Benchmark

python generate_tasks1.py \
  --difficulty easy \
  --num_samples 100 \
  --output_dir ./datasets \
  --output_file tree_task1_easy_1119.json \
  --verbose

python generate_tasks1.py \
  --difficulty medium \
  --num_samples 100 \
  --output_dir ./datasets \
  --output_file tree_task1_medium_1119.json \
  --verbose

python generate_tasks1.py \
  --difficulty hard \
  --num_samples 100 \
  --output_dir ./datasets \
  --output_file tree_task1_hard_1119.json \
  --verbose

python generate_tasks2.py \
  --difficulty easy \
  --num_samples 100 \
  --output_dir ./datasets \
  --output_file tree_task2_easy_1119.json \
  --verbose \
  --threshold 2

python generate_tasks2.py \
  --difficulty medium \
  --num_samples 100 \
  --output_dir ./datasets \
  --output_file tree_task2_medium_1119.json \
  --verbose

python generate_tasks2.py \
  --difficulty hard \
  --num_samples 100 \
  --output_dir ./datasets \
  --output_file tree_task2_hard_1119.json \
  --verbose \
  --threshold 4


python main.py \
  --model_type deepseek \
  --model_name deepseek-reasoner \
  --task_type task2 \
  --verbose \
  --api_key $DEEPSEEK_API_KEY \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 2

nohup python main.py \
  --model_type openai \
  --model_name gpt-5 \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  > logs/gpt-5_task2_hard_1120.log 2>&1 &

nohup python main.py \
  --model_type openai \
  --model_name gpt-4 \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  > logs/gpt-4_task2_hard_1120_2.log 2>&1 &

nohup python main.py \
  --model_type openai \
  --model_name o4-mini \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  > logs/o4-mini-high_task2_hard_1120.log 2>&1 &

nohup python main.py \
  --model_type openai \
  --model_name o4-mini \
  --task_type task1 \
  --verbose \
  --restricted_reasoning \
  --test_file ./datasets/tree_task1_hard_1119.json \
  --num_samples 100 \
  > logs/o4-mini-low_task1_hard_1220.log 2>&1 &

nohup python main.py \
  --model_type deepseek \
  --model_name deepseek-chat \
  --task_type task2 \
  --verbose \
  --test_file ./datasets/tree_task2_hard_1119.json \
  --num_samples 100 \
  --api_key $DEEPSEEK_API_KEY \
  > logs/deepseek-v3_task2_hard_1120.log 2>&1 &

nohup python main.py \
  --model_type deepseek \
  --model_name deepseek-reasoner \
  --task_type task1 \
  --verbose \
  --test_file ./datasets/tree_task1_hard_1119.json \
  --num_samples 100 \
  --api_key $DEEPSEEK_API_KEY \
  > logs/deepseek-r1_task1_hard_1211.log 2>&1 &

# Typewriter Benchmark

python scripts/generate_datasets.py --samples 100 --seed 42

nohup python main.py \
  --models gpt-5 \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --num-samples 100 \
  > logs/gpt-5_hard_1118.log 2>&1 &

nohup python main.py \
  --models o4-mini \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --num-samples 100 \
  > logs/o4-mini-high_hard_1118.log 2>&1 &

nohup python main.py \
  --models o4-mini \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --restricted-reasoning \
  --num-samples 100 \
  > logs/o4-mini-low_hard_1220.log 2>&1 &

nohup python main.py \
  --models deepseek-v3 \
  --difficulty hard \
  --test-file ./datasets/test_cases_hard.json \
  --verbose \
  --num-samples 100 \
  > logs/deepseek-v3_hard_1118.log 2>&1 &

# Wordle Benchmark

nohup python run_model_test.py \
  --config gpt_quick \
  --verbose \
  --output-dir results/gpt5_llm_io \
  > logs/gpt5_5_letters_1203.log 2>&1 &

nohup python run_model_test.py \
  --config deepseek_quick \
  --verbose \
  --output-dir results/deepseek_llm_io \
  > logs/deepseek-v3_5_letters_1203.log 2>&1 &


# rsa benchmark

python main.py \
    --generate-only \
    --num-samples 100 \
    --num-differences 9 \
    --pattern-width 40 \
    --pattern-height 23 \
    --key-size 2048 \
    --output-data data/difficult_9_difference.json

python main.py \
    --model-type openai \
    --model-name gpt-4 \
    --num-samples 100 \
    --input-data data/difficult_7_difference.json \
    --save-response logs/gpt4_difficult_7_difference.log \
    --verbose \
    --output-results outputs/gpt4_results_difficult_7_difference.json

nohup python main.py \
    --model-type openai \
    --model-name gpt-5 \
    --num-samples 100 \
    --input-data data/difficult_9_difference.json \
    --save-response logs/gpt-5_difficult_9_difference_1208.json \
    --output-results outputs/gpt-5_results_difficult_9_difference_1208.json \
    --verbose \
    > logs/gpt-5_results_difficult_9_difference_1208.log 2>&1 &

nohup python main.py \
    --model-type openai \
    --model-name gpt-4 \
    --num-samples 100 \
    --input-data data/difficult_3_difference.json \
    --save-response logs/gpt-4_difficult_3_difference_1208.json \
    --output-results outputs/gpt-4_results_difficult_3_difference_1208.json \
    --verbose \
    > logs/gpt-4_results_difficult_3_difference_1208.log 2>&1 &


python main.py \
    --model-type deepseek \
    --model-name deepseek-chat \
    --num-samples 100 \
    --input-data data/difficult_5_difference.json \
    --save-response logs/deepseek-v3_difficult_5_difference.json \
    --output-results outputs/deepseek-v3_results_difficult_5_difference.json


nohup python main.py \
    --model-type deepseek \
    --model-name deepseek-reasoner \
    --num-samples 100 \
    --input-data data/difficult_7_difference.json \
    --save-response logs/deepseek-r1_difficult_7_difference_1208.json \
    --output-results outputs/deepseek-r1_results_difficult_7_difference_1208.json \
    --verbose \
    > logs/deepseek-r1_results_difficult_7_difference_1208.log 2>&1 &

nohup python main.py \
    --model-type deepseek \
    --model-name deepseek-chat \
    --num-samples 100 \
    --input-data data/difficult_9_difference.json \
    --save-response logs/deepseek-v3_difficult_9_difference_1208.json \
    --output-results outputs/deepseek-v3_results_difficult_9_difference_1208.json \
    --verbose \
    > logs/deepseek-v3_results_difficult_9_difference_1208.log 2>&1 &


nohup python main.py \
    --model-type openai \
    --model-name o4-mini \
    --num-samples 100 \
    --input-data data/difficult_9_difference.json \
    --save-response logs/o4-mini_difficult_9_difference_1208.json \
    --output-results outputs/o4-mini_results_difficult_9_difference_1208.json \
    --verbose \
    > logs/o4-mini_results_difficult_9_difference_1208.log 2>&1 &

nohup python main.py \
    --model-type openai \
    --model-name o4-mini \
    --num-samples 100 \
    --input-data data/difficult_7_difference.json \
    --save-response logs/o4-mini-low_difficult_7_difference_1220.json \
    --output-results outputs/o4-mini-low_results_difficult_7_difference_1220.json \
    --verbose \
    --restricted-reasoning \
    --reasoning-effort low \
    > logs/o4-mini-low_results_difficult_7_difference_1220.log 2>&1 &


# map navigation benchmark

python -m generators.sokoban_generator \
    --size 12 \
    --num-maps 25 \
    --tasks-type-1 1 \
    --tasks-type-2 1 \
    --tasks-type-3 1 \
    --tasks-type-4 1 \
    --output data/sokoban_12x12.json \
    --seed 42

python -m generators.frozenlake_generator \
    --size 12 \
    --num-holes 5 \
    --num-maps 20 \
    --tasks-type-1 1 \
    --tasks-type-2 1 \
    --tasks-type-3 1 \
    --tasks-type-4 1 \
    --tasks-type-5 1 \
    --output data/frozenlake_12x12.json \
    --seed 42

python main.py \
    --model deepseek-v3 \
    --data data/sokoban_test.json \
    --output results/deepseek-v3_sokoban_test.json \
    --verbose \
    --save-response results/deepseek-v3_sokoban_detailed_test.json

python main.py \
    --model deepseek-r1 \
    --data data/sokoban_test.json \
    --output results/deepseek-r1_sokoban_test.json \
    --verbose \
    --save-response results/deepseek-r1_sokoban_detailed_test.json

python main.py \
    --model o4-mini \
    --data data/sokoban_test.json \
    --output results/o4-mini_sokoban_test.json \
    --verbose \
    --save-response results/o4-mini_sokoban_detailed_test.json


nohup python main.py \
    --model deepseek-v3 \
    --data data/sokoban_12x12.json \
    --output results/deepseek-v3_sokoban_12x12_1220.json \
    --verbose \
    --save-response results/deepseek-v3_sokoban_detailed_12x12_1220.json \
    > logs/deepseek-v3_sokoban_1220.log 2>&1 &

nohup python main.py \
    --model deepseek-r1 \
    --data data/sokoban_12x12.json \
    --output results/deepseek-r1_sokoban_12x12_1220.json \
    --verbose \
    --save-response results/deepseek-r1_sokoban_detailed_12x12_1220.json \
    > logs/deepseek-r1_sokoban_1220.log 2>&1 &

nohup python main.py \
    --model o4-mini \
    --data data/sokoban_12x12.json \
    --output results/o4-mini-high_sokoban_12x12_1220.json \
    --verbose \
    --save-response results/o4-mini-high_sokoban_detailed_12x12_1220.json \
    > logs/o4-mini-high_sokoban_1220.log 2>&1 &

nohup python main.py \
    --model o4-mini \
    --data data/sokoban_12x12.json \
    --output results/o4-mini-low_sokoban_12x12_1221.json \
    --verbose \
    --restricted-reasoning \
    --save-response results/o4-mini-low_sokoban_detailed_12x12_1221.json \
    > logs/o4-mini-low_sokoban_1221.log 2>&1 &

nohup python main.py \
    --model gpt-4 \
    --data data/sokoban_12x12.json \
    --output results/gpt-4_sokoban_12x12_1220.json \
    --verbose \
    --save-response results/gpt-4_sokoban_detailed_12x12_1220.json \
    > logs/gpt-4_sokoban_1220.log 2>&1 &

nohup python main.py \
    --model gpt-5 \
    --data data/sokoban_12x12.json \
    --output results/gpt-5_sokoban_12x12_1220.json \
    --verbose \
    --save-response results/gpt-5_sokoban_detailed_12x12_1220.json \
    > logs/gpt-5_sokoban_1220.log 2>&1 &

nohup python main.py \
    --model deepseek-v3 \
    --data data/frozenlake_12x12.json \
    --output results/deepseek-v3_frozenlake_12x12_1220.json \
    --verbose \
    --save-response results/deepseek-v3_frozenlake_detailed_12x12_1220.json \
    > logs/deepseek-v3_frozenlake_1220.log 2>&1 &

nohup python main.py \
    --model deepseek-r1 \
    --data data/frozenlake_12x12.json \
    --output results/deepseek-r1_frozenlake_12x12_1220.json \
    --verbose \
    --save-response results/deepseek-r1_frozenlake_detailed_12x12_1220.json \
    > logs/deepseek-r1_frozenlake_1220.log 2>&1 &

nohup python main.py \
    --model o4-mini \
    --data data/frozenlake_12x12.json \
    --output results/o4-mini-high_frozenlake_12x12_1220.json \
    --verbose \
    --save-response results/o4-mini-high_frozenlake_detailed_12x12_1220.json \
    > logs/o4-mini-high_frozenlake_1220.log 2>&1 &

nohup python main.py \
    --model o4-mini \
    --data data/frozenlake_12x12.json \
    --output results/o4-mini-low_frozenlake_12x12_1221.json \
    --verbose \
    --restricted-reasoning \
    --save-response results/o4-mini-low_frozenlake_detailed_12x12_1221.json \
    > logs/o4-mini-low_frozenlake_1221.log 2>&1 &

nohup python main.py \
    --model gpt-4 \
    --data data/frozenlake_12x12.json \
    --output results/gpt-4_frozenlake_12x12_1220.json \
    --verbose \
    --save-response results/gpt-4_frozenlake_detailed_12x12_1220.json \
    > logs/gpt-4_frozenlake_1220.log 2>&1 &

nohup python main.py \
    --model gpt-5 \
    --data data/frozenlake_12x12.json \
    --output results/gpt-5_frozenlake_12x12_1220.json \
    --verbose \
    --save-response results/gpt-5_frozenlake_detailed_12x12_1220.json \
    > logs/gpt-5_frozenlake_1220.log 2>&1 &

