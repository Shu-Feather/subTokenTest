#!/usr/bin/env bash

# Lightweight CLI runs against the small datasets in test/datasets.
# Assumes datasets were generated via test/generate_test_datasets.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${ROOT_DIR}/test/datasets"
RESULTS_ROOT="${ROOT_DIR}/test/results"
mkdir -p "${RESULTS_ROOT}"

# --- Adversarial Prompt (use generated contexts) ---
python cli.py run adversarial_prompt -- \
  --model_type openai \
  --config configs/adversarial_prompt/benchmark_config.yaml \
  --use_generated_contexts \
  --contexts_file "${DATA_ROOT}/adversarial_prompt_test.json" \
  --num_samples 1 \
  --verbose

# --- Aligned Table ---
python cli.py run aligned_table -- \
  --model_type openai \
  --model_name gpt-4 \
  --config configs/aligned_table/config.yaml \
  --test_file "${DATA_ROOT}/aligned_table_test.json" \
  --format latex \
  --output "${RESULTS_ROOT}/aligned_table_results.json" \
  --verbose

# --- Biological Sequence ---
python cli.py run biological_sequence -- \
  --config configs/biological_sequence/gpt_quick.json \
  --models gpt-4 \
  --load-data "${DATA_ROOT}/bio_seq_test.json" \
  --verbose

# --- Cipher-Decipher ---
python cli.py run cipher_decipher -- \
  --test-file "${DATA_ROOT}/cipher_decipher_test.json" \
  --models openai \
  --config configs/cipher_decipher/gpt_config.yaml \
  --samples 1 \
  --verbose

# --- Context-Aware Redaction ---
python cli.py run context_aware_redaction -- \
  --model_type api \
  --model_name gpt-4 \
  --api_key $OPENAI_API_KEY \
  --config configs/context_aware_redaction/config.yaml \
  --dataset "${DATA_ROOT}/context_aware_redaction_test.json" \
  --num_samples 1 \
  --verbose

# --- Gomoku ---
python cli.py run gomoku -- \
  --models gpt-4 \
  --test-counts 1 \
  --board-sizes 9 \
  --verbose \
  --test-file "${DATA_ROOT}/gomoku_test.json"

# --- Map Navigation (Sokoban) ---
python cli.py run map_navigation -- \
  --model gpt-4 \
  --model-type api \
  --data "${DATA_ROOT}/map_navigation_sokoban_test.json" \
  --output "${RESULTS_ROOT}/map_nav_sokoban.json" \
  --verbose

# --- Map Navigation (FrozenLake) ---
python cli.py run map_navigation -- \
  --model gpt-4 \
  --model-type api \
  --data "${DATA_ROOT}/map_navigation_frozenlake_test.json" \
  --output "${RESULTS_ROOT}/map_nav_frozenlake.json" \
  --verbose

# --- RSA RandomArt ---
python cli.py run rsa_randomart -- \
  --config configs/rsa_randomart/config.yaml \
  --model-type openai \
  --model-name gpt-4 \
  --num-samples 1 \
  --input-data "${DATA_ROOT}/rsa_randomart_test.json" \
  --output-results "${RESULTS_ROOT}/rsa_results.json" \
  --verbose

# --- Tree (task1 & task2) ---
python cli.py run tree -- \
  --model_type openai \
  --model_name gpt-4 \
  --task_type task1 \
  --test_file "${DATA_ROOT}/tree_task1_test.json" \
  --num_samples 1 \
  --verbose

python cli.py run tree -- \
  --model_type openai \
  --model_name gpt-4 \
  --task_type task2 \
  --test_file "${DATA_ROOT}/tree_task2_test.json" \
  --num_samples 1 \
  --verbose

# --- Typewriter ---
python cli.py run typewriter -- \
  --models gpt-4 \
  --difficulty hard \
  --test-file "${DATA_ROOT}/typewriter_test.json" \
  --num-samples 5 \
  --verbose

echo "All test CLI runs completed."
