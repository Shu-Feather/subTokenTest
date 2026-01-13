#!/bin/bash
# Example script to run the benchmark
# Path: examples/run_benchmark.sh

# Example 1: Run with VLLM model
python main.py \
    --model-type vllm \
    --model-path /path/to/your/model \
    --num-samples 20 \
    --num-differences 5 \
    --verbose \
    --save-response outputs/vllm_responses.json \
    --output-results outputs/vllm_results.json

# Example 2: Run with OpenAI model
python main.py \
    --model-type openai \
    --model-name gpt-4 \
    --num-samples 20 \
    --num-differences 5 \
    --save-response outputs/openai_responses.json \
    --output-results outputs/openai_results.json

# Example 3: Run with DeepSeek model
python main.py \
    --model-type deepseek \
    --model-name deepseek-chat \
    --num-samples 20 \
    --num-differences 5 \
    --save-response outputs/deepseek_responses.json \
    --output-results outputs/deepseek_results.json

# Example 4: Generate data only
python main.py \
    --generate-only \
    --num-samples 10 \
    --num-differences 5 \
    --output-data data/test_samples.json

# Example 5: Use pre-generated data
python main.py \
    --model-type vllm \
    --model-path /path/to/your/model \
    --input-data data/test_samples.json \
    --save-response outputs/responses.json \
    --output-results outputs/results.json