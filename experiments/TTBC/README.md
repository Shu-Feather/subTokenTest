# TTBC Experiments (Time-To-Budget Control)

This folder contains runnable reference implementations for the two budget-forcing experiments described in `test_time_scaling.md`.

## Overview

- **TTBC1 – Wait & Think More**  
  Suppress the model's `</think>` token and inject `Wait` to extend the thinking trace. Supports:
  - Sweeping the number of `Wait` injections (`n_wait`).
  - Optional upper bound on thinking tokens (`max_think_tokens`) to obtain the "shorter than baseline" points.

- **TTBC2 – Exact Thinking Tokens**  
  Forces every sample to use an exact thinking budget `t_exact` by iteratively injecting `Wait` until the budget is reached, then closing `</think>` and generating the final answer.

Both experiments:
- Target the DeepSeek-R1-Distill-Qwen-7B reasoning model by default (configurable).
- Use vLLM for fast local inference.
- Expect prompts that produce `<think> ... </think><answer> ... </answer>` style outputs.

## Quick Start

```bash
cd experiments/TTBC
python main.py ttbc1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset ../test/datasets/sample_ttbc.json \
  --n-wait 0 1 2 4 \
  --max-think-tokens none 256 512 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --output ttbc1_results.json

python main.py ttbc2 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset ../test/datasets/sample_ttbc.json \
  --t-exact 256 512 1024 \
  --output ttbc2_results.json
```

If no dataset is provided, a tiny built-in prompt (`"How many r in raspberry?"`) is used so the control loop can be smoke-tested without external files.

## Dataset Format

The runner accepts either JSON or JSONL. Each record should contain:
- `prompt` (string) – the user prompt.
- `answer` / `label` / `expected` (string) – ground-truth answer for accuracy.

## Notes

- All generation happens with `skip_special_tokens=False` during thinking to preserve the control tokens; answers can be post-processed with `extract_answer` to strip `<answer>...</answer>`.
- Stop tokens for thinking and answering are derived from the tokenizer for `</think>` and `</answer>`. If the model does not emit those tags, the runner will fall back to the configured max-token limits.
- The core logic lives in `main.py` under `TTBCController`, `run_ttbc1`, and `run_ttbc2`.
