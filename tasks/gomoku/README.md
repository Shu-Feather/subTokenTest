# Gomoku Benchmark for Large Language Models

Evaluate how well language models can read a Gomoku (Five-in-a-Row) board and decide whether White wins, Black wins, or there is no winner. The benchmark includes strict board generation, multiple model backends, reproducible datasets, and detailed scoring.

## Highlights
- Strict data generation: every winning board has exactly one winner and one winning line; no invalid dual-winner boards.
- Multiple backends: OpenAI (o-series, GPT-4/3.5), DeepSeek, local vLLM, vLLM server; Ollama/Transformers are kept as legacy options.
- Drop-in datasets: cached in `datasets/`, or generated on the fly; verification tooling included.
- Detailed metrics: accuracy, precision/recall/F1 per class, confusion matrix, error/parse-failure counts.
- Optional response logging with token usage summaries for cost/efficiency analysis.

## Repository Layout
- `main.py` - CLI entry point to run benchmarks.
- `config.py` - model presets (`DEFAULT_MODELS`), prompt templates, benchmark defaults.
- `src/board_generator.py` - strict board and dataset creation utilities.
- `src/benchmark_runner.py` - orchestrates data loading/generation, model calls, scoring, and result files.
- `src/model_interface.py` - adapters for OpenAI, DeepSeek, vLLM (local/server), Ollama, and Transformers.
- `src/vllm_interface.py` - high-throughput local vLLM runner.
- `src/evaluator.py` - response parsing, metrics, and confusion matrix.
- `src/response_logger.py` - optional JSON logger for per-case prompts/responses/tokens.
- `generate_datasets.py` - standalone dataset generator/validator.
- `configs/` - sample JSON configs (e.g., `configs/llama_config.json`).
- `datasets/` - pre-generated test sets; new ones are cached here by default.
- `results/`, `analysis/` - saved benchmark outputs and sample analyses.

## Setup
- Python 3.8+; GPU recommended for vLLM/local models.
- Install dependencies: `pip install -r requirements.txt`
- Set API keys as needed:
  - `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE` for custom endpoints)
  - `DEEPSEEK_API_KEY`

## Quick Start
- OpenAI example (uses presets from `config.py`):
  ```bash
  python main.py --models gpt-4 --board-sizes 9 --test-counts 50
  ```
- DeepSeek + OpenAI together on two board sizes:
  ```bash
  python main.py --models gpt-4 deepseek-v3 --board-sizes 9 15 --test-counts 100
  ```
- Local vLLM (ensure the `model_name` paths in `config.py` point to your weights):
  ```bash
  python main.py --models llama-3-8b qwen2-7b --board-sizes 9 --test-counts 20
  ```
- Run from a config file (fields mirror CLI flags): `python main.py --config-file configs/llama_config.json`
- Use an existing dataset instead of generating: 
  ```bash
  python main.py --models gpt-4 \
    --test-file datasets/easy_test_data_9x9.json \
    --board-sizes 9 --test-counts 200
  ```
- Generate data only (no model calls): 
  ```bash
  python main.py --generate-only --board-sizes 9 15 --test-counts 200 --data-dir datasets
  ```
- Debugging/analysis options:
  - `--verbose` prints full prompts and responses.
  - `--save-response logs/gpt4_9x9.json` stores per-case responses and token usage.

## CLI Flags (main.py)
- `--models` one or more model keys from `DEFAULT_MODELS` (`gpt-5`, `o4-mini`, `gpt-4`, `gpt-3.5-turbo`, `deepseek-v3`, `deepseek-r1`, `llama-3-8b`, `qwen2-7b`, `qwen2.5-14b`, `qwen2.5-32b`, `deepseek-r1-distill-qwen-32b`, `vllm-server`, legacy `qwen2.5`, `llama3.2`, or `all`).
- `--board-sizes` board sizes (defaults to 9/15/19 if omitted).
- `--test-counts` number of cases per board size (defaults to 100/200/500 if omitted).
- `--test-file` path to a JSON dataset (bypasses generation).
- `--output-dir` where per-run JSON results and logs go (default `results`).
- `--data-dir` cache for generated datasets (default `datasets`).
- `--generate-only` only creates datasets, does not query models.
- `--save-response` path to a JSON log of every prompt/response/token count.
- `--verbose` prints prompts/responses and token details during the run.

## Datasets and Verification
- Generate strict datasets with density and distribution controls:
  ```bash
  # Default: 9/15/19 boards, 100 cases each, density 0.2-0.5
  python generate_datasets.py

  # Dense diagonal-only wins, 15x15, 100 cases
  python generate_datasets.py --board-sizes 15 --test-counts 100 \
    --white-wins 0.5 --black-wins 0.5 --no-winner 0 \
    --horizontal 0 --vertical 0 --diagonal-down 0.5 --diagonal-up 0.5 \
    --min-density 0.6 --max-density 0.8 --name dense_diagonal
  ```
- Verify an existing dataset (strict winner/direction checks):
  ```bash
  python generate_datasets.py --verify-only --dataset-path datasets/hard_test_data_15x15.json
  ```
- Output files live in `datasets/` with optional metadata; `generate_datasets.py` automatically re-validates generated files.

## Outputs and Metrics
- Each run writes:
  - Per-configuration JSON files under `results/`.
  - `benchmark_summary_*.json` aggregating model accuracies, token usage, and efficiency scores.
  - `results/benchmark.log` with progress and warnings.
  - Optional response log from `--save-response` with every board, extracted answer, correctness, and token counts.
- Metrics include overall accuracy, per-class precision/recall/F1 for `WHITE_WINS`, `BLACK_WINS`, `NO_WINNER`, confusion matrix, error count, and parse-failure count. Use `src/evaluator.print_evaluation_summary` to render summaries programmatically.

## Board and Dataset Format
- Board symbols: `W` (White), `B` (Black), `E` (empty). Five in a row in any direction wins.
- Example board string (9x9):
  ```
  WEEWBEWBB
  BWWEWBWEB
  BEWBWEBWB
  WBWWBWEBB
  EEBEWBWWE
  EBEWWBBEE
  BEEWEBBBW
  WEWBWBEEB
  EBEEEEBBB
  ```
- Example dataset entry:
  ```json
  {
    "board": "WEEWBEWBB\nBWWEWBWEB\n...trimmed...",
    "expected": "WHITE_WINS",
    "board_size": 9,
    "win_direction": "HORIZONTAL",
    "density": 0.34
  }
  ```

## Adding or Tweaking Models
- Edit `config.py` to extend `DEFAULT_MODELS` with a new `ModelConfig`. Choose `model_type` from `openai`, `deepseek`, `vllm`, `vllm-server`, `ollama`, or `transformers`.
- For o-series OpenAI models, `reasoning_effort` in `ModelConfig` controls the Responses API thinking budget.
- Local vLLM models require accessible weight paths and a suitable GPU; see `vllm_usage.md` for throughput tips.

## Tips
- If a requested dataset file is missing, the runner auto-generates and caches it under `--data-dir`.
- Use smaller `--test-counts` with `--verbose` to sanity-check prompt formatting before long runs.
- When logging responses, place logs outside `results/` if you plan to diff summaries separately (e.g., `--save-response logs/run1.json`).
