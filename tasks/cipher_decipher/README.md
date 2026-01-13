# Cipher & Decipher Benchmark

Evaluate how reliably language models handle two classic cipher families (Morse and Caesar) with controlled prompts, difficulty-aware data generation, and detailed result exports.

Token Bench is designed to evaluate how tokenization impacts large language models’ performance across diverse linguistic tasks.
## What it Covers
- Morse encode/decode and Caesar encode/decode tasks with configurable shift sets.
- Difficulty-aware text generation (easy/medium/hard) and optional reproducible datasets.
- Prompt style controls (`basic`, `detailed`, `step_by_step`) with per-task overrides.
- Token/time tracking plus per-task and per-difficulty accuracy, similarity, and error summaries.
- Pluggable model backends: OpenAI, DeepSeek, HuggingFace transformers, and local vLLM.

## Project Layout
- `main.py` – async entrypoint for running benchmarks.
- `configs/cipher_decipher/` – ready-to-use configs (default, OpenAI o-series, DeepSeek, vLLM variants).
- `src/` – benchmark engine:
  - `benchmark_runner.py` orchestrates generation, prompting, evaluation, exports, and logging.
  - `ciphers/` Morse and Caesar reference implementations.
  - `data_generation/` template-based generator and optional LLM-driven dataset builder.
  - `evaluation/` answer extraction, normalization, metrics, and exporters.
  - `models/` adapters for OpenAI, DeepSeek, HuggingFace, and vLLM.
  - `utils/prompts.py` prompt templates and validators; `utils/logger.py` response logger.
- `datasets/` – sample difficulty-split text corpora and stats (JSON + txt).
- `logs/`, `results/`, `outputs/` – created at runtime for logs, response traces, and exports.
- `scripts/` – post-hoc analysis/visualization helpers.
- `tests/` – unit tests for ciphers, prompts, evaluation, and generators.

## Setup
1. (Optional) create a virtual env.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide API keys as env vars or directly in config:
   - `OPENAI_API_KEY` for OpenAI models or dataset generation.
   - `DEEPSEEK_API_KEY` for DeepSeek.
   - HuggingFace/vLLM entries expect a valid `model_name`/`model_path` and installed weights.

## Running the Benchmark
- Quick start with the default config:
  ```bash
  python main.py
  ```
- Common options:
  - `--config PATH` – choose a YAML under `configs/cipher_decipher/` (e.g., `configs/cipher_decipher/gpt_config.yaml` or `configs/cipher_decipher/deepseek_config.yaml`).
  - `--models m1 m2` – subset of models defined in the config (defaults to all available).
  - `--samples N` – samples per task split across difficulty ratios (see `data_generation.difficulty_distribution`).
  - `--test-file datasets/sample_dataset.json` – use a fixed dataset instead of on-the-fly generation.
  - `--prompt-style {basic,detailed,step_by_step}` – force a single prompt style for every task.
  - `--save-responses` (optional `--response-log PATH`) – persist prompts/responses + token usage to JSON/TXT under `logs/`.
  - `--verbose` – print prompts, responses, golden answers, and extracted answers while running.

### Expected LLM Response Format
The evaluator extracts the text inside `<answer>...</answer>`. A `<think>...</think>` block is optional and ignored for scoring.

```
<think>: Optional reasoning here. </think>
<answer>: .... . .-.. .-.. ---</answer>
```

## Configuration Highlights (`configs/cipher_decipher/config.yaml`)
- `test_config`: enable/disable tasks, set `num_samples`, min/max text length, and Caesar `caesar_shifts`.
- `data_generation`: difficulty distribution for newly generated texts; optional LLM generation fallback texts.
- `models`: one block per backend (`type` of `openai`, `deepseek`, `huggingface`, or `vllm`). API keys can be injected via `${ENV_VAR}`.
- `prompt_settings`: default prompt style and optional per-task overrides.
- `evaluation`: strict/lenient matching, case sensitivity, punctuation handling, and timeouts.
- `output`: toggle saving results and choose the `results_dir`.

Additional ready-made configs live in `configs/cipher_decipher/` for specific backends (OpenAI o-series, DeepSeek reasoner, vLLM Llama/Qwen). Point `--config` to them to switch stacks.

## Datasets
- On-the-fly generation uses `TextGenerator` with difficulty-balanced sampling.
- For reproducibility, pass `--test-file` a JSON with `easy`/`medium`/`hard` arrays (see `datasets/sample_dataset.json`).
- To create new corpora with an LLM (requires OpenAI key):
  ```bash
  python generate_dataset.py --samples 100 --output datasets/generated_texts.json
  ```

## Outputs
- Metrics are written to `results/` in JSON, CSV, and summary TXT using the pattern `{model}_experiment_<MMDD>_*`.
- `logs/responses_*.json|txt` (when `--save-responses` is set) store prompts, raw outputs, extracted answers, correctness, similarity, and token usage (total/input/output/reasoning) grouped by task and difficulty.
- Runtime logs stream to `benchmark.log`; console summary includes per-task and per-difficulty accuracy plus token-usage tables when available.

## Models
Define each model in the chosen config:
- **OpenAI**: `type: openai`, `model_name`, `api_key` (or env), optional `reasoning_effort` for o-series and `max_tokens`.
- **DeepSeek**: `type: deepseek`, `base_url`, `model_name`, `api_key`.
- **HuggingFace**: `type: huggingface`, `model_name`, device, and generation params (downloads weights unless locally cached).
- **vLLM**: `type: vllm`, `model_path` (local or HF id), GPU settings, sampling params, chat-template toggle.

## Testing and Examples
- Run unit tests:
  ```bash
  pytest
  ```
- `example_usage.py` walks through cipher helpers, text generation, prompt templates, and evaluation flow without calling external APIs.

## Tips
- Keep answers inside `<answer>` tags; everything else is ignored for grading.
- Use `--save-responses` when comparing token efficiency across models; the CLI summary will include per-task/difficulty token stats if logs are present.
- For long runs, pin a config and dataset (`--test-file`) to make benchmarks comparable across model revisions.
