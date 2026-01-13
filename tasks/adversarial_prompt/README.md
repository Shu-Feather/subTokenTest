# Adversarial Prompt Canonicalization Benchmark

Benchmark for measuring how well language models detect and normalize adversarially perturbed prompts back to their canonical form.

## What This Benchmark Covers
- Character-level perturbations: leet substitutions, noise insertions, and mixed variants implemented in `src/perturbation.py`.
- Template-based prompts plus optional GPT-generated contexts across easy/medium/hard difficulty tiers.
- Chat-style prompting that requires the canonicalized answer between `<answer>` and `</answer>`; configurable few-shot examples.
- Pluggable model backends: local vLLM models, OpenAI Chat/Responses API (including o-series reasoning tokens), and DeepSeek chat models.
- Metrics and reports with exact-match accuracy, normalized Levenshtein similarity, tag-usage rates, and token-usage aggregates broken down by category, difficulty, and perturbation type.

## Quick Start
- Install deps (Python 3.9+ recommended, GPU required for vLLM):
  ```bash
  pip install -r requirements.txt
  ```
- Fast run on built-in templates with OpenAI (requires `OPENAI_API_KEY`):
  ```bash
  python main.py --model_type openai --num_samples 20
  ```
- Evaluate a local model with vLLM:
  ```bash
  python main.py --model_type vllm \
    --model_path meta-llama/Llama-3-8b-Instruct \
    --num_samples 50 \
    --gpu_memory_utilization 0.6
  ```
- Generate adversarial contexts with GPT and then evaluate:
  ```bash
  python generate_contexts.py --output outputs/generated_contexts.json --samples_per_difficulty 8
  python main.py --model_type openai --use_generated_contexts --contexts_file outputs/generated_contexts.json --num_samples 40
  ```
- Generate data only or reuse saved data:
  ```bash
  python main.py --model_type openai --generate_only
  python main.py --model_type openai --load_data outputs/test_data_*.json
  ```
- Add `--verbose` to see full prompts, responses, extraction traces, and token counts.

## CLI Reference (main.py)
- `--model_type {vllm,openai,deepseek}` choose backend; `--model_path` is required for vLLM (local path or HF repo id).
- Benchmark controls: `--num_samples`, `--seed`, `--output_dir`, `--config config/benchmark_config.yaml`.
- Context sources: `--use_generated_contexts`, `--contexts_file <json>`.
- vLLM tuning: `--gpu_memory_utilization`, `--tensor-parallel-size`, `--enforce-eager` / `--no-enforce-eager`.
- Data handling: `--generate_only` to skip evaluation, `--load_data <json>` to reuse samples.
- Logging: `--verbose` for detailed traces.

## Context Generation (generate_contexts.py)
- Uses OpenAI Chat API (default `gpt-4`, override `OPENAI_API_KEY` or `--api_key`).
- Key flags: `--categories`, `--difficulty_level {easy,medium,hard}`, `--samples_per_difficulty`, `--delay` between calls, `--cost` to track token spend, `--verbose` for logs.
- Outputs nested JSON mapping category → difficulty → contexts (written to `--output`, default `outputs/generated_contexts.json`). Point `benchmark.use_generated_contexts: true` and `benchmark.generated_contexts_file` to consume it.

## Configuration Highlights (`config/benchmark_config.yaml`)
- `benchmark`: sample count, seed, `output_dir`, and whether to use GPT-generated contexts.
- `models.vllm`: temperature, max_tokens, top_p, `gpu_memory_utilization`, `tensor_parallel_size`, `enforce_eager`.
- `models.openai`: `model_name` default `gpt-5`, `api_key_env`, temperature, `max_tokens`, `reasoning_effort`, timeout.
- `models.deepseek`: base URL, `model_name` default `deepseek-reasoner`, temperature, `max_tokens`, timeout.
- `prompt`: `use_few_shot` (default false) and `num_few_shot_examples` (up to 4 provided in `src/prompts.py`).
- `perturbation`: default enabled type is `leet_speak` with high substitution ratio; insertion and mixed modes are available with Arabic noise characters and `insertion_ratio` if you add them to `types`.
- `context_generation`: categories list, `samples_per_difficulty`, available difficulty levels, and generation model (`gpt-3.5-turbo` by default).
- `data.categories`: template pools for `harmful_instructions` (easy), `jailbreak_attempts` (medium), and `benign_queries` (easy) used when `use_generated_contexts` is false.

## Outputs and Metrics
- Generated samples are saved as `outputs/test_data_YYYYMMDD_HHMMSS.json`.
- Each run writes `outputs/<model>_<timestamp>/results.json`, `metrics.json`, `token_usage.json`, and `report.txt`.
- Metrics include exact-match rate, normalized Levenshtein similarity, average Levenshtein distance, tag-found rate, and breakdowns by category/difficulty/perturbation. Token usage is aggregated overall and by the same dimensions, including reasoning-token ratios for models that expose them.

## Analysis Utilities
- Quick failure and tag analysis:
  ```bash
  python utils/analyze_results.py --results outputs/<run>/results.json --threshold 0.9 --analyze_difficulty --analyze_tags
  ```
- Multi-run plotting and summaries: `scripts/benchmark_analyzer.py` loads all subfolders in a base directory to produce comparison tables and matplotlib/seaborn plots; see `accuracy_cost_analysis/` for example outputs.

## Project Layout
```
.
├── main.py                     # Benchmark entry point
├── generate_contexts.py        # GPT-based context generator with cost tracking
├── config/benchmark_config.yaml
├── src/
│   ├── perturbation.py         # Leet/insertion/mixed perturbations
│   ├── data_generator.py       # Template/GPT context sampler
│   ├── context_generator.py    # OpenAI-driven context authoring
│   ├── prompts.py              # System prompt and few-shot examples
│   ├── answer_extractor.py     # Tag-based answer parsing with fallback
│   ├── evaluator.py            # Metrics and batching
│   └── models/                 # Base + vLLM + OpenAI/DeepSeek adapters
├── tests/test_components.py    # Unit tests for core pieces
├── utils/analyze_results.py    # CLI helpers for inspecting runs
└── outputs/, datasets/, logs/  # Sample data and saved runs
```

## Testing
- Run core checks locally:
  ```bash
  python -m unittest tests/test_components.py
  ```

## Notes
- Models should return the canonical text between `<answer>` and `</answer>`; the extractor falls back to a cleaned first line if tags are missing.
- Use a consistent `--seed` when comparing models so templates/contexts and perturbations stay aligned.
