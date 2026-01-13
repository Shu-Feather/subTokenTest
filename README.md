# SubTokenTest

SubTokenTest is a suite of independent, task-specific benchmarks for large language models with a unified runner. Each task ships with its own prompts, data generation, and evaluation logic, while `cli.py` in the repo root lets you launch any task with consistent model backends (APIs or local vLLM).

## Highlights
- Unified entrypoint: `python cli.py run <task> -- <task-flags>` discovers `tasks/<task>/main.py`, sets the working directory, and forwards all arguments.
- Centralized configs: copies of every task’s configs live under `configs/<task>/`; `configs/locator.py` resolves relative paths for you.
- Shared model adapters in `models/`: OpenAI-compatible APIs (including reasoning-token accounting for o-series/DeepSeek) and vLLM with chat-template fallback.
- Rich datasets and generators: ready-made JSONL corpora in `datasets/` plus per-task generators (e.g., `generate_contexts.py`, `generate_datasets.py`, `run_model_test.py`).
- Built-in token usage reporting: most tasks record prompt/completion/reasoning tokens and ratios alongside accuracy metrics.

## Setup
- Python 3.10+ recommended; GPU is needed for vLLM.
- Install dependencies with `uv` (see `/scripts/UV_SETUP.md` for doc in `uv`):
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv sync          # add --group dev for lint/test tools
  ```
- Set API keys as needed: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` (DeepSeek also honors `--base_url`).

## Quickstart
- List tasks:
  ```bash
  python cli.py list
  ```
- Run a task (flags after `--` go straight to the task script):
  ```bash
  # Adversarial prompt canonicalization with OpenAI API
  python cli.py run adversarial_prompt -- \
    --model_type openai \
    --config configs/adversarial_prompt/benchmark_config.yaml \
    --num_samples 20

  # Map navigation with a local vLLM model
  python cli.py run map_navigation -- \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --model-type vllm \
    --config configs/map_navigation/model_config.py \
    --data data/sokoban_test.json \
    --output results/llama3_sokoban.json
  ```
- Please view `/scripts/cli_run.sh` for detailed instructions to run each task.
- You can also call scripts directly, e.g., `python tasks/gomoku/main.py --help`.
- You can also view the experiments log in the `/scripts` repo for example.

## Benchmarks at a Glance
- `adversarial_prompt`: Normalize perturbed/jailbreak prompts back to their canonical form with exact-match/Levenshtein metrics and token-usage breakdowns; supports GPT-generated contexts.
- `aligned_table`: Reconstruct aligned tables (LaTeX/Markdown/plaintext) from structured data; reports alignment/content scores per format.
- `biological_sequence`: DNA/RNA complements and protein code conversions across four sub-tasks; configurable model list and data loading/generation.
- `cipher_decipher`: Morse and Caesar encode/decode tasks with async runner, prompt-style overrides, and response logging.
- `context_aware_redaction`: Detect and mask PII (phone, ID, credit card) in context; works with API or vLLM backends.
- `gomoku`: Classify Gomoku board states (win/draw) across board sizes; includes dataset generators and response saving.
- `map_navigation`: QA over Sokoban and FrozenLake maps using system/user prompts; exact-match evaluator with batch support.
- `rsa_randomart`: Find coordinate differences between RSA randomart patterns; supports data-only generation and weighted coordinate/replacement scoring.
- `tree`: Binary-tree reasoning (structure queries and path analysis) with similarity scoring and difficulty-aware stats.
- `typewriter`: Typewriter/backspace simulation tasks with system/few-shot prompting, optional vLLM batching, and multi-model evaluation.

All tasks understand `--restricted_reasoning` (where implemented) to nudge models toward concise outputs, and most accept `--verbose` to print prompts/responses and token tallies.

## Configs, Models, and Data
- Config resolution: pass relative names and let `configs/locator.py` find them, e.g., `--config config.yaml` resolves to `configs/<task>/config.yaml` when available.
- Model backends:
  - API (`openai`/`deepseek`/`api`): set `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`; optional `--reasoning-effort` for o-series; `models/api.py` normalizes usage fields.
  - vLLM: enable with `--model_type vllm` or `--use-vllm`; tune `--gpu_memory_utilization`, `--tensor-parallel-size`, `--batch-size`, `--enforce-eager` as supported by each task.
- Datasets: curated JSONL files in `datasets/` cover every benchmark (see `datasets/README.md` for examples and counts). Most tasks also ship generators under `tasks/<task>/scripts/` or root-level helpers (e.g., `generate_datasets.py`, `generate_contexts.py`).
- Outputs: tasks typically write results/metrics (and optional response logs) under `results/` or `outputs/` with timestamps; many also emit token-usage summaries and human-readable reports.

## Repository Map
- `cli.py`: unified dispatcher; legacy `python cli.py <task>` also works.
- `tasks/`: benchmark implementations (see subfolders listed above).
- `configs/`: centralized task configs plus `locator.py` helper.
- `models/`: shared API and vLLM adapters with token accounting utilities.
- `datasets/`: ready-made JSONL datasets spanning all tasks.
- `experiments/`: research notebooks/scripts (e.g., linear probes, TTBC, reproduction helpers).
- `test/`: sample datasets and simple runner scripts for smoke checks.
- `run.sh` / `cli_run.sh`: curated command examples for common runs.

## Testing and Validation
- Task-level tests exist where applicable (e.g., `tasks/adversarial_prompt/tests/test_components.py`). Run with `python -m pytest` from the repo root after installing dev extras: `uv sync --group dev`.
- For functional checks, run a small-sample benchmark with `--num_samples` trimmed to a handful of cases before launching full sweeps.

## Tips
- Keep runs inside the repo root so relative paths and centralized configs resolve correctly.
- Use `python cli.py run <task> -- --help` to see task-specific flags (batching, generation-only modes, logging, etc.).
- When comparing models, set seeds and reuse generated datasets to keep samples aligned across runs.
