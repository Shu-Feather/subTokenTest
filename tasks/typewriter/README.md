# Typewriter Benchmark

Benchmark for Large Language Models that simulates how a person types: character-by-character growth of a word and keystrokes that include backspace. The suite covers API models (OpenAI, DeepSeek), Hugging Face downloads, and local weights with optional vLLM acceleration, and reports both quality metrics and token usage when available.

## What It Tests
- Task 1 — Progressive typing: given a word, produce its growth sequence (e.g., `hello` → `h→he→hel→hell→hello`).
- Task 2 — Backspace handling: given a spaced typing log that may include `←`, produce the final text (e.g., `h e l l o ← ← k o` → `helko`).
- Two prompt styles are supported per task: `system` (instructional) and `few_shot` (in-context examples).
- Metrics include per-task accuracy/score statistics plus overall accuracy/score; API models also return token-usage aggregates.

## Repository Layout
- `main.py`: CLI entry point for running evaluations.
- `tasks/`: task definitions, prompts, and response evaluators for Task 1 and Task 2.
- `evaluation/`: scoring logic and overall benchmark aggregator.
- `datasets/`: pre-built JSON test sets with easy/medium/hard splits and a wordlist for generation.
- `configs/typewriter/model_config.py`: built-in model configurations (API, Hugging Face, local, vLLM-capable).
- `models/`: wrappers for OpenAI, DeepSeek, Hugging Face, local weights, and vLLM batch inference.
- `scripts/`: utilities for dataset generation, quick model tests, result comparison, and token analysis.
- `results/`: default output directory for JSON results and text summaries.

## Setup
1) Install dependencies (use a virtual environment if desired):
```bash
pip install -r requirements.txt
# Optional: pip install vllm  # or vllm-cuda121 for CUDA 12.1
```
2) Provide credentials in `.env` (or environment variables):
```
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
# For local models, set paths or edit configs/typewriter/model_config.py
# Example: LLAMA3_8B_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
```

## Run the Benchmark
- List available models:
```bash
python main.py --list-models
```
- Evaluate one model on the bundled datasets (all difficulties, system prompt):
```bash
python main.py --models gpt-4 --prompt-types system --test-dir datasets
```
- Compare several models and prompt types, limiting samples and choosing difficulty:
```bash
python main.py --models gpt-4 o4-mini deepseek-chat \
  --prompt-types system few_shot \
  --difficulty medium \
  --num-samples 50 \
  --output-dir results
```
- Run a local model with vLLM batch inference:
```bash
python main.py --models llama3-8b --prompt-types system --use-vllm --batch
```
- Use a custom test file (must contain `task1` and `task2` keys):
```bash
python main.py --models gpt-4 --prompt-types few_shot --test-file path/to/tests.json
```

Key flags:
- `--prompt-types system|few_shot` choose prompt style (can pass both).
- `--difficulty easy|medium|hard|all` filters difficulty slices when loading from `--test-dir` (default `datasets`).
- `--num-samples N` limits samples per task after loading.
- `--use-vllm` uses vLLM for local models; `--batch` turns on batched generation (vLLM only).
- `--verbose` prints stack traces if an evaluation error occurs.

## Datasets
- Pre-generated files live in `datasets/`:
  - `test_cases_full.json` combines all difficulties and includes `metadata`.
  - `test_cases_easy.json`, `test_cases_medium.json`, `test_cases_hard.json` expose individual slices.
  - `wordlist.txt` seeds dataset generation.
- JSON structure example:
```json
{
  "task1": {
    "easy": ["worn", "coos", "..."],
    "medium": ["binary", "..."],
    "hard": ["astonishing", "..."],
    "all": ["... combined ..."]
  },
  "task2": {
    "easy": ["h e l l o ← ← k o", "..."],
    "medium": ["a b c d ← ← e f", "..."],
    "hard": ["z x c v ← ← ← b n m ← ←", "..."],
    "all": ["... combined ..."]
  },
  "metadata": {
    "total_task1_samples": 180,
    "total_task2_samples": 180,
    "samples_per_difficulty": 60,
    "task2_difficulty_distribution": { "...": "..." }
  }
}
```
- If no dataset file is found, the CLI falls back to the legacy on-the-fly generator.
- Generate fresh datasets from the wordlist:
```bash
python scripts/generate_datasets.py --samples 40 --seed 42 --output datasets
```

## Models
- API: `gpt-5`, `o4-mini`, `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `deepseek-chat`, `deepseek-reasoner`.
- Hugging Face downloads: `llama2-7b`, `mistral-7b`.
- Local weights (paths configurable): Llama 3 variants, multiple Qwen releases, DeepSeek R1 distill models, etc. Token usage is not reported for local models.
- Add a model in `configs/typewriter/model_config.py` by extending `MODEL_CONFIGS`:
```python
from config.model_config import ModelConfig

MODEL_CONFIGS["my-model"] = ModelConfig(
    name="my-model",
    model_type="openai",  # or deepseek | huggingface | local
    model_id="org/model-id",
    api_key=os.getenv("MY_API_KEY"),
    max_tokens=512,
    temperature=0.0
)
```

## Outputs and Metrics
- Results are saved to `results/<model>_<timestamp>.json` plus a human-readable `<model>_<timestamp>_summary.txt`.
- JSON highlights:
```json
{
  "model_info": {"name": "gpt-4", "type": "openai", "model_id": "gpt-4"},
  "prompt_type": "system",
  "used_batch": false,
  "supports_usage_tracking": true,
  "metrics": {
    "overall_accuracy": 0.95,
    "overall_score": 0.96,
    "task1_metrics": {"accuracy": 0.92, "average_score": 0.93, "std_score": 0.08, "...": "..."},
    "task2_metrics": {"accuracy": 0.98, "average_score": 0.99, "std_score": 0.05, "...": "..."}
  },
  "total_usage": {"total_tokens": 1234, "input_tokens": 800, "output_tokens": 434, "samples_with_usage": 40},
  "task1_results": [{"input": "hello", "response": "h→he→...", "exact_match": true, "score": 1.0, "...": "..."}],
  "task2_results": [{"input": "h e l l o ← ← k o", "response": "helko", "exact_match": true, "char_accuracy": 1.0, "...": "..."}]
}
```
- Per-task metrics include accuracy, average/median/min/max score, and standard deviation. Overall metrics average the two tasks. Task 2 also reports character-level accuracy, and Task 1 scoring rewards step-by-step correctness even without an exact match.

## Utility Scripts
- Quick smoke test for one model:
```bash
python scripts/test_model.py gpt-4 --verbose
```
- Compare multiple result files:
```bash
python scripts/compare_results.py results/*.json
```
- Analyze token usage for a run:
```bash
python scripts/analysis_tokens.py results/gpt-4_*.json
```
- `scripts/visualize_response.html` provides a simple HTML viewer for saved responses.

## Notes
- `--batch` only applies when using vLLM-backed local models; API models always run sequentially.
- Token usage is reported for OpenAI and DeepSeek APIs; local/Hugging Face models return usage as `null`.
- Typing logs use spaces between keystrokes and `←` for backspace; prompts expect that exact symbol.
