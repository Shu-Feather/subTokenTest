# RSA Randomart Difference Benchmark

Identify all character-level differences between two RSA fingerprint “randomart” patterns and score how well an LLM recovers them. The project ships a synthetic data generator, unified model interfaces (vLLM, OpenAI, DeepSeek), and an evaluator that reports coordinate- and replacement-level accuracy.

## What’s inside
- `main.py`: CLI for data generation and evaluation.
- `src/data_generator.py`: builds paired randomart patterns with controlled differences.
- `src/utils.py`: prompt construction, result/response persistence, config loading.
- `src/model_interface.py`: adapters for vLLM (local), OpenAI API (incl. o-series via Responses API), and DeepSeek.
- `src/evaluator.py`: parses model outputs and computes metrics.
- `examples/`: ready-to-run benchmark scripts.
- `tests/`: generator, evaluator, and pipeline checks.

## Installation
Requirements: Python 3.8+, `pip`, and (for vLLM) a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set API keys when using hosted models:
```bash
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
```

## Quick start
Generate 5 samples and evaluate with an OpenAI model, saving responses and results:
```bash
python main.py \
  --model-type openai \
  --model-name gpt-4o-mini \
  --num-samples 5 \
  --save-response outputs/openai_responses.json \
  --output-results outputs/openai_results.json
```

Run locally with vLLM (needs the model on disk and a GPU):
```bash
python main.py \
  --model-type vllm \
  --model-path /path/to/Qwen2-7B-Instruct \
  --num-samples 5 \
  --verbose
```

Generate data only (no model call):
```bash
python main.py \
  --generate-only \
  --num-samples 20 \
  --num-differences 7 \
  --output-data data/custom_samples.json
```

Use pre-generated data for multiple model runs:
```bash
python main.py \
  --model-type deepseek \
  --model-name deepseek-chat \
  --input-data data/test.json \
  --num-samples 10 \
  --output-results outputs/deepseek_results.json
```

## Task format
- Patterns follow the SSH-style randomart layout with a centered `[ RSA <key_size> ]` header, borders made of `+`/`-`/`|`, and an interior grid of ASCII tokens.
- Coordinates use the top-left `+` as `(0, 0)`; valid editable cells are `x ∈ [1, width]`, `y ∈ [1, height]`.
- Ground truth differences are stored as `{x, y, original, modified}` where `original`/`modified` are single characters (a space denotes blank).

Example sample snippet:
```json
{
  "pattern1": ["+-------------[ RSA 2048 ]-------------+", "..."],
  "pattern2": ["+-------------[ RSA 2048 ]-------------+", "..."],
  "ground_truth": [
    {"x": 7, "y": 3, "original": " ", "modified": "o"},
    {"x": 8, "y": 3, "original": "o", "modified": " "}
  ]
}
```

## Prompt and expected answer
`src/utils.create_prompt` builds the user message with both patterns and the coordinate conventions. Models should return differences inside `<answer>...</answer>` using lines like:
```
(7, 3):   -> o
(8, 3): o ->  
```
`Evaluator.parse_prediction` tolerates `:` or `：` and `->`, `→`, `-->` arrows; malformed lines are ignored.

## CLI essentials
- `--model-type {vllm, openai, deepseek}` and `--model-path` (vLLM) or `--model-name` (APIs) select the backend.
- `--num-samples`, `--num-differences`, `--pattern-width`, `--pattern-height`, `--key-size` control data generation. Defaults fall back to `config/config.yaml` when provided.
- `--generate-only`, `--output-data`, `--input-data` manage datasets without evaluation.
- `--coordinate-weight` / `--replacement-weight` tune the overall score blend (defaults 0.5/0.5).
- `--save-response` stores prompts/responses/predictions per sample; `--output-results` (default `outputs/results.json`) stores aggregated metrics and metadata.
- `--verbose` prints prompts, raw model outputs, parsed predictions, per-sample scores, and token usage.

## Configuration
`config/config.yaml` overrides defaults:
- `models`: per-backend options (vLLM tensor parallel size, GPU utilization, max sequence length; OpenAI/DeepSeek API base, timeout, max tokens, reasoning effort for o-series).
- `data_generation`: default `width`, `height`, `num_differences`, and `available_elements` used when CLI values are left at their defaults.
- `evaluation`: default weights for coordinate F1 vs. replacement accuracy.

## Metrics and outputs
- Coordinate precision, recall, and F1 on `(x, y)` matches.
- Replacement accuracy on cells whose coordinates match.
- Overall score = `coordinate_weight * F1 + replacement_weight * replacement_accuracy`.
- Results JSON (from `--output-results`) includes averages, totals, per-sample details, metadata (model, sizes, weights), token usage sums/averages, and a timestamp.
- Response JSON (from `--save-response`) preserves prompt, raw response, parsed prediction, ground truth, and per-sample token usage.

## Testing
Run the checks after installing dependencies:
```bash
pytest tests
```

## Extras
- Ready-made datasets live in `data/` (e.g., `test.json`, `difficult_*`).
- `examples/run_benchmark.sh` demonstrates typical invocations; adapt paths and model names as needed.
- `logs/` and `outputs/` are created automatically when you save responses or results.
