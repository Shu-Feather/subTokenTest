# Benchmark Reproduce

Run lightweight, reproducible evaluations over the prepared task datasets (`/datasets/*_datasets.jsonl`) using the shared `models` interfaces.

## Usage

```bash
cd experiments/benchmark_reproduce

python main.py \
  --datasets /datasets \             # directory containing *_datasets.jsonl
  --model-type api \                 # api | openai | deepseek | vllm
  --model-name gpt-4o \              # model id or path
  --batch-size 4 \                   # optional, useful for vLLM
  --max-samples 100 \                # optional cap per dataset
  --output results.json \            # optional aggregated output file
  --save-response responses.jsonl \  # optional: save prompts/responses
  --verbose                          # optional: print prompts/responses live
```

To run a specific file (e.g., `bio_seq_datasets.jsonl`):
```bash
python main.py \
  --datasets /datasets \
  --files bio_seq_datasets.jsonl \
  --model-type api \
  --model-name gpt-4o
```

Flags:
- `--datasets`: directory holding the generated JSONL files (default: `/datasets` at repo root).
- `--files`: specific JSONL files to load (relative to `--datasets` unless absolute); overrides globbing all `*_datasets.jsonl`.
- `--model-type`: backend type (`api`, `openai`, `deepseek`, `vllm`).
- `--model-name`: model identifier/path forwarded to the shared factory.
- `--batch-size`: hint for vLLM batch size (default: 1).
- `--max-samples`: optional limit per dataset file.
- `--output`: optional JSON output with per-dataset metrics.
- `--verbose`: print per-sample progress and token usage snippets.
- `--save-response`: path to JSONL file storing prompts/responses/usage.

## Output

For each `<task>_datasets.jsonl`, the script prints and (optionally) saves:
- `exact_match_rate`: fraction of predictions exactly matching the ground truth (case-insensitive) extracted strictly from `<answer>...</answer>`.
- `token_usage`: aggregated `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, and `total_tokens` across samples.

If `--output` is provided, a JSON file with per-dataset entries is written to the specified path.
