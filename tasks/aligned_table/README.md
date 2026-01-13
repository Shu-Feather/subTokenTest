# Aligned-Table Benchmark

Evaluate how well large language models can turn structured content into perfectly aligned tables in LaTeX, Markdown, and plain-text ASCII. The benchmark generates table descriptions, prompts an LLM with strict formatting rules, and scores both the correctness of the cell values and the visual alignment of delimiters.

## What the Benchmark Covers
- Three output formats: LaTeX (`&`/`\hline`), Markdown (`|` separators), and text ASCII boxes.
- Two score components: cell-level content accuracy and delimiter alignment; the weighted total is used for ranking.
- Multiple backends: OpenAI, DeepSeek (OpenAI-compatible), and local models via vLLM.
- Built-in synthetic data generation plus support for pre-made or GPT-generated contexts.
- Optional verbose and logging modes that expose prompts, responses, and token-usage breakdowns (including reasoning tokens for o-series models).

## Repository Layout
- `main.py` — CLI entry point for running the benchmark.
- `config/config.yaml` — default settings for tests, models, and evaluation weights.
- `src/` — data generation, prompt construction, LLM interfaces, evaluation, and table utilities.
- `datasets/` — ready-made context files (`easy`, `medium`, `hard`, `simple`...) consumable via `--test_file`.
- `scripts/generate_contexts.py` — create new contexts with GPT.
- `examples/analyze_results.py` — compare multiple result files and plot scores (requires `matplotlib`).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # includes vLLM/torch; for cloud-only runs you can slim this down to openai, pyyaml, numpy, tqdm.
```

Set credentials when using hosted APIs:
- OpenAI: `export OPENAI_API_KEY=...`
- DeepSeek: `export DEEPSEEK_API_KEY=...`

## Running the Benchmark
OpenAI example:
```bash
python main.py \
  --model_type openai \
  --model_name gpt-4o \
  --num_samples 25 \
  --output results.json
```

DeepSeek example:
```bash
python main.py \
  --model_type deepseek \
  --model_name deepseek-chat \
  --output results.json
```

Local vLLM example:
```bash
python main.py \
  --model_type vllm \
  --model_name /path/to/model \
  --num_samples 10 \
  --output results.json
```

Useful flags:
- `--test_file datasets/easy_contexts.json` — reuse saved contexts instead of synthesizing.
- `--format latex|markdown|text` — restrict evaluation to one format.
- `--save_test_cases cases.json` — export the generated cases for later reuse.
- `--save_responses` — include prompts/responses in the result JSON.
- `--save_log logs/run_log.json` — persist a detailed log with prompts, usage info, and evaluations.
- `--verbose` — stream each test case with its prompt, response, scores, and token stats.

## Configuration Highlights (`config/config.yaml`)
- `test`: `num_samples`, `table_formats`, optional `format_distribution` (used when normalizing external contexts), and row/column ranges (defaults: 3–8 rows, 3–6 cols).
- `data_generation`: entity types used by the synthetic generator (countries, companies, movies, books, universities, products, athletes, cities).
- `models`: backend-specific parameters (temperature, max tokens, timeouts; reasoning effort for OpenAI o-series).
- `evaluation`: weights for `content_weight` and `alignment_weight` (default 0.6 / 0.4).
- `context_generation`: defaults for `scripts/generate_contexts.py`.

## Test Cases and Custom Contexts
- Generated cases include `id`, `entity_type`, `table_data`, `context`, `table_format`, `num_rows`, `num_cols`.
- Context files from `scripts/generate_contexts.py` (or your own) only need `table_data`, `context`/`description`, `num_rows`, and `num_cols`; `main.py` will normalize them by adding `id` and assigning a `table_format` using `format_distribution` if not provided.
- To create new contexts with GPT:
```bash
python scripts/generate_contexts.py \
  --model gpt-4o \
  --num_contexts 50 \
  --output datasets/custom_contexts.json
```

## Scoring
- **Content score**: cell-level accuracy after normalizing table shapes; missing rows/columns count against the score. Comparison is case-insensitive.
- **Alignment score**: checks vertical alignment of delimiters (`&` for LaTeX, `|/+` for text, `|` for Markdown) across all lines.
- **Total score**: `content_weight * content_score + alignment_weight * alignment_score`.

## Outputs
`results.json` (path configurable via `--output`) contains:
- `config`: `model_type`, `model_name`, `num_test_cases`.
- `overall_statistics`: averages, alignment rate, and perfect-score counts/rates.
- `token_statistics`: total and per-case averages for prompt, completion, reasoning, and visible output tokens (0 when not reported by the backend).
- `format_statistics` and `format_token_statistics`: same metrics broken down by table format.
- `detailed_results`: per-case scores, table format, entity type, `content_details` (row/col counts, cell accuracy), and `usage_info`.
- Optional `responses`: prompt/response/usage triplets when `--save_responses` is used.
- When `--save_log` is set, an additional log file stores every prompt, response, usage record, and evaluation alongside the summary.

## Prompting Rules
Prompts include:
- Format-specific instructions and an inlined golden example.
- Strict alignment rules (one-space padding per cell, vertically aligned delimiters).
- Required `<answer>...</answer>` tags; evaluation extracts the content between these tags.

## Extending
- Add new entity templates in `src/data_generator.py` to vary content.
- Integrate another backend by subclassing `BaseLLM` in `src/llm_interface.py` and registering it in `LLMInterface.create`.
- Adjust scoring by editing `content_weight`, `alignment_weight`, or the content extraction logic in `src/evaluator.py` and `src/utils.py`.

## Result Analysis
Compare multiple runs:
```bash
pip install matplotlib  # if not already installed
python examples/analyze_results.py results_openai.json results_deepseek.json
```
This prints a score table and saves `model_comparison.png`.
