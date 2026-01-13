# Context-Aware Redaction Benchmark

Evaluate how well LLMs can identify and mask sensitive numbers (phones, ID cards, credit cards) in realistic text while preserving the surrounding context.

## What the task checks
- 18-digit ID card: keep first 6 and last 2 digits, mask the middle 10 with `*`.
- Phone number formatted as `+<area> <11-digit>`: keep the plus sign and area code, then the first 3 and last 4 digits of the number, mask the middle 4.
- Credit card: keep the first 6 and last 4 digits, mask all middle digits (spaces/dashes are removed before masking).
- Detection patterns (see `src/utils.py`): phone numbers require a leading `+` and space, credit cards must include separators (spaces/dashes) and 13–19 digits total, ID cards are exactly 18 consecutive digits.

## Repository layout
```
config/config.yaml        # Default test, data-generation, and model settings
data/generated/           # Pre-generated sample datasets
main.py                   # Benchmark runner
generate_data.py          # Synthetic data generator (OpenAI API)
src/
  benchmark.py            # Prompting, orchestration, aggregation
  evaluator.py            # Number-level metrics + leakage checks
  data_generator.py       # Placeholder prompts + ground-truth creation
  models/                 # VLLM and API backends
scripts/                  # Optional plotting/HTML viewers for results
results/                  # Example result and summary JSON files
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```
VLLM requires a GPU-capable environment. Visualization helpers in `scripts/` also need `pandas`, `matplotlib`, and `seaborn` (install separately if you plan to plot results).

## Datasets
Start with the JSONs in `data/generated/` or create new ones.

### Generate new data (OpenAI API)
```bash
export OPENAI_API_KEY=sk-...
python generate_data.py \
  --num_samples 90 \
  --difficulty easy medium hard \
  --output dataset.json \
  --verbose
```
CLI difficulty flags map to internal lengths: `easy→short`, `medium→medium`, `hard→long`. Each generated context must include at least 3/4/6 placeholders respectively, covering `[PHONE]`, `[ID_CARD]`, `[CREDIT_CARD]` with combinations sampled for coverage. The generator:
- Asks an OpenAI model (default `gpt-3.5-turbo`) to write realistic text containing the placeholders.
- Fills placeholders with synthetic numbers (`+<area> <11-digit>` phones, 18-digit IDs, spaced credit cards).
- Builds ground-truth redactions using the same masking rules as evaluation.
- Saves a list of samples containing `id`, `difficulty`, `original_context`, `redacted_context`, `sensitive_info` (type/value/start/end), `sensitive_types`, and `num_sensitive_items`.

Generation defaults live under `data_generation` in `config/config.yaml` (model, temperature, output directory, samples per difficulty). When `--num_samples` is provided, it is divided evenly across the selected difficulties (remainder goes to the first difficulty).

## Running the benchmark
CLI entrypoint (see `main.py` for all arguments):
```bash
# OpenAI-compatible API (GPT, DeepSeek, etc.)
python main.py \
  --model_type api \
  --model_name gpt-4 \
  --api_key $OPENAI_API_KEY \
  --dataset data/generated/dataset_medium.json \
  --num_samples 100 \
  --verbose

# Local model via VLLM
python main.py \
  --model_type vllm \
  --model_name meta-llama/Llama-3-8b-instruct \
  --dataset data/generated/dataset_medium.json \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 1
```
Key flags:
- `--config`: YAML with defaults (test sizes, model params, API timeouts, VLLM settings).
- `--model_type`: `api` (OpenAI-compatible; auto-switches to Responses API for `gpt-5`/`o*`) or `vllm` for local models.
- `--model_name`: Model ID or path; `--base_url` sets a custom OpenAI-compatible endpoint (e.g., DeepSeek).
- `--dataset`: Path to dataset JSON; `--num_samples` truncates if set, otherwise falls back to `test.num_samples` in config.
- VLLM overrides: `--gpu_memory_utilization`, `--tensor_parallel_size`, `--enforce_eager/--no_enforce_eager`.
- `--output_dir`: Destination for timestamped result and summary JSON files.

## Metrics and evaluation
Evaluation is number-level binary classification plus leakage detection (see `src/evaluator.py`):
- `exact_match`: Prediction matches ground truth after whitespace normalization.
- `precision/recall/F1`: Computed over sensitive numbers in the original text; TP means the expected masked form is present and the raw number is not leaked.
- `leakage_rate`: Fraction of samples where any original number appears.
- `avg_leakage_ratio`: Average of (leaked count / sensitive count) per sample.
Metrics are reported overall and by difficulty. API backends also track token usage (input/output/reasoning) and aggregate averages by difficulty.

## Outputs
Two JSON files are written to `results/` (timestamped using the model name):
- `results_*.json`: `overall_metrics`, `difficulty_metrics`, `token_usage_stats`, and `detailed_results` per sample (`sample_id`, `original_text`, `ground_truth`, `model_output`, extracted `prediction`, evaluation breakdown, token usage).
- `summary_*.json`: Compact slice with model, dataset path, total samples, and aggregate metrics.
Optional exploration: `scripts/scatter.py` (accuracy vs token cost plots) or `scripts/visualize_response.html` (Chart.js viewer).

## Configuration guide
`config/config.yaml` controls:
- `test`: default sample counts and difficulty ordering.
- `sensitive_info.types`: categories the evaluator searches for.
- `models.vllm` / `models.api`: generation settings, timeouts, max tokens, reasoning effort for o-series models, optional `base_url`.
- `evaluation.metrics`: list of tracked metrics.

## Example prompt/result pair
Input  
`Customer registration: John Doe, ID: 123456789012345678, Phone: +86 18355132086, Credit Card: 1234 5678 9012 3456`

Expected redaction  
`Customer registration: John Doe, ID: 123456**********78, Phone: +86 183****2086, Credit Card: 123456******3456`
