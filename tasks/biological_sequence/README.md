# Biological Sequence Manipulation Benchmark

Evaluate language models on core molecular biology sequence tasks with reproducible data generation, rich prompts, and detailed accuracy/cost reporting.

## Scope
- DNA complement: produce the complementary DNA strand.
- RNA complement: produce the complementary RNA strand.
- Protein 3-letter → 1-letter: `GLY-ARG-PHE` → `GRF` (no separators in the output).
- Protein 1-letter → 3-letter: `GRF` → `GLY-ARG-PHE` (hyphen-separated output).

Prompts include the pairing/conversion tables and ask the model to return the final answer inside `<answer>...</answer>` tags; the evaluator strips and normalizes outputs before exact-match scoring.

## Highlights
- Turnkey runners for OpenAI/DeepSeek APIs, vLLM (chat or raw), HuggingFace, and Ollama via `src/model_interface.py`.
- Automatic, biologically valid dataset generation with adjustable lengths and case counts; optionally load pre-made JSON datasets from `datasets/`.
- Token-usage aggregation (prompt/completion/reasoning/output) saved alongside accuracy for every model and task.
- Rich artifacts per run: per-task CSVs, a human-readable report, metrics JSON, and cross-model comparison when multiple models are evaluated.
- Tested utilities and prompts (`tests/`), plus quick configs for fast smoke tests (`configs/biological_sequence/gpt_quick.json`, `configs/biological_sequence/deepseek_quick.json`).

## Repository Layout
- `main.py` — CLI entrypoint for benchmarking configured models.
- `src/` — generators, prompts, evaluators, and model adapters.
- `configs/biological_sequence/` — task/model configs (edit placeholders like local model paths).
- `datasets/` — sample pre-generated datasets (easy/medium/hard).
- `generate_datasets.py` — standalone dataset generator.
- `examples/run_benchmark.py` — small script with quick-test options.
- `results/` — default output directory (created on demand).
- `scripts/scatter.py` — visualize accuracy vs. token cost from saved results.

## Setup
1. Use a modern Python 3 environment (virtualenv/conda recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Open-source model backends (vLLM/HuggingFace/Ollama) need their respective runtimes and weights; the API-based flows only require `openai` + keys.
3. Export API keys as needed:
   ```bash
   export OPENAI_API_KEY=...
   export DEEPSEEK_API_KEY=...
   ```

## Run the Benchmark (CLI)
Pick or edit a config in `configs/biological_sequence/` and point the runner at models defined there (keys under `"models"`):
```bash
python main.py \
  --config configs/biological_sequence/gpt_quick.json \
  --models gpt-4 o4-mini \
  --output-dir results \
  --verbose
```

Useful flags:
- `--models`: subset of model keys from the config.
- `--tasks`: limit to specific tasks, e.g. `--tasks dna_complement protein_three_to_one`.
- `--load-data`: reuse an existing dataset JSON instead of generating new cases (supports `datasets/bio_seq_dataset_easy.json` and files from `generate_datasets.py`).
- `--output-dir`: where artifacts are written (default `results`).
- `--no-save`: run and print summaries without writing files.
- `--verbose`: print prompts, raw responses, and token usage per case.

Example: run a single local vLLM chat model defined in your config (update the path/model id first):
```bash
python main.py --config configs/biological_sequence/model_config.json --models llama-3-8b-instruct --verbose
```

`examples/run_benchmark.py` offers a `--quick-test` mode that shrinks case counts for a fast sanity check.

## Generate Datasets Only
Produce reusable JSON datasets without calling any model:
```bash
python generate_datasets.py \
  --num-cases 50 \
  --min-length 8 --max-length 15 \
  --tasks dna_complement rna_complement protein_three_to_one protein_one_to_three \
  --output-dir datasets \
  --pretty
```
Use `--split` to write one file per task, `--seed` for reproducibility, and `--quiet/--verbose` to control logging.

## Programmatic Use
```python
from src.benchmark import BiologicalSequenceBenchmark
from src.model_interface import ModelFactory

benchmark = BiologicalSequenceBenchmark("configs/biological_sequence/gpt_quick.json")
benchmark.verbose = True

model = ModelFactory.create_model_interface(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
)

test_data = benchmark.generate_test_data()
results = benchmark.run_single_model(model, test_data, verbose=True)
print(benchmark.evaluator.generate_detailed_report(
    [r for cases in results.values() for r in cases]
))
```

## Configuration Reference
- Tasks live under `"tasks"`: `enabled`, `num_cases`, and `sequence_length_range` control what is generated.
  ```json
  "dna_complement": { "enabled": true, "num_cases": 100, "sequence_length_range": [8, 20] }
  ```
- Models live under `"models"`: each entry declares a `provider` (one of `openai`, `deepseek`, `huggingface`, `ollama`, `vllm`, `vllm_chat`), a `model_name`, and optional parameters.
  ```json
  "gpt-4": {
    "provider": "openai",
    "model_name": "gpt-4",
    "parameters": { "temperature": 0.1, "max_tokens": 1024 }
  },
  "llama-3-8b-instruct": {
    "provider": "vllm_chat",
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "parameters": {
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.9,
      "temperature": 0.1,
      "max_tokens": 512,
      "system_message": "You are a knowledgeable biology assistant."
    }
  }
  ```
  Update local model paths and provider names to match `src/model_interface.py`; placeholder paths in configs will not run as-is.

## Outputs
Each model run produces `results/<model_key>_<timestamp>/` containing:
- `<task>_results.csv` — per-case correctness and token counts.
- `detailed_results.json` — raw responses (optional), normalized outputs, and usage info.
- `metrics.json` — accuracy, per-task breakdown, and aggregated token statistics.
- `benchmark_report.txt` — human-readable summary plus usage tables.
When more than one model is evaluated in a single run, `model_comparison_<timestamp>.txt` is written at the top level of the results directory.

## Prompting & Evaluation
- Prompts in `src/prompt_templates.py` embed the necessary nucleotide/amino-acid rules and ask for `<answer>...</answer>` wrapping.
- `SequenceEvaluator` normalizes case/spacing, computes exact-match correctness, and derives a confidence score from similarity when mismatched.
- Token usage (including reasoning tokens for supported providers) is aggregated per task and overall.

## Testing
Run the test suite from the project root:
```bash
pytest
```
