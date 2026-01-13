# Tree Benchmark

Binary tree reasoning benchmark with two complementary tasks: answering structural questions about a rendered tree (Task 1) and recovering the path between two nodes (Task 2). Prompts include an ASCII drawing of the tree plus task instructions; the evaluation script runs models (OpenAI, DeepSeek, or local vLLM) and scores exactness and path similarity while tracking token usage when available.

## Repo Layout
- `main.py`: entry point for running evaluations; generates data on the fly or loads JSON test sets.
- `config/config.py`: shared runtime configuration (task mix, decoding params, vLLM knobs).
- `data/`: tree generation utilities (`TreeGenerator`, `DataManager`).
- `evaluation/`: automatic grading and similarity scoring for both tasks.
- `models/model_interface.py`: thin wrapper for OpenAI / DeepSeek APIs and local vLLM models.
- `datasets/`: curated JSON test suites by task and difficulty; ready to use with `--test_file`.
- `generate_tasks1.py`, `generate_tasks2.py`: offline generators to create new Task 1/2 datasets with metadata.

## Setup
- Python 3.9+ recommended.
- Install dependencies: `pip install -r requirements.txt`
- For API models, provide `--api_key` or rely on `OPENAI_API_KEY`; optionally set `--base_url` for custom endpoints.
- vLLM runs require a GPU and a local model path or Hugging Face repo id.

## Running the Benchmark
The runner can synthesize fresh trees or consume a JSON dataset. Results are written to `--output_dir` as `results_<task>_<source>_<timestamp>.json` and include per-sample judgments plus token tallies when supported.

### Evaluate with an API model (freshly generated data)
```
python main.py \
  --model_type openai \
  --model_name gpt-4o-mini \
  --api_key "$OPENAI_API_KEY" \
  --task_type both \
  --num_samples 50 \
  --verbose
```

### Evaluate using a pre-generated dataset
```
python main.py \
  --model_type deepseek \
  --model_name deepseek-chat \
  --api_key "$DEEPSEEK_API_KEY" \
  --task_type task1 \
  --test_file datasets/tree_task1_medium.json \
  --num_samples 100
```
`--test_file` expects either the metadata+`samples` format produced by the generators or a plain list of sample dicts.

### Evaluate a local vLLM model
```
python main.py \
  --model_type vllm \
  --model_name /path/to/model \
  --task_type task2 \
  --num_samples 20 \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 2
```
You can force or disable eager mode with `--enforce_eager` / `--no_enforce_eager`.

## Tasks and Data Format
- **Task 1 – Structure questions**: Given the tree ASCII art, answer parent/left-child/right-child existence or count nodes. Answers are single numbers or `None`.
- **Task 2 – Path analysis**: Return the path between two nodes as `a -> b -> ... -> z`.

Sample fields (from generated or curated JSON):
- `prompt`: complete instruction shown to the model (includes the rendered tree).
- `expected_answer`: canonical answer string.
- `task_type`: `task1` or `task2`; optional `difficulty`, `question_type`, `tree_depth`.
- For Task 2 samples, `path_analysis` carries the numeric path list and length.

Trees are rendered with `/` for left edges and `\` for right edges; spacing encodes depth.

## Generating New Test Suites
Use the dedicated generators to build reproducible datasets with metadata and distributions.
- Task 1 (structure QA):  
  `python generate_tasks1.py --difficulty medium --num_samples 200 --output_dir datasets`
- Task 2 (path finding):  
  `python generate_tasks2.py --difficulty hard --threshold 4 --num_samples 150 --output_dir datasets`

Outputs include per-sample prompts and statistics (depth, question/path length distributions).

## Evaluation Details
- Task 1: Extracts the line after `### My answer is:` and checks exact match (numeric or `None` aliases).
- Task 2: Normalizes the returned path (`->` separators, digits only) and scores both exactness and a similarity metric (sequence match + endpoint correctness).
- Token accounting: when using OpenAI/DeepSeek APIs, total/input/output/reasoning tokens are captured per sample and summarized across tasks and difficulties.

## Tips
- Set `--max_depth` to control how large generated trees get when not using a dataset.
- Increase `--num_samples` for more stable accuracy estimates; combine both tasks with `--task_type both`.
- Inspect saved JSON results for mispredictions; verbose mode prints prompts, model outputs, and token usage for each sample.
