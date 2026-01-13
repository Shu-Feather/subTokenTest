# Map Navigation Benchmark

A light-weight benchmark for probing spatial reasoning in 2D grid maps. It covers two small environments (Sokoban and FrozenLake), generates structured Q&A tasks automatically, and scores model outputs with exact match.

## What Is Included
- Task generation for Sokoban (deterministic walls/box) and FrozenLake (stochastic holes) with configurable sizes.
- Five task primitives that exercise element lookup, localization, neighborhood inspection, relative offsets, and counting.
- Evaluation harness that works with API models (OpenAI/DeepSeek) or local vLLM deployments, plus optional detailed logging.
- Result aggregation helper for comparing runs across models and task types.

## Task & Map Specification
Coordinate system uses the top-left corner as `(0, 0)`, `x` grows to the right, `y` grows downward. Maps are stored as space-separated rows in a string.

**Sokoban**
- Elements: `#` wall (borders enforced), `_` empty, `P` player (single, not on border), `X` box (single), `O` goal (single).
- Tasks:
  1. Element at coordinate `(x, y)`.
  2. Coordinate of `P`, `X`, or `O` (`(x, y)`).
  3. Eight neighbors of `P` as JSON keyed by directions.
  4. Relative offset from `X` to `O` (`(dx, dy)`).

**FrozenLake**
- Elements: `_` ice, `O` hole (multiple), `P` start (single, not on border), `G` goal (single).
- Tasks:
  1. Element at coordinate `(x, y)`.
  2. Coordinate of `P` or `G` (`(x, y)`).
  3. Eight neighbors of `P` as JSON.
  4. Relative offset from `P` to `G` (`(dx, dy)`).
  5. Number of holes (`O`) as an integer string.

## Dataset Format
Generated JSON files contain metadata and a flat list of tasks:
```json
{
  "metadata": {
    "env_type": "sokoban",
    "map_size": 8,
    "num_maps": 50,
    "tasks_per_type": {"1": 5, "2": 3, "3": 2, "4": 2},
    "seed": 42,
    "total_tasks": 600
  },
  "data": [
    {
      "env_type": "sokoban",
      "map_id": 0,
      "map_size": 8,
      "map": "# # # # # # # #\n# _ _ _ _ _ _ #\n# _ # _ _ # _ #\n# _ _ X P _ _ #\n# _ _ _ _ # _ #\n# _ _ _ O _ _ #\n# _ _ _ _ _ _ #\n# # # # # # # #",
      "task_type": 1,
      "question": "What element is at coordinates (2, 1)?",
      "answer": "_",
      "coordinates": {"x": 2, "y": 1}
    }
  ]
}
```
Answers in the dataset always match the format expected from models (e.g., `(x, y)` tuples, JSON strings, integer strings).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# Optional for result aggregation
pip install pandas
```
API evaluation expects `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` in the environment (or pass `--api-key`).

## Generate Data
Sokoban (8x8, default task mix, reproducible seed):
```bash
python -m generators.sokoban_generator \
  --size 8 --num-maps 50 --output data/sokoban_8x8.json --seed 42
```
FrozenLake (8x8, five holes per map):
```bash
python -m generators.frozenlake_generator \
  --size 8 --num-holes 5 --num-maps 50 \
  --output data/frozenlake_8x8.json --seed 42
```
Adjust `--tasks-type-*` flags to change per-map sampling counts.

## Run Evaluation
API model (auto-detected config for known names):
```bash
export OPENAI_API_KEY=...
python main.py \
  --model gpt-4o \
  --data data/sokoban_8x8.json \
  --output results/gpt4o_sokoban.json \
  --save-response results/gpt4o_sokoban_detailed.json \
  --verbose
```
Local vLLM model:
```bash
python main.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --model-type vllm \
  --data data/frozenlake_8x8.json \
  --output results/llama3_8b_frozenlake.json \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.9
```
Useful flags:
- `--temperature` and `--max-tokens` control generation; `--reasoning-effort` applies to o-series models.
- `--save-response` stores per-task prompts, raw responses, parsed answers, correctness, and token usage for later analysis.

Outputs are printed as overall accuracy plus breakdowns by environment, task type, and (env, task type); results JSON mirrors these metrics.

## Aggregate Results
Combine multiple detailed response files (requires pandas):
```bash
python scripts/aggregate_results.py \
  --directory results \
  --pattern "*_detailed.json" \
  --output-csv results/summary.csv \
  --output-md results/summary.md
```
This computes overall accuracy, per-environment and per-task-type accuracy, and average token statistics per model.

## Testing
```bash
python -m pytest tests -v
```
Integration tests cover data generation, parsing, and metric computation to ensure the pipeline stays stable.

## Repository Map
- `main.py` – evaluation entrypoint.
- `generators/` – map and task generators (`sokoban_generator.py`, `frozenlake_generator.py`).
- `prompts/` – system/user prompts with `<answer>...</answer>` tagging instructions.
- `models/` – API and vLLM interfaces.
- `evaluators/` – exact-match scorer and metrics aggregation.
- `utils/` – answer parsing and logging helpers.
- `scripts/aggregate_results.py` – cross-run comparison utilities.
