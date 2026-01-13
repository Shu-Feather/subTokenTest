# Linear Probe Experiments

Character-count (bag-of-characters) linear probes for language-model hidden states. Probes are trained per layer to predict how many times each character appears in a word, using the last non-padding token representation from every transformer block and the embedding layer.

## Layout
- `experiments/`: runnable scripts
  - `number_probe.py`: train per-layer probes and optionally cache hidden states.
  - `probe_infer.py`: run a saved probe on a word list and write a QA-style report.
  - `mutual_predict.py`: evaluate probes trained on normal words against perturbed/random/special variants from the same test split.
  - `data_prep.py`: build perturbed/random/special word lists from a base list.
  - `plot_suite.py`: plot EM/Macro-F1 across layers for multiple experiments (incl. shuffle baselines).
  - `analysis_mutual.py`: plot EM/Macro-F1 for mutual-predict metrics (ignores shuffle baselines).
  - `analyze_reports.py`: summarize per-layer reports into JSON + plots; emits per-character stats.
- `linear_probe.py`: probe heads (`Prober`, `ProberNumber`, `ProberSeq`).
- `dataset.py`: generic `ProbeDataset` for exist/number/sequence targets with optional shuffle baseline.
- `normalizer.py`: confusable Cyrillic/Greek → ASCII mapping plus `build_char_stats`.
- `util.py`: `CrossEntropyLossWithPositionalWeights` (per-position weighting for count classes).
- `utils/`: tokenizer helpers (strip `|trim` from chat templates), parameter counters, DeepSpeed config samples.
- `data/`: word lists (`words.txt`, `words_perturbed.txt`, `words_random.txt`, `words_special.txt`). Defaults in the scripts assume files next to the code; pass explicit paths to use the files under `data/`.

## Data and labels
- Input: one word per line. Multi-subtoken words still use the **whole-word** label.
- Alphabets / preprocessing (`--dataset_type`):
  - `normal`: lowercase a–z.
  - `perturbed`: confusable characters normalized to ASCII then lowercased (see `normalizer.CONFUSABLE_TO_ASCII`).
  - `random`: random a–z with the same length distribution as the base list.
  - `special`: symbols from `_PGO#Xo+=.B*-@%&^`.
- `data_prep.py` generates perturbed/random/special lists from a base list (pass `--base_path` to point at `data/words.txt`).
- Labels are character-count vectors (`len(alphabet)`) truncated by `max_num_each_char` and optionally reshaped to `(batch, max_count+1, char_num)` for the number probe head.

## Hidden-state extraction
- Tokenization: `add_special_tokens=False`, padding enabled, pad token set to `eos_token`, chat template `|trim` removed.
- For each batch of words, the model returns hidden states for **embedding + all transformer layers**; the last non-padding position is gathered per sample.
- Shapes: `(num_layers+1, num_tokens, hidden_dim)` stored on CPU (`float32`).
- `--hidden_state_cache` stores/loads this tensor; mismatched shape or token count triggers recomputation. A `.meta.pt` file with `{num_layers,num_tokens,char2label,alphabet}` is also written when available.

## Training number probes (`experiments/number_probe.py`)
- One probe per requested layer (`--layers all` or comma-separated indices; layer 0 = embeddings).
- Probe head: `ProberNumber(input_dim, char_num*(max_count+1))` with optional hidden layers via `--probe_layers` (depth>1 inserts GELU MLP blocks).
- Loss: `CrossEntropyLossWithPositionalWeights`, weighting each (count, character) class by inverse frequency.
- Metrics: accuracy over characters, exact match (all counts correct), macro precision/recall/F1 per character.
- Split: single train/test split controlled by `--train_ratio` and `--seed`; dataloader/probe seeds offset by layer index for reproducibility.
- Baseline: `--shuffle_baseline` permutes hidden states once (labels fixed) before training.
- Outputs: metrics JSON (`--output_json`), optional per-layer reports (`--per_layer_report_dir/layer_{i}.txt`), saved probes (`--save_probe_dir/layer_{i}.pt`), optional W&B logging.

Example:
```bash
python -m experiments.linear_probe.experiments.number_probe \
  --model_path /path/to/model \
  --dataset_path experiments/linear_probe/data/words.txt \
  --dataset_type normal \
  --layers all \
  --epochs 200 \
  --batch_size 4096 \
  --lr 1e-4 \
  --hidden_state_cache experiments/linear_probe/experiments/cache.pt \
  --output_json experiments/linear_probe/experiments/metrics.json \
  --per_layer_report_dir experiments/linear_probe/experiments/reports \
  --save_probe_dir experiments/linear_probe/experiments/probe_ckpts \
  --device_map auto --dtype bfloat16
```

## Reusing probes
- **Quick inference** (`probe_infer.py`): run a saved `layer_{i}.pt` on a word list and emit a QA-style report with EM.
```bash
python -m experiments.linear_probe.experiments.probe_infer \
  --model_path /path/to/model \
  --probe_path experiments/linear_probe/experiments/probe_ckpts/layer_20.pt \
  --layer 20 \
  --word_list experiments/linear_probe/data/words.txt \
  --dataset_type normal \
  --hidden_state_cache experiments/linear_probe/experiments/cache.pt \
  --output_report experiments/linear_probe/experiments/probe_infer_report.txt
```
- **Cross-variant evaluation** (`mutual_predict.py`): take probes trained on normal words and evaluate the same checkpoints on perturbed/random/special variants of the **same test split**. Hidden states for each variant are freshly collected; reports are written per dataset.
```bash
python -m experiments.linear_probe.experiments.mutual_predict \
  --probe_dir experiments/linear_probe/experiments/probe_ckpts \
  --model_path /path/to/model \
  --dataset_path experiments/linear_probe/data/words.txt \
  --layers all \
  --train_ratio 0.9 --batch_size 4096 --seed 20250315 \
  --hidden_state_cache experiments/linear_probe/experiments/cache.pt \
  --per_layer_report_dir experiments/linear_probe/experiments/mutual_reports \
  --output_json experiments/linear_probe/experiments/mutual_metrics.json
```

## Analysis and plotting
- `analyze_reports.py`: parse `layer_{i}.txt` reports, compute EM + weighted macro F1 (character weights depend on dataset), dump `summary.json`, and draw EM/F1 vs. layer plots. Auto-detects dataset type from directory names unless overridden.
- `plot_suite.py`: given a base directory containing subfolders like `normal/metrics.json`, `normal_shuffle/metrics.json`, …, plot EM and Macro-F1 across layers (baseline vs. main curves share colors). Outputs `test_em_vs_layer_hy.pdf` and `test_macro_f1_vs_layer_hy.pdf`.
- `analysis_mutual.py`: plot Macro-F1/EM vs. layer for mutual-predict metrics, skipping shuffle datasets. Outputs `<prefix>_mf1.png` and `<prefix>_em.png`.

## Tips and gotchas
- Layer indexing includes the embedding layer as 0; transformer blocks start at 1 in the checkpoints.
- Pass explicit `--dataset_path` pointing at `data/*.txt`; the baked-in defaults assume files live alongside the code.
- Hidden-state cache must match **both** layer count and token count; otherwise it is recomputed automatically.
- Probe batch size is independent of hidden-state collection batch size; adjust both for memory.
- `--device_map none` keeps the base model on CPU; otherwise `auto` will spread across available GPUs. Dtype choices: `bfloat16` / `float16` / `float32`.
- `probe_type`/`dataset_type` in `mutual_predict.py` are fixed to `normal` for training probes; only the evaluation variants differ.
