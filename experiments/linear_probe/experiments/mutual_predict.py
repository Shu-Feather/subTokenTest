"""
Run saved normal probes on a normal dataset split, then test the same probes
on normal / perturbed / random / special variants of the test split.

Example:
python -m experiments.linear_probe.experiments.mutual_predict \
  --probe_dir /path/to/normal/probe_ckpts \
  --model_path /path/to/base/model \
  --dataset_path experiments/linear_probe/datasets/words.txt \
  --device_map auto \
  --dtype bfloat16 \
  --batch_size 4096 \
  --seed 20250315 \
  --per_layer_report_dir /path/to/mutual_reports \
  --output_json /path/to/mutual_metrics.json \
  --hidden_state_cache /path/to/cache.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..experiments.number_probe import (
    NumberProbeDataset,
    build_labels,
    collect_hidden_states,
    evaluate,
    get_label_builder,
    load_words,
    parse_layers,
    split_indices,
)
from ..experiments.data_prep import perturb_words, random_words, special_words
from ..linear_probe import ProberNumber
from ..utils.tokenizer_utils import tokenizer_remove_trim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mutual predict with saved probes on a dataset split.")
    parser.add_argument("--probe_dir", type=Path, required=True, help="Directory containing layer_{idx}.pt probes.")
    parser.add_argument("--model_path", type=Path, required=True, help="Base model path (needed for hidden states).")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Word list path to probe.")
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices or 'all'. Layer 0 = embedding.",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20250315)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument(
        "--hidden_state_cache",
        type=Path,
        default=None,
        help="Optional cache for hidden states (recomputed if mismatched).",
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--per_layer_report_dir",
        type=Path,
        required=True,
        help="Directory to write layer_{idx}.txt reports.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Where to write aggregated metrics.",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=0.3,
        help="Probability of perturbing each ASCII character when building perturbed test data (matches data_prep.py).",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="normal",
        help="Kept for backward compatibility; must be 'normal' for this script.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="normal",
        help="Kept for backward compatibility; must be 'normal' for this script.",
    )
    return parser.parse_args()


def maybe_collect_states(
    cache_path: Path | None,
    model: AutoModelForCausalLM,
    tokenizer,
    words: Sequence[str],
    input_device: torch.device,
) -> torch.Tensor:
    expected_layers = model.config.num_hidden_layers + 1
    expected_tokens = len(words)
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.dim() == 3 and cached.shape[0] == expected_layers and cached.shape[1] == expected_tokens:
            return cached
        else:
            print(
                f"Ignoring hidden state cache (shape {tuple(cached.shape)}), "
                f"expect ({expected_layers}, {expected_tokens}, hidden_dim); recomputing."
            )
    states = collect_hidden_states(model, tokenizer, words, input_device)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(states, cache_path)
    return states


def write_report(
    path: Path,
    words: Sequence[str],
    preds: torch.Tensor,
    targets: torch.Tensor,
    alphabet: str,
    test_idx: Sequence[int],
) -> None:
    def fmt_counts(counts: torch.Tensor) -> str:
        return ", ".join(f"{ch}:{int(cnt)}" for ch, cnt in zip(alphabet, counts.tolist()))

    lines = [f"===== {path.stem} Test Report =====\n"]
    for local_idx, global_idx in enumerate(test_idx):
        word = words[global_idx]
        pred_counts = preds[local_idx]
        target_counts = targets[local_idx]
        lines.append(f"Q: {word}\n")
        lines.append(f"Model: {fmt_counts(pred_counts)}\n")
        lines.append(f"Target: {fmt_counts(target_counts)}\n\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.probe_type != "normal":
        raise ValueError("mutual_predict currently supports only normal probes (probe_type must be 'normal').")
    if args.dataset_type != "normal":
        raise ValueError("mutual_predict currently expects a normal dataset (dataset_type must be 'normal').")

    # Load base words and labels using normal alphabet/preprocess
    words = load_words(args.dataset_path)
    alphabet_probe, preprocess_normal = get_label_builder("normal")
    labels_full, _, _ = build_labels(words, alphabet_probe, preprocess_normal)

    # Split indices
    train_idx, test_idx = split_indices(len(words), args.train_ratio, args.seed)
    test_words_normal = [words[i] for i in test_idx]
    labels_test_normal = labels_full.index_select(0, torch.tensor(test_idx))

    # Build derived test sets from the normal test split
    rng_perturb = random.Random(args.seed)
    rng_random = random.Random(args.seed + 1)
    rng_special = random.Random(args.seed + 2)
    test_words_perturbed = perturb_words(test_words_normal, args.perturb_ratio, rng_perturb)
    test_words_random = random_words(test_words_normal, rng_random)
    test_words_special = special_words(test_words_normal, rng_special)

    preprocess_perturbed = get_label_builder("perturbed")[1]
    labels_test_perturbed, _, _ = build_labels(test_words_perturbed, alphabet_probe, preprocess_perturbed)
    labels_test_random, _, _ = build_labels(test_words_random, alphabet_probe, preprocess_normal)
    labels_test_special, _, _ = build_labels(test_words_special, alphabet_probe, preprocess_normal)

    # Prepare model/tokenizer and hidden states
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer_remove_trim(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    device_map = None if args.device_map.lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=dtype_map[args.dtype],
    )
    input_device = model.get_input_embeddings().weight.device

    hidden_states_normal = maybe_collect_states(args.hidden_state_cache, model, tokenizer, words, input_device)
    total_layers = hidden_states_normal.shape[0]
    layer_indices = parse_layers(args.layers, total_layers)

    # Prepare hidden states for each test variant
    test_states_normal = hidden_states_normal[:, test_idx, :]
    test_states_perturbed = collect_hidden_states(model, tokenizer, test_words_perturbed, input_device)
    test_states_random = collect_hidden_states(model, tokenizer, test_words_random, input_device)
    test_states_special = collect_hidden_states(model, tokenizer, test_words_special, input_device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_states = {
        "normal": test_states_normal,
        "perturbed": test_states_perturbed,
        "random": test_states_random,
        "special": test_states_special,
    }
    dataset_labels = {
        "normal": labels_test_normal,
        "perturbed": labels_test_perturbed,
        "random": labels_test_random,
        "special": labels_test_special,
    }
    dataset_words = {
        "normal": test_words_normal,
        "perturbed": test_words_perturbed,
        "random": test_words_random,
        "special": test_words_special,
    }

    results: dict[str, list[dict[str, float | int]]] = {k: [] for k in dataset_states}

    for layer_idx in layer_indices:
        ckpt_path = args.probe_dir / f"layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"Skip layer {layer_idx}: checkpoint not found at {ckpt_path}")
            continue

        state = torch.load(ckpt_path, map_location="cpu")
        if "inner_model.layers.0.weight" not in state:
            raise RuntimeError(f"Checkpoint {ckpt_path} missing inner_model.layers.0.weight")
        output_dim, input_dim = state["inner_model.layers.0.weight"].shape
        if output_dim % len(alphabet_probe) != 0:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} output_dim {output_dim} not divisible by alphabet size {len(alphabet_probe)}"
            )
        max_count_probe = output_dim // len(alphabet_probe) - 1

        probe = ProberNumber(
            input_dim=input_dim,
            char_num=len(alphabet_probe),
            max_num_each_char=max_count_probe,
            layer=1,
            hidden_dim=input_dim,
        )
        probe.load_state_dict(state)
        probe.to(device)

        for dataset_name in ["normal", "perturbed", "random", "special"]:
            states_tensor = dataset_states[dataset_name][layer_idx]
            labels_tensor = dataset_labels[dataset_name]

            dataset = NumberProbeDataset(states_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            # Predict for reports (probe alphabet)
            preds_list = []
            targets_list = []
            with torch.no_grad():
                probe.eval()
                for states_batch, labels_batch in dataloader:
                    states_batch = states_batch.to(device)
                    logits = probe(states_batch)
                    preds_list.append(logits.argmax(dim=1).cpu())
                    targets_list.append(labels_batch)
            preds = torch.cat(preds_list, dim=0)
            targets = torch.cat(targets_list, dim=0)

            # Evaluate
            acc, em, mp, mr, mf1 = evaluate(probe, dataloader, device)

            # Report
            report_path = args.per_layer_report_dir / dataset_name / f"layer_{layer_idx}.txt"
            write_report(report_path, dataset_words[dataset_name], preds, targets, alphabet_probe, range(len(dataset_words[dataset_name])))

            results[dataset_name].append(
                {
                    "layer": layer_idx,
                    "test_acc": acc,
                    "test_em": em,
                    "test_mp": mp,
                    "test_mr": mr,
                    "test_mf1": mf1,
                }
            )
            print(f"[{dataset_name}][Layer {layer_idx:02d}] test_acc={acc:.4f}, test_em={em:.4f}, test_mf1={mf1:.4f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "probe_dir": str(args.probe_dir),
                "model_path": str(args.model_path),
                "dataset_path": str(args.dataset_path),
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "perturb_ratio": args.perturb_ratio,
                "results": results,
                "num_samples": {k: len(v) for k, v in dataset_words.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
