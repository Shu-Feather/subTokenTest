"""
Infer character-count probes on a given word list and write a QA-style report.

Layer indexing: 0 = embedding layer, 1..N = transformer blocks (matches number_probe).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..experiments.number_probe import (
    NumberProbeDataset,
    build_labels,
    collect_hidden_states,
    get_label_builder,
    load_words,
)
from ..linear_probe import ProberNumber
from ..utils.tokenizer_utils import tokenizer_remove_trim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained number probe on a word list.")
    parser.add_argument("--model_path", type=Path, required=True, help="Base LLM path.")
    parser.add_argument("--probe_path", type=Path, required=True, help="Path to probe weights (e.g., layer_0.pt).")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to probe (0 = embedding).")
    parser.add_argument("--word_list", type=Path, required=True, help="Path to word list (one word per line).")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["normal", "perturbed", "random", "special"],
        default="normal",
        help="Controls alphabet and preprocessing (should match training).",
    )
    parser.add_argument("--probe_layers", type=int, default=1, help="Probe depth (must match saved weights).")
    parser.add_argument(
        "--hidden_state_cache",
        type=Path,
        default=None,
        help="Optional cache for hidden states to avoid recomputation.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for loading the base model. Use 'auto' or 'cuda:0', etc.; 'none' for CPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the base model.",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        required=True,
        help="Where to write the QA-style report txt.",
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


def format_counts(counts: torch.Tensor, alphabet: str) -> str:
    return ", ".join(f"{ch}:{int(cnt)}" for ch, cnt in zip(alphabet, counts.tolist()))


def main() -> None:
    args = parse_args()

    words = load_words(args.word_list)
    alphabet, preprocess = get_label_builder(args.dataset_type)
    labels, _, max_count = build_labels(words, alphabet, preprocess)

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

    hidden_states = maybe_collect_states(args.hidden_state_cache, model, tokenizer, words, input_device)
    total_layers = hidden_states.shape[0]
    if args.layer < 0 or args.layer >= total_layers:
        raise ValueError(f"Layer {args.layer} out of range (0, {total_layers - 1}).")

    layer_states = hidden_states[args.layer]  # (num_tokens, hidden_dim)

    dataset = NumberProbeDataset(layer_states, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probe = ProberNumber(
        input_dim=layer_states.shape[1],
        char_num=labels.shape[1],
        max_num_each_char=max_count,
        layer=args.probe_layers,
        hidden_dim=layer_states.shape[1],
    )
    state = torch.load(args.probe_path, map_location="cpu")
    probe.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)
    probe.eval()

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for states_batch, _ in dataloader:
            states_batch = states_batch.to(device)
            logits = probe(states_batch)
            preds.append(logits.argmax(dim=1).cpu())
    preds_tensor = torch.cat(preds, dim=0)  # (num_tokens, char_num)
    em_hits = (preds_tensor == labels).all(dim=1)
    em_sum = int(em_hits.sum().item())
    total = len(words)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        f.write(f"Layer {args.layer} probe report\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Probe: {args.probe_path}\n")
        f.write(f"Words: {args.word_list}\n")
        f.write(f"Alphabet: {alphabet}\n\n")
        f.write(f"Exact Match: {em_sum} / {total} = {em_sum / total:.4f}\n\n")
        for idx, word in enumerate(words):
            pred_counts = preds_tensor[idx]
            target_counts = labels[idx]
            f.write(f"Q: {word}\n")
            f.write(f"Pred: {format_counts(pred_counts, alphabet)}\n")
            f.write(f"Target: {format_counts(target_counts, alphabet)}\n\n")


if __name__ == "__main__":
    main()
