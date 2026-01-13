"""
Train number (bag-of-characters) linear probes on last-token hidden states.

Workflow:
1. Load a word list (normal / perturbed / random / special).
2. Build bag-of-characters labels (alphabet depends on dataset type).
3. Run the language model once to collect hidden states of the last token for every layer.
4. For each layer, train a ProberNumber and report train/test accuracy and exact-match rate.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import BASE_DIR
from ..linear_probe import ProberNumber
from ..util import CrossEntropyLossWithPositionalWeights
from ..normalizer import normalize_confusables
from ..utils.tokenizer_utils import tokenizer_remove_trim

ASCII_ALPHABET = string.ascii_lowercase
SPECIAL_ALPHABET = "_PGO#Xo+=.B*-@%&^"


class NumberProbeDataset(Dataset):
    def __init__(self, states: torch.Tensor, labels: torch.Tensor):
        assert states.shape[0] == labels.shape[0], "States and labels must align on batch dimension."
        self.states = states
        self.labels = labels

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.labels[idx]


def load_words(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_indices(n: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * train_ratio)
    return indices[:split], indices[split:]


def build_labels(
    words: Sequence[str],
    alphabet: str,
    preprocess: Callable[[str], str],
) -> tuple[torch.Tensor, dict[str, int], int]:
    char2label = {ch: i for i, ch in enumerate(alphabet)}
    labels = torch.zeros((len(words), len(alphabet)), dtype=torch.long)
    max_count = 0
    for i, word in enumerate(words):
        token = preprocess(word)
        for ch in token:
            if ch in char2label:
                labels[i, char2label[ch]] += 1
        max_count = max(max_count, int(labels[i].max().item()))
    return labels, char2label, max_count


def compute_pos_weight(labels: torch.Tensor, max_count: int, char_num: int) -> torch.Tensor:
    pos_nums = torch.ones((max_count + 1, char_num))
    for label in labels:
        for char_idx, count in enumerate(label.tolist()):
            pos_nums[count, char_idx] += 1
    pos_nums /= len(labels)
    pos_weight = 1 / pos_nums
    return torch.clamp(pos_weight, min=1.0, max=100.0)


def collect_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer,
    words: Sequence[str],
    input_device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    hidden_states: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            batch_words = words[start : start + batch_size]
            encoded = tokenizer(
                batch_words,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {k: v.to(input_device) for k, v in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            hs = outputs.hidden_states  # tuple length num_layers + 1, [0] is embedding
            last_indices = (encoded["attention_mask"].sum(dim=1) - 1).to(input_device)
            gathered = []
            for layer_hidden in hs:  # include embedding as layer 0
                # layer_hidden: (batch, seq_len, hidden_dim)
                reps = layer_hidden[torch.arange(layer_hidden.size(0), device=input_device), last_indices]
                gathered.append(reps.detach().cpu())
            hidden_states.append(torch.stack(gathered, dim=0))  # (layers, batch, hidden_dim)
    stacked = torch.cat(hidden_states, dim=1)  # (layers, total_samples, hidden_dim)
    return stacked.to(torch.float32)


def _maybe_recollect_states(
    cached: torch.Tensor | None,
    model: AutoModelForCausalLM,
    tokenizer,
    words: Sequence[str],
    input_device: torch.device,
    cache_path: Path | None,
) -> torch.Tensor:
    """
    Load cached states if they match the current dataset/model layout; otherwise recompute.
    """
    expected_layers = model.config.num_hidden_layers + 1  # embedding + transformer blocks
    expected_tokens = len(words)
    need_recollect = cached is None
    reason: str | None = None

    if cached is not None:
        if cached.dim() != 3:
            need_recollect, reason = True, f"cached dim={cached.dim()} (expect 3)"
        elif cached.shape[0] != expected_layers:
            need_recollect, reason = True, f"cached layers={cached.shape[0]} (expect {expected_layers})"
        elif cached.shape[1] != expected_tokens:
            need_recollect, reason = True, f"cached samples={cached.shape[1]} (expect {expected_tokens})"

    if need_recollect:
        if reason:
            print(f"Ignoring hidden state cache ({reason}); recomputing states.")
        cached = collect_hidden_states(model, tokenizer, words, input_device)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cached, cache_path)
            meta_path = cache_path.with_suffix(".meta.pt")
            torch.save({"num_layers": expected_layers, "num_tokens": expected_tokens}, meta_path)

    return cached


def parse_layers(layer_str: str, total_layers: int) -> list[int]:
    if layer_str.lower() == "all":
        return list(range(total_layers))
    return [int(x.strip()) for x in layer_str.split(",") if x.strip()]


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float, float | None, float | None, float | None]:
    model.eval()
    total_correct = 0
    total_tokens = 0
    exact_matches = 0
    total_samples = 0
    # Macro metrics per character across count classes
    per_char_confusion = None
    with torch.no_grad():
        for states, labels in dataloader:
            states = states.to(device)
            labels = labels.to(device)
            logits = model(states)
            preds = logits.argmax(dim=1).cpu()
            labels_cpu = labels.cpu()
            total_correct += (preds == labels_cpu).sum().item()
            total_tokens += labels_cpu.numel()
            exact_matches += (preds == labels_cpu).all(dim=1).sum().item()
            total_samples += labels_cpu.size(0)
            # accumulate confusion per character
            if per_char_confusion is None:
                num_classes = preds.max().item() + 1
                num_chars = preds.shape[1]
                per_char_confusion = torch.zeros(num_chars, num_classes, num_classes, dtype=torch.long)
            num_classes = per_char_confusion.shape[-1]
            # pad to current num_classes if needed
            current_max = max(preds.max().item(), labels_cpu.max().item())
            if current_max + 1 > num_classes:
                new_size = current_max + 1
                new_conf = torch.zeros(per_char_confusion.shape[0], new_size, new_size, dtype=torch.long)
                new_conf[:, :num_classes, :num_classes] = per_char_confusion
                per_char_confusion = new_conf
                num_classes = new_size
            # update confusion
            for char_idx in range(preds.shape[1]):
                p = preds[:, char_idx]
                t = labels_cpu[:, char_idx]
                per_char_confusion[char_idx].index_put_(
                    (t, p), torch.ones_like(t, dtype=torch.long), accumulate=True
                )
    acc = total_correct / total_tokens if total_tokens else 0.0
    em = exact_matches / total_samples if total_samples else 0.0
    # compute macro precision/recall/F1 per character, then average
    macro_precision = macro_recall = macro_f1 = None
    if per_char_confusion is not None:
        precisions = []
        recalls = []
        f1s = []
        for conf in per_char_confusion:
            tp = conf.diag()
            pred_pos = conf.sum(dim=0)
            true_pos = conf.sum(dim=1)
            precision = (tp / pred_pos.clamp(min=1)).mean().item()
            recall = (tp / true_pos.clamp(min=1)).mean().item()
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        macro_f1 = sum(f1s) / len(f1s)
    return acc, em, macro_precision, macro_recall, macro_f1


def train_one_layer(
    layer_idx: int,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    max_count: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    probe_layers: int,
    wandb_run=None,
    words: Sequence[str] | None = None,
    alphabet: str | None = None,
    save_dir=None,
    wandb_step_base: int = 0,
    seed_base: int | None = None,
    per_layer_report_dir=None,
) -> dict[str, float]:
    if seed_base is not None:
        torch.manual_seed(seed_base)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_base)

    train_states = hidden_states[layer_idx].index_select(0, torch.tensor(train_idx))
    test_states = hidden_states[layer_idx].index_select(0, torch.tensor(test_idx))
    train_labels = labels.index_select(0, torch.tensor(train_idx))
    test_labels = labels.index_select(0, torch.tensor(test_idx))

    rng_loader = torch.Generator().manual_seed(seed_base if seed_base is not None else torch.initial_seed())

    train_loader = DataLoader(
        NumberProbeDataset(train_states, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=rng_loader,
    )
    test_loader = DataLoader(
        NumberProbeDataset(test_states, test_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    model = ProberNumber(
        input_dim=train_states.shape[1],
        char_num=train_labels.shape[1],
        max_num_each_char=max_count,
        layer=probe_layers,
        hidden_dim=train_states.shape[1],
    ).to(device)

    pos_weight = compute_pos_weight(train_labels, max_count=max_count, char_num=train_labels.shape[1])
    loss_fn = CrossEntropyLossWithPositionalWeights(weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        seen = 0
        for states, label in train_loader:
            states = states.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            batch_size_eff = label.size(0)
            epoch_loss += loss.item() * batch_size_eff
            seen += batch_size_eff

        epoch_loss = epoch_loss / seen if seen else 0.0
        # Evaluate on training set for EM/acc tracking
        train_acc_epoch, train_em_epoch, train_mp_epoch, train_mr_epoch, train_mf1_epoch = evaluate(model, train_loader, device)
        model.train()  # reset back to training mode
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"layer_{layer_idx}/train_loss": epoch_loss,
                    f"layer_{layer_idx}/train_acc": train_acc_epoch,
                    f"layer_{layer_idx}/train_em": train_em_epoch,
                    f"layer_{layer_idx}/train_mp": train_mp_epoch,
                    f"layer_{layer_idx}/train_mr": train_mr_epoch,
                    f"layer_{layer_idx}/train_mf1": train_mf1_epoch,
                },
                step=wandb_step_base + epoch + 1,
            )

    train_acc, train_em, train_mp, train_mr, train_mf1 = evaluate(model, train_loader, device)
    test_acc, test_em, test_mp, test_mr, test_mf1 = evaluate(model, test_loader, device)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"layer_{layer_idx}.pt"
        torch.save(model.state_dict(), ckpt_path)

    if per_layer_report_dir is not None and words is not None and alphabet is not None:
        per_layer_report_dir.mkdir(parents=True, exist_ok=True)
        report_file = per_layer_report_dir / f"layer_{layer_idx}.txt"

        report_lines: list[str] = [f"===== Layer {layer_idx} Test Report =====\n"]
        pred_list = []
        target_list = []
        with torch.no_grad():
            model.eval()
            for states, label in test_loader:
                states = states.to(device)
                logits = model(states)
                pred_list.append(logits.argmax(dim=1).cpu())
                target_list.append(label.cpu())
        preds = torch.cat(pred_list, dim=0)
        targets = torch.cat(target_list, dim=0)

        def fmt_counts(counts: torch.Tensor) -> str:
            return ", ".join(f"{ch}:{int(cnt)}" for ch, cnt in zip(alphabet, counts.tolist()))

        for local_idx, global_idx in enumerate(test_idx):
            word = words[global_idx]
            pred_counts = preds[local_idx]
            target_counts = targets[local_idx]
            report_lines.append(f"Q: {word}\n")
            report_lines.append(f"Model: {fmt_counts(pred_counts)}\n")
            report_lines.append(f"Target: {fmt_counts(target_counts)}\n\n")

        with report_file.open("w", encoding="utf-8") as f:
            f.writelines(report_lines)

    return {
        "train_acc": train_acc,
        "train_em": train_em,
        "train_mp": train_mp,
        "train_mr": train_mr,
        "train_mf1": train_mf1,
        "test_acc": test_acc,
        "test_em": test_em,
        "test_mp": test_mp,
        "test_mr": test_mr,
        "test_mf1": test_mf1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Number (bag-of-characters) linear probe.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the base LLM.")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=BASE_DIR / "words_full_clean.txt",
        help="Path to the word list to probe.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["normal", "perturbed", "random", "special"],
        default="normal",
        help="Controls the label alphabet and normalization.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices or 'all'. Layer indexing starts at 0 (first transformer block).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20250315)
    parser.add_argument("--probe_layers", type=int, default=1, help="Depth of the probe MLP.")
    parser.add_argument(
        "--hidden_state_cache",
        type=Path,
        default=None,
        help="Optional path to save/load hidden states to avoid recomputation.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for loading the base model. Use 'auto' for multi-GPU, 'cuda:0', etc. Use 'none' to keep on CPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the base model when loading.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=BASE_DIR / "experiments/number_probe_metrics.json",
        help="Path to save structured metrics as JSON.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="number_probe",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--per_layer_report_dir",
        type=Path,
        default=None,
        help="If set, write each layer's test report to layer_{idx}.txt in this directory instead of a single file.",
    )
    parser.add_argument(
        "--save_probe_dir",
        type=Path,
        default=None,
        help="Optional directory to save per-layer probe weights for later reuse.",
    )
    parser.add_argument(
        "--shuffle_baseline",
        action="store_true",
        help="If set, shuffle hidden states across tokens to break input-label alignment (baseline).",
    )
    return parser.parse_args()


def get_label_builder(dataset_type: str) -> tuple[str, Callable[[str], str]]:
    if dataset_type == "special":
        return SPECIAL_ALPHABET, (lambda x: x)
    if dataset_type == "perturbed":
        return ASCII_ALPHABET, (lambda x: normalize_confusables(x).lower())
    return ASCII_ALPHABET, (lambda x: x.lower())


def setup_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb is not installed; continuing without logging.")
        return None
    run_name = args.wandb_run_name or f"{args.model_path.name}-number-probe"
    run = wandb.init(project=args.wandb_project, name=run_name, config={"args": vars(args)})
    return run


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    words = load_words(args.dataset_path)
    alphabet, preprocess = get_label_builder(args.dataset_type)
    labels, char2label, max_count = build_labels(words, alphabet, preprocess)

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    cached = None
    if args.hidden_state_cache and args.hidden_state_cache.exists():
        cached = torch.load(args.hidden_state_cache, map_location="cpu")
    hidden_states = _maybe_recollect_states(
        cached=cached,
        model=model,
        tokenizer=tokenizer,
        words=words,
        input_device=input_device,
        cache_path=args.hidden_state_cache,
    )

    wandb_run = setup_wandb(args)

    total_layers = hidden_states.shape[0]
    layer_indices = parse_layers(args.layers, total_layers)
    for idx in layer_indices:
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Layer index {idx} is out of range (0, {total_layers - 1}).")

    train_idx, test_idx = split_indices(len(words), args.train_ratio, args.seed)

    if args.shuffle_baseline:
        rng = torch.Generator().manual_seed(args.seed)
        shuffle_idx = torch.randperm(len(words), generator=rng)
        hidden_states = hidden_states.index_select(1, shuffle_idx)
        print("Shuffle baseline ON: hidden states have been permuted across tokens.")
    else:
        print("Shuffle baseline OFF: using aligned hidden states.")

    if args.per_layer_report_dir:
        args.per_layer_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_path} ({args.dataset_type})")
    print(f"Alphabet size: {len(alphabet)}, max count per char: {max_count}")
    print(f"Samples: train={len(train_idx)}, test={len(test_idx)}")

    if wandb_run is not None:
        wandb_run.config.update(
            {
                "total_layers": total_layers,
                "alphabet_size": len(alphabet),
                "max_count": max_count,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
            }
        )

    results = []
    for layer_idx in layer_indices:
        wandb_step_base = layer_idx * (args.epochs + 1)
        metrics = train_one_layer(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            labels=labels,
            train_idx=train_idx,
            test_idx=test_idx,
            max_count=max_count,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=probe_device,
            probe_layers=args.probe_layers,
            wandb_run=wandb_run,
            words=words,
            alphabet=alphabet,
            save_dir=args.save_probe_dir,
            wandb_step_base=wandb_step_base,
            seed_base=args.seed + layer_idx,
            per_layer_report_dir=args.per_layer_report_dir,
        )
        results.append((layer_idx, metrics))
        print(
            f"[Layer {layer_idx:02d}] "
            f"train_acc={metrics['train_acc']:.4f}, train_em={metrics['train_em']:.4f}, "
            f"test_acc={metrics['test_acc']:.4f}, test_em={metrics['test_em']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"layer_{layer_idx}/final_train_acc": metrics["train_acc"],
                    f"layer_{layer_idx}/final_train_em": metrics["train_em"],
                    f"layer_{layer_idx}/final_train_mp": metrics["train_mp"],
                    f"layer_{layer_idx}/final_train_mr": metrics["train_mr"],
                    f"layer_{layer_idx}/final_train_mf1": metrics["train_mf1"],
                    f"layer_{layer_idx}/final_test_acc": metrics["test_acc"],
                    f"layer_{layer_idx}/final_test_em": metrics["test_em"],
                    f"layer_{layer_idx}/final_test_mp": metrics["test_mp"],
                    f"layer_{layer_idx}/final_test_mr": metrics["test_mr"],
                    f"layer_{layer_idx}/final_test_mf1": metrics["test_mf1"],
                },
                step=wandb_step_base + args.epochs,
            )

    # Keep char2label info for downstream analysis if caching is used.
    if args.hidden_state_cache:
        meta_path = args.hidden_state_cache.with_suffix(".meta.pt")
        torch.save({"char2label": char2label, "alphabet": alphabet}, meta_path)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "model_path": str(args.model_path),
                "dataset_path": str(args.dataset_path),
                "dataset_type": args.dataset_type,
                "layers": layer_indices,
                "train_ratio": args.train_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "results": [
                    {
                        "layer": layer_idx,
                        "train_acc": metrics["train_acc"],
                        "train_em": metrics["train_em"],
                        "train_mp": metrics["train_mp"],
                        "train_mr": metrics["train_mr"],
                        "train_mf1": metrics["train_mf1"],
                        "test_acc": metrics["test_acc"],
                        "test_em": metrics["test_em"],
                        "test_mp": metrics["test_mp"],
                        "test_mr": metrics["test_mr"],
                        "test_mf1": metrics["test_mf1"],
                    }
                    for layer_idx, metrics in results
                ],
            },
            args.output_json.open("w"),
            indent=2,
        )
        print(f"Saved metrics to {args.output_json}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
