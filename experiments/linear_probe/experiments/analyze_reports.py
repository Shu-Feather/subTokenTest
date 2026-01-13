"""
Analyze per-layer test reports (layer_{num}.txt) produced by number_probe.

For each layer file:
- Parse per-sample predictions/targets.
- Compute total cases, exact-match count, and EM rate.
- Save a JSON summary for all layers in the same directory.
- Print a table of EM vs. layer.
- Output detailed per-character statistics to text files.

Character weighting:
- normal/random/perturbed: uniform weight (1/26) for all lowercase letters
- special: weighted based on generation probabilities (_,#,B: 3x; others: 1x)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import torch


SPECIAL_ALPHABET = "_PGO#Xo+=.B*-@%&^"


def infer_dataset_type(report_dir: Path) -> str:
    """
    Infer dataset type from directory name.
    """
    dir_name = report_dir.name.lower()
    if "special" in dir_name:
        return "special"
    elif "perturbed" in dir_name:
        return "perturbed"
    elif "random" in dir_name:
        return "random"
    elif "normal" in dir_name:
        return "normal"
    else:
        # fallback: check parent directory
        parent_name = report_dir.parent.name.lower()
        if "special" in parent_name:
            return "special"
        elif "perturbed" in parent_name:
            return "perturbed"
        elif "random" in parent_name:
            return "random"
        else:
            return "normal"


def get_character_weights(dataset_type: str, alphabet: List[str]) -> Dict[str, float]:
    """
    Return character weights for macro averaging.
    
    - normal/random/perturbed: uniform (1/26)
    - special: weighted by generation probability
    """
    if dataset_type == "special":
        # Match the generation weights: _, #, B have weight 3; others have weight 1
        weights = {}
        total_weight = 0.0
        for ch in alphabet:
            if ch in {"_", "#", "B"}:
                weights[ch] = 3.0
            else:
                weights[ch] = 1.0
            total_weight += weights[ch]
        # Normalize to sum to 1
        for ch in weights:
            weights[ch] /= total_weight
        return weights
    else:
        # Uniform weights for normal/random/perturbed
        uniform_weight = 1.0 / len(alphabet)
        return {ch: uniform_weight for ch in alphabet}


def parse_counts(line: str) -> Dict[str, int]:
    """
    Parse a line like "Model: a:1, b:0, ..." into a dict {char: count}.
    """
    counts = {}
    parts = line.split(": ", 1)
    if len(parts) != 2:
        return counts
    rest = parts[1]
    for kv in rest.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            continue
        ch, val = kv.split(":", 1)
        counts[ch.strip()] = int(val.strip())
    return counts


def parse_layer_report_detailed(
    path: Path, dataset_type: str
) -> Tuple[int, int, float, Dict[str, Dict]]:
    """
    Return (total_samples, em_hits, weighted_macro_f1, char_stats) for a layer report file.
    
    char_stats: {char: {f1, precision, recall, support, weight}}
    
    Weighted macro F1:
    - Compute F1 per character (multiclass over counts)
    - Average across characters using dataset-specific weights
    """
    text = path.read_text(encoding="utf-8")
    # Split by "Q:" blocks
    blocks = [b.strip() for b in text.split("\nQ: ") if b.strip()]
    total = 0
    em_hits = 0
    preds_list: List[Dict[str, int]] = []
    targets_list: List[Dict[str, int]] = []
    
    for block in blocks:
        lines = block.splitlines()
        model_line = next((l for l in lines if l.startswith("Model:")), None)
        target_line = next((l for l in lines if l.startswith("Target:")), None)
        if not model_line or not target_line:
            continue
        model_counts = parse_counts(model_line)
        target_counts = parse_counts(target_line)
        total += 1
        if model_counts == target_counts:
            em_hits += 1
        preds_list.append(model_counts)
        targets_list.append(target_counts)

    weighted_macro_f1 = 0.0
    char_stats = {}
    
    if preds_list and targets_list:
        alphabet = list(targets_list[0].keys())
        num_chars = len(alphabet)
        
        # Get character weights based on dataset type
        char_weights = get_character_weights(dataset_type, alphabet)
        
        max_count = 0
        for t in targets_list + preds_list:
            if t:
                max_count = max(max_count, max(t.values()))
        num_classes = max_count + 1
        conf = torch.zeros(num_chars, num_classes, num_classes, dtype=torch.long)
        
        for p_dict, t_dict in zip(preds_list, targets_list):
            for idx, ch in enumerate(alphabet):
                t = t_dict.get(ch, 0)
                p = p_dict.get(ch, 0)
                if t >= num_classes or p >= num_classes:
                    # expand if needed
                    new_size = max(t, p) + 1
                    new_conf = torch.zeros(num_chars, new_size, new_size, dtype=torch.long)
                    new_conf[:, :num_classes, :num_classes] = conf
                    conf = new_conf
                    num_classes = new_size
                conf[idx, t, p] += 1
        
        # Compute weighted macro F1 and collect per-character stats
        weighted_f1_sum = 0.0
        for idx, ch in enumerate(alphabet):
            tp = conf[idx].diag()
            pred_pos = conf[idx].sum(dim=0)
            true_pos = conf[idx].sum(dim=1)
            
            # Per-class precision/recall
            precision_per_class = tp / pred_pos.clamp(min=1)
            recall_per_class = tp / true_pos.clamp(min=1)
            
            # Average across count classes
            precision = precision_per_class.mean().item()
            recall = recall_per_class.mean().item()
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            support = true_pos.sum().item()
            
            char_stats[ch] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'support': int(support),
                'weight': char_weights[ch],
            }
            
            weighted_f1_sum += f1 * char_weights[ch]
        
        weighted_macro_f1 = weighted_f1_sum
        
    return total, em_hits, weighted_macro_f1, char_stats


def save_detailed_stats(
    layer: int,
    char_stats: Dict[str, Dict],
    output_dir: Path,
    dataset_type: str,
) -> None:
    """
    Save detailed per-character statistics to a text file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layer_{layer}_char_stats.txt"
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"===== Layer {layer} Character Statistics =====\n")
        f.write(f"Dataset Type: {dataset_type}\n\n")
        
        # Header
        f.write(f"{'Char':<6} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Support':<10} {'Weight':<10}\n")
        f.write(f"{'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}\n")
        
        # Sort by F1 score (descending)
        sorted_chars = sorted(
            char_stats.items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )
        
        for ch, stats in sorted_chars:
            f.write(
                f"{ch:<6} "
                f"{stats['f1']:<10.4f} "
                f"{stats['precision']:<12.4f} "
                f"{stats['recall']:<10.4f} "
                f"{stats['support']:<10} "
                f"{stats['weight']:<10.4f}\n"
            )
        
        # Summary statistics
        f.write(f"\n{'='*68}\n")
        f.write("Summary Statistics:\n")
        
        all_f1s = [s['f1'] for s in char_stats.values()]
        all_precisions = [s['precision'] for s in char_stats.values()]
        all_recalls = [s['recall'] for s in char_stats.values()]
        all_weights = [s['weight'] for s in char_stats.values()]
        
        f.write(f"  Number of characters: {len(char_stats)}\n")
        f.write(f"  Mean F1: {sum(all_f1s) / len(all_f1s):.4f}\n")
        f.write(f"  Mean Precision: {sum(all_precisions) / len(all_precisions):.4f}\n")
        f.write(f"  Mean Recall: {sum(all_recalls) / len(all_recalls):.4f}\n")
        f.write(f"  Weighted Macro F1: {sum(f * w for f, w in zip(all_f1s, all_weights)):.4f}\n")
        f.write(f"  Min F1: {min(all_f1s):.4f} (char: {min(char_stats.items(), key=lambda x: x[1]['f1'])[0]})\n")
        f.write(f"  Max F1: {max(all_f1s):.4f} (char: {max(char_stats.items(), key=lambda x: x[1]['f1'])[0]})\n")
        
        # Character groups analysis for special dataset
        if dataset_type == "special":
            f.write(f"\n{'='*68}\n")
            f.write("Character Group Analysis (Special Dataset):\n\n")
            
            high_freq_chars = ["_", "#", "B"]
            low_freq_chars = [ch for ch in char_stats.keys() if ch not in high_freq_chars]
            
            high_freq_f1s = [char_stats[ch]['f1'] for ch in high_freq_chars if ch in char_stats]
            low_freq_f1s = [char_stats[ch]['f1'] for ch in low_freq_chars if ch in char_stats]
            
            f.write(f"  High-frequency characters (_, #, B):\n")
            f.write(f"    Count: {len(high_freq_f1s)}\n")
            f.write(f"    Mean F1: {sum(high_freq_f1s) / len(high_freq_f1s):.4f}\n")
            f.write(f"    Min F1: {min(high_freq_f1s):.4f}\n")
            f.write(f"    Max F1: {max(high_freq_f1s):.4f}\n\n")
            
            f.write(f"  Low-frequency characters:\n")
            f.write(f"    Count: {len(low_freq_f1s)}\n")
            f.write(f"    Mean F1: {sum(low_freq_f1s) / len(low_freq_f1s):.4f}\n")
            f.write(f"    Min F1: {min(low_freq_f1s):.4f}\n")
            f.write(f"    Max F1: {max(low_freq_f1s):.4f}\n")


def find_layer_files(report_dir: Path) -> Dict[int, Path]:
    layer_files: Dict[int, Path] = {}
    for path in report_dir.glob("layer_*.txt"):
        m = re.match(r"layer_(\d+)\.txt", path.name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        layer_files[layer_idx] = path
    return dict(sorted(layer_files.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-layer probe reports.")
    parser.add_argument(
        "--report_dir",
        type=Path,
        required=True,
        help="Directory containing layer_{num}.txt files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Path to save summary JSON (default: report_dir/summary.json).",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["normal", "random", "perturbed", "special", "auto"],
        default="auto",
        help="Dataset type for character weighting (auto: infer from directory name).",
    )
    args = parser.parse_args()

    # Infer dataset type if auto
    if args.dataset_type == "auto":
        dataset_type = infer_dataset_type(args.report_dir)
        print(f"Auto-detected dataset type: {dataset_type}")
    else:
        dataset_type = args.dataset_type

    layer_files = find_layer_files(args.report_dir)
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.txt found in {args.report_dir}")

    # Create detailed stats directory
    detailed_stats_dir = args.report_dir / "detailed_stats"
    detailed_stats_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    layers = []
    em_rates = []
    macro_f1s = []

    print(f"\nDataset type: {dataset_type}")
    print("Layer\tTotal\tEM\tEM Rate\tWeighted MacroF1")
    
    for layer, path in layer_files.items():
        total, em_hits, macro_f1, char_stats = parse_layer_report_detailed(path, dataset_type)
        em_rate = em_hits / total if total else 0.0
        
        summary[layer] = {
            "total": total,
            "em": em_hits,
            "em_rate": em_rate,
            "weighted_macro_f1": macro_f1,
            "dataset_type": dataset_type,
            "char_stats": char_stats,
        }
        
        print(f"{layer}\t{total}\t{em_hits}\t{em_rate:.4f}\t{macro_f1:.4f}")
        layers.append(layer)
        em_rates.append(em_rate)
        macro_f1s.append(macro_f1)
        
        # Save detailed character statistics
        save_detailed_stats(layer, char_stats, detailed_stats_dir, dataset_type)

    print(f"\nDetailed character statistics saved to {detailed_stats_dir}/")

    output_path = args.output_json or args.report_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")

    # Plot EM vs layer
    plt.figure(figsize=(8, 4))
    plt.plot(layers, em_rates, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Exact Match Rate")
    plt.title(f"Exact Match vs. Layer ({dataset_type})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    png_path = args.report_dir / "em_vs_layer.png"
    plt.savefig(png_path, dpi=200)
    print(f"Plot saved to {png_path}")

    # Plot weighted macro_f1 vs layer
    if any(m is not None for m in macro_f1s):
        plt.figure(figsize=(8, 4))
        plt.plot(layers, macro_f1s, marker="o")
        plt.xlabel("Layer")
        plt.ylabel("Weighted Macro F1")
        plt.title(f"Weighted Macro F1 vs. Layer ({dataset_type})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        png_path_f1 = args.report_dir / "weighted_macro_f1_vs_layer.png"
        plt.savefig(png_path_f1, dpi=300)
        print(f"Weighted Macro F1 plot saved to {png_path_f1}")


if __name__ == "__main__":
    main()