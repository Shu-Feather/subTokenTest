"""Create question/answer JSONL using the adversarial_prompt pipeline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from src.perturbation import PerturbationEngine
from src.prompts import get_canonicalization_prompt


def load_entries(path: Path) -> List[Dict[str, Any]]:
    """Load dataset entries and flatten category/difficulty structures."""
    raw = json.loads(path.read_text())
    entries: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("samples"), list):
            entries = raw["samples"]
        else:
            for category, diff_map in raw.items():
                if not isinstance(diff_map, dict):
                    continue
                for difficulty, contexts in diff_map.items():
                    if not isinstance(contexts, list):
                        continue
                    for ctx in contexts:
                        if isinstance(ctx, dict):
                            entry = dict(ctx)
                        else:
                            entry = {"original": ctx}
                        entry.setdefault("category", category)
                        entry.setdefault("difficulty", difficulty)
                        entries.append(entry)
    else:
        raise ValueError("Unsupported dataset structure; expected list or dict.")

    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        original = entry.get("original") or entry.get("text") or entry.get("context")
        if original is None and isinstance(entry, str):
            original = entry
        if original is None:
            continue
        normalized.append(
            {
                "id": entry.get("id", idx),
                "category": entry.get("category", "unknown"),
                "difficulty": entry.get("difficulty", "unknown"),
                "original": str(original),
                "perturbed": entry.get("perturbed"),
                "perturbation_type": entry.get("perturbation_type"),
            }
        )
    return normalized


def ensure_perturbations(
    entries: Iterable[Dict[str, Any]],
    perturb_config: Dict[str, Any],
    num_samples: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    """Guarantee each entry has a perturbed variant using the official engine."""
    random.seed(seed)
    engine = PerturbationEngine(perturb_config)
    perturb_types = perturb_config.get("types", [])
    rng = random.Random(seed)

    base_entries = list(entries)
    if num_samples is not None:
        base_entries = base_entries[:num_samples]

    samples: List[Dict[str, Any]] = []
    for idx, entry in enumerate(base_entries):
        perturbed = entry.get("perturbed")
        pert_type = entry.get("perturbation_type")
        if perturbed is None:
            pert_type = pert_type or (rng.choice(perturb_types) if perturb_types else "unknown")
            perturbed = engine.perturb(entry["original"], pert_type)
        samples.append(
            {
                "id": entry.get("id", idx),
                "category": entry.get("category", "unknown"),
                "difficulty": entry.get("difficulty", "unknown"),
                "original": entry["original"],
                "perturbed": perturbed,
                "perturbation_type": pert_type or "unknown",
            }
        )
    return samples


def build_question(record: Dict[str, Any], prompt_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create the canonicalization chat messages."""
    return get_canonicalization_prompt(
        perturbed_text=record["perturbed"],
        use_few_shot=prompt_cfg.get("use_few_shot", True),
        num_examples=prompt_cfg.get("num_few_shot_examples"),
        restricted_reasoning=prompt_cfg.get("restricted_reasoning", False),
    )


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    prompt_cfg: Dict[str, Any],
    out_path: Path,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "question": build_question(rec, prompt_cfg),
                "answer": rec["original"],
                "meta": {
                    "category": rec.get("category"),
                    "difficulty": rec.get("difficulty"),
                    "perturbation_type": rec.get("perturbation_type"),
                    "id": rec.get("id"),
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from adversarial_prompt datasets using task prompts"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK_ROOT / "config" / "benchmark_config.yaml",
        help="Path to benchmark config (default: config/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_stem>_qa.jsonl next to dataset)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional limit on the number of samples to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for perturbation selection",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    prompt_cfg = config.get("prompt", {})
    perturb_cfg = config.get("perturbation", {})

    entries = load_entries(args.dataset)
    samples = ensure_perturbations(entries, perturb_cfg, args.num_samples, args.seed)

    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")
    write_jsonl(samples, prompt_cfg, output_path)
    print(f"Wrote {len(samples)} records to {output_path}")


if __name__ == "__main__":
    main()
