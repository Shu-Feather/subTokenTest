"""Create question/answer JSONL using the RSA random art prompt pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from src.utils import create_prompt


def load_records(path: Path) -> List[Dict[str, any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "samples", "datasets"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [raw]
    raise ValueError("Unsupported dataset structure; expected list or dict.")


def format_ground_truth(ground_truth: List[Dict[str, any]]) -> str:
    """Format differences as lines `(x, y): original -> modified`."""
    sorted_gt = sorted(ground_truth, key=lambda d: (d.get("y", 0), d.get("x", 0)))
    lines = []
    for diff in sorted_gt:
        x, y = diff.get("x"), diff.get("y")
        orig = diff.get("original", " ")
        mod = diff.get("modified", " ")
        lines.append(f"({x}, {y}): {orig} -> {mod}")
    return "\n".join(lines)


def write_jsonl(
    records: Iterable[Dict[str, any]],
    out_path: Path,
    restricted_reasoning: bool,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            pattern1 = rec.get("pattern1") or []
            pattern2 = rec.get("pattern2") or []
            ground_truth = rec.get("ground_truth") or rec.get("differences") or []
            obj = {
                "question": create_prompt(pattern1, pattern2, restricted_reasoning=restricted_reasoning),
                "answer": format_ground_truth(ground_truth),
                "id": rec.get("id"),
                "metadata": rec.get("metadata"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from rsa_randomart datasets using task prompts"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK_ROOT / "config" / "config.yaml",
        help="Path to benchmark config (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_stem>_qa.jsonl next to dataset)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    prompt_cfg = config.get("prompt", {})

    records = load_records(args.dataset)
    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")
    write_jsonl(records, output_path, prompt_cfg.get("restricted_reasoning", False))
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
