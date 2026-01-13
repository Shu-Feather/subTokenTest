"""Create question/answer JSONL using the map_navigation prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parents[1]
for candidate in (TASK_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prompts import get_sokoban_prompt, get_frozenlake_prompt


def load_records(path: Path) -> List[Dict[str, any]]:
    """Load task records from dataset files."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            return raw["data"]
        for key in ("samples", "datasets"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [raw]
    raise ValueError("Unsupported dataset structure; expected list or dict.")


def build_prompt(record: Dict[str, any], restricted_reasoning: bool) -> str:
    """Construct the prompt sent to the model."""
    env_type = (record.get("env_type") or record.get("type") or "sokoban").lower()
    map_str = record.get("map") or record.get("map_str") or ""
    question = record.get("question") or ""
    if env_type == "frozenlake":
        return get_frozenlake_prompt(map_str, question, restricted_reasoning=restricted_reasoning)
    return get_sokoban_prompt(map_str, question, restricted_reasoning=restricted_reasoning)


def write_jsonl(
    records: Iterable[Dict[str, any]],
    out_path: Path,
    restricted_reasoning: bool,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "question": build_prompt(rec, restricted_reasoning),
                "answer": rec.get("answer") or rec.get("expected") or "",
                "id": rec.get("id") or rec.get("task_id"),
                "env_type": rec.get("env_type"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from map_navigation datasets using task prompts"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_stem>_qa.jsonl next to dataset)",
    )
    parser.add_argument(
        "--restricted-reasoning",
        action="store_true",
        help="Append restricted reasoning hint used by the benchmark",
    )
    args = parser.parse_args()

    records = load_records(args.dataset)
    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")

    write_jsonl(records, output_path, args.restricted_reasoning)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
