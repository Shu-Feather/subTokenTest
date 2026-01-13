"""Create question/answer JSONL using the tree task prompts (with <answer> tags)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from config.config import Config
from data.data_manager import DataManager


def load_records(path: Path) -> List[Dict[str, any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("samples", "data", "datasets", "test_cases"):
            if key in raw:
                val = raw[key]
                if isinstance(val, list):
                    return val
        return [raw]
    raise ValueError("Unsupported dataset structure; expected list or dict.")


def build_prompt(manager: DataManager, record: Dict[str, any], restricted_reasoning: bool) -> str:
    task_type = record.get("task_type", "task1").lower()
    tree_str = record.get("tree_structure", "")
    question = record.get("question", "")
    if task_type == "task2":
        prompt = manager._create_task2_prompt(tree_str, question)
    else:
        prompt = manager._create_task1_prompt(tree_str, question)

    if restricted_reasoning:
        prompt = (
            f"{prompt}\n\nAnswer directly inside <answer> tags without extended reasoning. "
            "Begin your answer now: <answer>"
        )
    return prompt


def write_jsonl(
    records: Iterable[Dict[str, any]],
    out_path: Path,
    restricted_reasoning: bool,
    manager: DataManager,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "question": build_prompt(manager, rec, restricted_reasoning),
                "answer": rec.get("expected_answer") or rec.get("answer") or "",
                "task_type": rec.get("task_type"),
                "id": rec.get("sample_id") or rec.get("id"),
                "difficulty": rec.get("difficulty"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from tree datasets using task prompts"
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

    dummy_config = Config(model_type="openai", model_name="placeholder", restricted_reasoning=args.restricted_reasoning)
    manager = DataManager(dummy_config)

    records = load_records(args.dataset)
    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")
    write_jsonl(records, output_path, args.restricted_reasoning, manager)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
