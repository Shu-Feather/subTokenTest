"""Build question/answer JSONL using the biological_sequence prompt pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from src.prompt_templates import PromptTemplates


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """Load dataset entries and normalize them into a common shape."""
    raw = json.loads(path.read_text())
    cases: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        cases = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("datasets"), dict):
            for task_type, items in raw["datasets"].items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    entry = {
                        "task_type": item.get("task_type", task_type),
                        "input": item.get("input"),
                        "expected_output": item.get("expected_output"),
                        "metadata": item,
                    }
                    cases.append(entry)
        elif isinstance(raw.get("samples"), list):
            cases = raw["samples"]
        else:
            cases = [raw]
    else:
        raise ValueError("Unsupported dataset structure; expected list or dict.")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(cases):
        task_type = item.get("task_type")
        input_sequence = item.get("input")
        expected = item.get("expected_output") or item.get("output") or item.get("answer")
        if task_type and input_sequence is not None:
            normalized.append(
                {
                    "id": item.get("id", idx),
                    "task_type": task_type,
                    "input": input_sequence,
                    "expected_output": expected or "",
                    "metadata": item,
                }
            )
    return normalized


def build_question(record: Dict[str, Any], restricted_reasoning: bool) -> str:
    """Use the official prompt templates to construct the model prompt."""
    return PromptTemplates.get_prompt_for_task(
        record["task_type"],
        record["input"],
        restricted_reasoning=restricted_reasoning,
    )


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    out_path: Path,
    restricted_reasoning: bool,
) -> None:
    """Write prompts and answers to JSONL."""
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "question": build_question(rec, restricted_reasoning),
                "answer": rec["expected_output"],
                "task_type": rec["task_type"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from biological_sequence datasets"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_stem>_qa.jsonl next to the dataset)",
    )
    parser.add_argument(
        "--restricted-reasoning",
        action="store_true",
        help="Append restricted reasoning hint used by the benchmark prompts",
    )
    args = parser.parse_args()

    records = load_cases(args.dataset)
    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")

    write_jsonl(records, output_path, args.restricted_reasoning)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
