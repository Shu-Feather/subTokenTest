"""Create question/answer JSONL using the gomoku benchmark prompts."""

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

from configs.gomoku.config import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def load_records(path: Path) -> List[Dict[str, any]]:
    """Load test cases (with or without metadata wrapper)."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("test_cases"), list):
            return raw["test_cases"]
        for key in ("data", "samples", "datasets"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [raw]
    raise ValueError("Unsupported dataset format; expected list or dict.")


def build_prompt(board_repr: str, board_size: int, restricted_reasoning: bool) -> List[Dict[str, str]]:
    """Construct the system/user chat messages mirroring ModelInterface.format_prompt."""
    user_prompt = USER_PROMPT_TEMPLATE.format(board_size=board_size, board_representation=board_repr)
    if restricted_reasoning:
        user_prompt = (
            f"{user_prompt}\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def write_jsonl(
    records: Iterable[Dict[str, any]],
    out_path: Path,
    restricted_reasoning: bool,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            board = rec.get("board") or rec.get("board_representation") or ""
            board_size = rec.get("board_size") or rec.get("size") or 0
            obj = {
                "question": build_prompt(str(board), int(board_size), restricted_reasoning),
                "answer": rec.get("expected") or rec.get("label") or rec.get("answer") or "",
                "id": rec.get("id"),
                "board_size": board_size,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from gomoku datasets using task prompts"
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
