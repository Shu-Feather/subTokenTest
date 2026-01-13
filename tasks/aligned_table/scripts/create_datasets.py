"""Create question/answer JSONL using the aligned_table prompt builder."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from src.prompt_builder import PromptBuilder
from src.utils import format_table


def load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load raw test cases from a dataset file."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("test_cases", "data", "samples", "datasets"):
            if key in raw:
                val = raw[key]
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    cases: List[Dict[str, Any]] = []
                    for v in val.values():
                        if isinstance(v, list):
                            cases.extend(v)
                    if cases:
                        return cases
        return [raw]
    raise ValueError("Unsupported dataset format; expected list or dict.")


def normalize_test_cases(
    test_cases: List[Dict[str, Any]],
    config: Dict[str, Any],
    format_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Mirror the benchmark normalization to attach formats and ids."""
    normalized_cases: List[Dict[str, Any]] = []
    table_formats = config.get("test", {}).get("table_formats", ["latex", "markdown", "text"])

    format_dist = config.get("test", {}).get("format_distribution")
    if format_dist:
        formats = list(format_dist.keys())
        weights = list(format_dist.values())
        total = sum(weights)
        weights = [w / total for w in weights]
    else:
        formats = table_formats
        weights = None

    for idx, test_case in enumerate(test_cases):
        if "table_format" not in test_case:
            if format_filter:
                table_format = format_filter.lower()
            else:
                if weights:
                    table_format = np.random.choice(formats, p=weights)
                else:
                    table_format = random.choice(formats)

            normalized_case = {
                "id": test_case.get("id", idx),
                "entity_type": test_case.get("entity_type", "custom"),
                "table_data": test_case.get("table_data", []),
                "context": test_case.get("context", ""),
                "table_format": table_format,
                "num_rows": test_case.get("num_rows", len(test_case.get("table_data", []))),
                "num_cols": test_case.get(
                    "num_cols",
                    len(test_case.get("table_data", [[]])[0]) if test_case.get("table_data") else 0,
                ),
                "description": test_case.get("description", ""),
            }
        else:
            normalized_case = dict(test_case)
            if "id" not in normalized_case:
                normalized_case["id"] = idx
        normalized_cases.append(normalized_case)

    return normalized_cases


def write_jsonl(
    cases: Iterable[Dict[str, Any]],
    builder: PromptBuilder,
    out_path: Path,
) -> None:
    """Write question/answer pairs to JSONL."""
    with out_path.open("w", encoding="utf-8") as f:
        for case in cases:
            prompt = builder.build_prompt(case)
            answer = format_table(case.get("table_data", []), case.get("table_format", ""))
            f.write(
                json.dumps(
                    {
                        "question": prompt,
                        "answer": answer,
                        "table_format": case.get("table_format"),
                        "id": case.get("id"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from aligned_table datasets using the task prompt builder"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_stem>_qa.jsonl in the same directory)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK_ROOT / "config" / "config.yaml",
        help="Path to benchmark config (default: config/config.yaml)",
    )
    parser.add_argument(
        "--format",
        dest="format_filter",
        default=None,
        choices=["latex", "markdown", "text"],
        help="Force all prompts to use a specific table format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for format selection",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = yaml.safe_load(Path(args.config).read_text())
    prompt_cfg = config.get("prompt", {})

    records = load_test_cases(args.dataset)
    cases = normalize_test_cases(records, config, args.format_filter)

    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")
    builder = PromptBuilder(restricted_reasoning=prompt_cfg.get("restricted_reasoning", False))
    write_jsonl(cases, builder, output_path)
    print(f"Wrote {len(cases)} records to {output_path}")


if __name__ == "__main__":
    main()
