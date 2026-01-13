"""Create question/answer JSONL using the context_aware_redaction prompt pipeline."""

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


def load_records(path: Path) -> List[Dict[str, any]]:
    """Load dataset entries from common formats."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "samples", "datasets", "records"):
            if key in raw:
                val = raw[key]
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    out: List[Dict[str, any]] = []
                    for v in val.values():
                        if isinstance(v, list):
                            out.extend(v)
                    if out:
                        return out
        return [raw]
    raise ValueError("Unsupported dataset format; expected list or dict.")


def build_prompt(original_text: str, restricted_reasoning: bool) -> str:
    """Mirror the benchmark create_prompt logic."""
    prompt = f"""You are a data privacy assistant. Your task is to redact (mask) sensitive information in the given text according to specific rules.

**Redaction Rules:**

1. **18-digit ID Card Number**: Keep the first 6 digits and last 2 digits, mask the middle 10 digits with asterisks (*)
   - Example: 123456789012345678 → 123456**********78

2. **Phone Number** (format: +[area code] [11-digit number]): Keep the '+' sign and area code, keep the first 3 digits and last 4 digits of the phone number, mask the middle 4 digits with asterisks (*)
   - Example: +12 12345678901 → +12 123****8901

3. **Credit Card Number**: Keep the first 6 digits and last 4 digits, mask all middle digits with asterisks (*)
   - Example: 1234 5678 9012 3456 → 123456******3456

**Instructions:**
- Carefully identify all sensitive information in the text (phone numbers, ID card numbers, credit card numbers)
- Apply the appropriate masking rule for each type of sensitive information
- Keep all other text exactly the same
- Place your final redacted text between <answer> and </answer> tags

**Text to redact:**
{original_text}

**Your redacted text (place between <answer></answer> tags):**"""

    if restricted_reasoning:
        prompt += (
            "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        )
    return prompt


def write_jsonl(
    records: Iterable[Dict[str, any]],
    out_path: Path,
    restricted_reasoning: bool,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            original = rec.get("original_context") or rec.get("text") or ""
            answer = rec.get("redacted_context") or rec.get("answer") or ""
            obj = {
                "question": build_prompt(str(original), restricted_reasoning),
                "answer": str(answer),
                "id": rec.get("id"),
                "difficulty": rec.get("difficulty"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from context_aware_redaction datasets using task prompts"
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
