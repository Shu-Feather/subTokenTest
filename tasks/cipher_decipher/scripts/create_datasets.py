"""Create question/answer JSONL using the cipher_decipher prompt templates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
from enum import Enum

TASK_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = TASK_ROOT / "src"


def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()

for p in (PROJECT_ROOT, TASK_ROOT, SRC_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ciphers.caesar_cipher import CaesarCipher
from ciphers.morse_code import MorseCode
from utils.prompts import PromptTemplates


class TaskTypeEnum(str, Enum):
    MORSE_ENCODE = "morse_encode"
    MORSE_DECODE = "morse_decode"
    CAESAR_ENCODE = "caesar_encode"
    CAESAR_DECODE = "caesar_decode"


def load_text_dataset(path: Path) -> Dict[str, List[str]]:
    """Load difficulty-keyed text dataset produced by generate_dataset.py."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Dataset must be a dict keyed by difficulty levels.")
    dataset: Dict[str, List[str]] = {}
    for difficulty, texts in raw.items():
        if isinstance(texts, list):
            dataset[difficulty] = [str(t) for t in texts]
    return dataset


def prompt_style(task_name: str, prompt_settings: Dict[str, any]) -> str:
    per_task = prompt_settings.get("per_task_styles", {}) or {}
    return per_task.get(task_name, prompt_settings.get("default_style", "detailed"))


def generate_samples(
    dataset: Dict[str, List[str]],
    config: Dict[str, any],
    max_per_difficulty: int | None,
) -> List[Dict[str, any]]:
    """Generate prompts/answers for all enabled tasks."""
    test_cfg = config.get("test_config", {})
    tasks = test_cfg.get(
        "tasks",
        {"morse_encode": True, "morse_decode": True, "caesar_encode": True, "caesar_decode": True},
    )
    caesar_shifts = test_cfg.get("caesar_shifts", [1, 3, 5, 7, 13, 25])
    prompt_settings = config.get("prompt_settings", {})
    restricted_reasoning = prompt_settings.get("restricted_reasoning", False)

    templates = PromptTemplates()
    samples: List[Dict[str, any]] = []
    shift_index = 0

    for difficulty, texts in dataset.items():
        texts_to_use = texts if max_per_difficulty is None else texts[:max_per_difficulty]
        for text in texts_to_use:
            if tasks.get("morse_encode", False):
                style = prompt_style(TaskTypeEnum.MORSE_ENCODE.value, prompt_settings)
                prompt = templates.get_morse_encode_prompt(text, style, restricted_reasoning)
                samples.append(
                    {
                        "question": prompt,
                        "answer": MorseCode.encode(text),
                        "task_type": TaskTypeEnum.MORSE_ENCODE.value,
                        "difficulty": difficulty,
                    }
                )

            if tasks.get("morse_decode", False):
                style = prompt_style(TaskTypeEnum.MORSE_DECODE.value, prompt_settings)
                morse_text = MorseCode.encode(text)
                prompt = templates.get_morse_decode_prompt(morse_text, style, restricted_reasoning)
                samples.append(
                    {
                        "question": prompt,
                        "answer": text,
                        "task_type": TaskTypeEnum.MORSE_DECODE.value,
                        "difficulty": difficulty,
                    }
                )

            if tasks.get("caesar_encode", False):
                shift = caesar_shifts[shift_index % len(caesar_shifts)]
                shift_index += 1
                style = prompt_style(TaskTypeEnum.CAESAR_ENCODE.value, prompt_settings)
                prompt = templates.get_caesar_encode_prompt(text, shift, style, restricted_reasoning)
                samples.append(
                    {
                        "question": prompt,
                        "answer": CaesarCipher.encode(text, shift),
                        "task_type": TaskTypeEnum.CAESAR_ENCODE.value,
                        "difficulty": difficulty,
                        "shift": shift,
                    }
                )

            if tasks.get("caesar_decode", False):
                shift = caesar_shifts[shift_index % len(caesar_shifts)]
                shift_index += 1
                encrypted = CaesarCipher.encode(text, shift)
                style = prompt_style(TaskTypeEnum.CAESAR_DECODE.value, prompt_settings)
                prompt = templates.get_caesar_decode_prompt(encrypted, shift, style, restricted_reasoning)
                samples.append(
                    {
                        "question": prompt,
                        "answer": text,
                        "task_type": TaskTypeEnum.CAESAR_DECODE.value,
                        "difficulty": difficulty,
                        "shift": shift,
                    }
                )
    return samples


def write_jsonl(records: Iterable[Dict[str, any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from cipher_decipher datasets using prompt templates"
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
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional cap on samples per difficulty",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dataset = load_text_dataset(args.dataset)
    samples = generate_samples(dataset, config, args.num_samples)

    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")
    write_jsonl(samples, output_path)
    print(f"Wrote {len(samples)} records to {output_path}")


if __name__ == "__main__":
    main()
