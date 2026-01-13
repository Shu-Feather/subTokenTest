"""Create question/answer JSONL using the typewriter task prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TASK_ROOT = Path(__file__).resolve().parents[1]
PARENT_OF_TASK = TASK_ROOT.parent
PROJECT_ROOT = TASK_ROOT.parents[1]
for candidate in (PARENT_OF_TASK, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from typewriter.tasks.task1_typewriter import Task1TypewriterEffect
from typewriter.tasks.task2_backspace import Task2BackspaceHandling


def _select_cases(values: object, difficulty: Optional[str]) -> List[str]:
    """Pick cases for a given difficulty (or all)."""
    if isinstance(values, list):
        return [str(v) for v in values]
    if isinstance(values, dict):
        if difficulty and difficulty != "all":
            return [str(v) for v in values.get(difficulty, [])]
        combined: List[str] = []
        for lst in values.values():
            if isinstance(lst, list):
                combined.extend(str(v) for v in lst)
        return combined
    return []


def load_cases(path: Path, difficulty: str) -> Dict[str, List[str]]:
    """Load task1/task2 inputs from dataset file."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "task1" not in raw or "task2" not in raw:
        raise ValueError("Dataset must contain 'task1' and 'task2' entries.")
    return {
        "task1": _select_cases(raw["task1"], difficulty),
        "task2": _select_cases(raw["task2"], difficulty),
    }


def build_task1_prompt(task: Task1TypewriterEffect, word: str, prompt_type: str, restricted: bool) -> str:
    if prompt_type == "few_shot":
        return task.get_few_shot_prompt(word, restricted)
    return f"{task.get_system_prompt(restricted)}\n\n{task.get_user_prompt(word, restricted)}"


def build_task2_prompt(task: Task2BackspaceHandling, log: str, prompt_type: str, restricted: bool) -> str:
    if prompt_type == "few_shot":
        return task.get_few_shot_prompt(log, restricted)
    return f"{task.get_system_prompt(restricted)}\n\n{task.get_user_prompt(log, restricted)}"


def write_jsonl(
    task1_inputs: Iterable[str],
    task2_inputs: Iterable[str],
    prompt_type: str,
    restricted_reasoning: bool,
    out_path: Path,
) -> None:
    t1 = Task1TypewriterEffect()
    t2 = Task2BackspaceHandling()

    with out_path.open("w", encoding="utf-8") as f:
        for word in task1_inputs:
            question = build_task1_prompt(t1, word, prompt_type, restricted_reasoning)
            answer = t1.generate_expected_output(word)
            f.write(
                json.dumps(
                    {"question": question, "answer": answer, "task_type": "task1", "input": word},
                    ensure_ascii=False,
                )
                + "\n"
            )

        for typing_log in task2_inputs:
            question = build_task2_prompt(t2, typing_log, prompt_type, restricted_reasoning)
            answer = t2.generate_expected_output(typing_log)
            f.write(
                json.dumps(
                    {"question": question, "answer": answer, "task_type": "task2", "input": typing_log},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create question/answer JSONL from typewriter datasets using task prompts"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_stem>_qa.jsonl next to dataset)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty split to use when dataset is split by level",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["system", "few_shot"],
        default="system",
        help="Prompt variant to use (default: system + user)",
    )
    parser.add_argument(
        "--restricted-reasoning",
        action="store_true",
        help="Append restricted reasoning hint used by the benchmark",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset, args.difficulty)
    output_path = args.output or args.dataset.with_name(f"{args.dataset.stem}_qa.jsonl")
    write_jsonl(cases["task1"], cases["task2"], args.prompt_type, args.restricted_reasoning, output_path)
    print(f"Wrote {len(cases['task1']) + len(cases['task2'])} records to {output_path}")


if __name__ == "__main__":
    main()
