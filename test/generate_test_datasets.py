"""
Generate small test datasets for all tokenbench tasks (excluding Wordle).

Each task gets a minimal dataset (often 1 sample) using the task's own
generation scripts. This keeps the pipelines testable without large or
expensive runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run(cmd: List[str], cwd: Path) -> None:
    """Run a command and stream its output."""
    print(f"\n=== Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tasks_root = repo_root / "tasks"
    out_root = repo_root / "test" / "datasets"
    out_root.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, List[str], Path]] = [
        (
            "adversarial_prompt",
            [
                sys.executable,
                "generate_contexts.py",
                "--samples_per_difficulty",
                "1",
                "--output",
                str(out_root / "adversarial_prompt_contexts.json"),
                "--difficulty_level",
                "easy",
                "--cost",
            ],
            tasks_root / "adversarial_prompt",
        ),
        (
            "aligned_table",
            [
                sys.executable,
                "scripts/generate_contexts.py",
                "--model",
                "gpt-4",
                "--num_contexts",
                "1",
                "--output",
                str(out_root / "aligned_table_contexts.json"),
                "--min_rows",
                "3",
                "--max_rows",
                "3",
                "--min_cols",
                "3",
                "--max_cols",
                "3",
            ],
            tasks_root / "aligned_table",
        ),
        (
            "biological_sequence",
            [
                sys.executable,
                "generate_datasets.py",
                "--num-cases",
                "1",
                "--min-length",
                "20",
                "--max-length",
                "20",
                "--output-dir",
                str(out_root / "biological_sequence"),
            ],
            tasks_root / "biological_sequence",
        ),
        (
            "cipher_decipher",
            [
                sys.executable,
                "generate_dataset.py",
                "--samples",
                "1",
                "--batch-size",
                "1",
                "--output",
                str(out_root / "cipher_decipher_dataset.json"),
                "--difficulties",
                "easy",
            ],
            tasks_root / "cipher_decipher",
        ),
        (
            "context_aware_redaction",
            [
                sys.executable,
                "generate_data.py",
                "--num_samples",
                "1",
                "--difficulty",
                "easy",
                "--output",
                str(out_root / "context_aware_redaction.json"),
            ],
            tasks_root / "context_aware_redaction",
        ),
        (
            "gomoku",
            [
                sys.executable,
                "generate_datasets.py",
                "--board-sizes",
                "9",
                "--test-counts",
                "1",
                "--name",
                "test_small",
                "--with-metadata",
                "--seed",
                "42",
            ],
            tasks_root / "gomoku",
        ),
        (
            "map_navigation_sokoban",
            [
                sys.executable,
                "-m",
                "generators.sokoban_generator",
                "--size",
                "6",
                "--num-maps",
                "1",
                "--tasks-type-1",
                "1",
                "--tasks-type-2",
                "1",
                "--tasks-type-3",
                "1",
                "--tasks-type-4",
                "1",
                "--output",
                str(out_root / "map_navigation_sokoban.json"),
                "--seed",
                "42",
            ],
            tasks_root / "map_navigation",
        ),
        (
            "map_navigation_frozenlake",
            [
                sys.executable,
                "-m",
                "generators.frozenlake_generator",
                "--size",
                "6",
                "--num-holes",
                "2",
                "--num-maps",
                "1",
                "--tasks-type-1",
                "1",
                "--tasks-type-2",
                "1",
                "--tasks-type-3",
                "1",
                "--tasks-type-4",
                "1",
                "--tasks-type-5",
                "1",
                "--output",
                str(out_root / "map_navigation_frozenlake.json"),
                "--seed",
                "42",
            ],
            tasks_root / "map_navigation",
        ),
        (
            "rsa_randomart",
            [
                sys.executable,
                "main.py",
                "--generate-only",
                "--num-samples",
                "1",
                "--num-differences",
                "3",
                "--pattern-width",
                "15",
                "--pattern-height",
                "9",
                "--key-size",
                "1024",
                "--output-data",
                str(out_root / "rsa_randomart.json"),
            ],
            tasks_root / "rsa_randomart",
        ),
        (
            "typewriter",
            [
                sys.executable,
                "scripts/generate_datasets.py",
                "--samples",
                "5",
                "--seed",
                "42",
                "--output",
                str(out_root / "typewriter_test_cases.json"),
            ],
            tasks_root / "typewriter",
        ),
        (
            "tree_task1",
            [
                sys.executable,
                "generate_tasks1.py",
                "--difficulty",
                "easy",
                "--num_samples",
                "1",
                "--output_dir",
                str(out_root),
                "--output_file",
                "tree_task1_easy.json",
            ],
            tasks_root / "tree",
        ),
        (
            "tree_task2",
            [
                sys.executable,
                "generate_tasks2.py",
                "--difficulty",
                "easy",
                "--num_samples",
                "1",
                "--output_dir",
                str(out_root),
                "--output_file",
                "tree_task2_easy.json",
                "--threshold",
                "2",
            ],
            tasks_root / "tree",
        ),
    ]

    for name, cmd, cwd in tasks:
        print(f"\n### Generating dataset for {name}")
        run(cmd, cwd)

    print("\nAll test datasets attempted. Check the 'test/datasets' directory for outputs.")


if __name__ == "__main__":
    main()
