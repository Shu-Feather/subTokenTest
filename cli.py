"""
Unified entrypoint for running any tokenbench task.

Usage:
    python cli.py list
    python cli.py adversarial_prompt --model_type openai --config configs/adversarial_prompt/benchmark_config.yaml ...
    python cli.py map_navigation --config configs/map_navigation/model_config.py --model gpt-4 --model-type api ...

The CLI discovers task scripts under /tasks automatically (defaulting to each
task's `main.py`) and runs them in a subprocess with the task directory as cwd
so relative imports and config paths continue to work as in task-specific runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parent
TASKS_ROOT = ROOT / "tasks"


def discover_tasks() -> Dict[str, Path]:
    """
    Discover runnable task entrypoints.
    - Prefers <task>/main.py
    - Adds a Wordle alias pointing to run_full_benchmark.py if present.
    """
    tasks: Dict[str, Path] = {}

    search_roots = [TASKS_ROOT] if TASKS_ROOT.exists() else [ROOT]
    for base in search_roots:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            candidate = child / "main.py"
            if candidate.exists():
                tasks[child.name] = candidate

    # Wordle has multiple entry scripts; pick the full benchmark as default alias.
    wordle_dir = TASKS_ROOT / "wordle" if TASKS_ROOT.exists() else ROOT / "wordle"
    if wordle_dir.exists():
        wf = wordle_dir / "run_full_benchmark.py"
        if wf.exists():
            tasks.setdefault("wordle", wf)

    return tasks


def run_task(task: str, task_args: list[str], tasks: Dict[str, Path]) -> int:
    """Run the selected task script in a subprocess with its own working directory."""
    script = tasks.get(task)
    if not script:
        available = ", ".join(sorted(tasks))
        print(f"Unknown task '{task}'. Available tasks: {available}")
        return 1

    cmd = [sys.executable, str(script), *task_args]
    env = os.environ.copy()
    pythonpath_entries = [str(ROOT)]
    if existing := env.get("PYTHONPATH"):
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    print(f"Executing: {' '.join(cmd)} (cwd={script.parent})")
    result = subprocess.run(cmd, cwd=script.parent, env=env)
    return result.returncode


def main():
    tasks = discover_tasks()

    parser = argparse.ArgumentParser(
        description="TokenBench unified runner for all tasks",
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    # list command
    subparsers.add_parser("list", help="List available tasks")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a specific task",
        description="Run a task; arguments after '--' are forwarded to the task script",
    )
    run_parser.add_argument("task", help=f"Task name ({', '.join(sorted(tasks))})")
    run_parser.add_argument(
        "task_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the task's main script (prefix with '--' to stop cli.py parsing)",
    )

    # Backward compatibility: allow `python cli.py <task> ...` without `run`
    parser.add_argument(
        "legacy_task",
        nargs="?",
        help="Task name (legacy mode: omitting 'run' command)",
    )
    parser.add_argument(
        "legacy_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the task script in legacy mode",
    )

    args = parser.parse_args()

    if args.command == "list":
        print("Available tasks:")
        for name in sorted(tasks):
            print(f"  - {name} -> {tasks[name].relative_to(ROOT)}")
        return

    # Legacy direct mode: python cli.py task --args
    if args.command is None and args.legacy_task:
        rc = run_task(args.legacy_task, args.legacy_args, tasks)
        sys.exit(rc)

    if args.command == "run":
        task_args = args.task_args
        # Strip a leading '--' that argparse leaves in the remainder
        if task_args and task_args[0] == "--":
            task_args = task_args[1:]
        rc = run_task(args.task, task_args, tasks)
        sys.exit(rc)

    parser.print_help()


if __name__ == "__main__":
    main()
