"""
Helpers to resolve per-task config files from the centralized /configs directory.

Usage:
    from configs.locator import resolve_config_path
    cfg_path = resolve_config_path("adversarial_prompt", "benchmark_config.yaml", fallback="config/benchmark_config.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


CONFIG_ROOT = Path(__file__).resolve().parent


def resolve_config_path(task: str, filename: str, fallback: Optional[str] = None) -> str:
    """
    Resolve a config file under the central configs directory, falling back to an existing path if needed.

    Args:
        task: Task name (matches subdirectory under /configs).
        filename: Relative filename inside the task config folder.
        fallback: Optional fallback path to use if the centralized copy is missing.
    """
    provided = Path(filename)

    # Absolute path: use directly
    if provided.is_absolute():
        return str(provided)

    # Provided path relative to current working directory
    if provided.exists():
        return str(provided.resolve())

    # Path relative to repository root (handles inputs like 'configs/<task>/config.yaml')
    repo_candidate = (CONFIG_ROOT.parent / provided).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    # Centralized configs/<task>/<filename>
    task_dir = CONFIG_ROOT / task
    candidate = (task_dir / provided).resolve()
    if candidate.exists():
        return str(candidate)

    # Fall back to matching only the basename inside the task config folder
    name_candidate = (task_dir / provided.name).resolve()
    if name_candidate.exists():
        return str(name_candidate)

    if fallback:
        return fallback
    return str(candidate)
