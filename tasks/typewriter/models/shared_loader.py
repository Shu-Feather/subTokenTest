import importlib
import sys
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


def load_shared_models():
    """Load the repository-level models module, avoiding local shadowing."""
    project_root = find_project_root()
    original_sys_path = list(sys.path)
    try:
        local_dir = str(Path(__file__).resolve().parent)
        sys.path = [str(project_root)] + [p for p in sys.path if p != local_dir]
        return importlib.import_module("models")
    finally:
        sys.path = original_sys_path
