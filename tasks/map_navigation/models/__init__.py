import importlib
import sys
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


def _load_shared_models():
    project_root = find_project_root()
    original_sys_path = list(sys.path)
    try:
        task_dir = str(Path(__file__).resolve().parent)
        sys.path = [str(project_root)] + [p for p in sys.path if p != task_dir]
        return importlib.import_module("models")
    finally:
        sys.path = original_sys_path


_shared_models = _load_shared_models()
BaseModel = _shared_models.BaseModel
VLLMModel = _shared_models.VLLMModel
APIModel = _shared_models.APIModel

__all__ = ["BaseModel", "VLLMModel", "APIModel"]
