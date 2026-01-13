"""
Model implementations for various LLM providers.
"""

from src.models.base_model import BaseModel
from pathlib import Path
import sys


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


# Ensure project root is on sys.path so the shared /models package is importable
PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import VLLMModel, OpenAIModel, DeepSeekModel, BaseModel  # type: ignore

__all__ = ["VLLMModel", "OpenAIModel", "DeepSeekModel", "BaseModel"]

__all__ = ['BaseModel', 'VLLMModel', 'OpenAIModel', 'DeepSeekModel']
