"""Linear probe experiments for token representations."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def get_base_dir() -> Path:
    """Return the root directory for the linear_probe experiment assets."""
    return BASE_DIR
