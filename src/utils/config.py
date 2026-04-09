"""
src/utils/config.py
-------------------
Loads the YAML configuration file and resolves paths.
Every other module imports from here — no hardcoded paths anywhere.

Usage:
    from src.utils.config import load_config, resolve_path
    cfg = load_config()
    lat = cfg["site"]["latitude"]
"""

from pathlib import Path
import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "site.yaml"


def load_config(config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    """Load YAML config and return as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return absolute path to the project root."""
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """Convert a config-relative path string to an absolute Path."""
    return get_project_root() / relative_path
