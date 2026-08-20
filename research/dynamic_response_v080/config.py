from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform
import sys
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "dynamic_response_v080.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the V0.8 research configuration."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = ["strategy", "data", "walk_forward", "features", "surface_models", "hmm"]
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"V0.8 config missing sections: {missing}")
    if not config["strategy"].get("research_only", False):
        raise ValueError("V0.8 must remain research_only")
    if config["strategy"].get("production_alert_integration", True):
        raise ValueError("V0.8 cannot integrate with production alerts")
    config["_config_path"] = str(config_path)
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def output_dir(config: dict[str, Any]) -> Path:
    path = resolve_project_path(config["strategy"]["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def environment_manifest() -> dict[str, Any]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "pyarrow",
        "lightgbm",
        "xgboost",
        "hmmlearn",
        "shap",
        "optuna",
        "matplotlib",
        "seaborn",
        "PyYAML",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }
