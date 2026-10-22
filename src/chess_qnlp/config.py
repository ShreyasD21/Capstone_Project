"""Load a YAML config and resolve its paths against the repository root."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: ROOT / value for key, value in cfg["paths"].items()}
    cfg["config_path"] = str(path)
    return cfg
