"""Configuration loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    root = config_path.parent.parent
    config["_root"] = str(root.resolve())
    config["data"] = {
        key: str((root / value).resolve()) if isinstance(value, str) else value
        for key, value in config.get("data", {}).items()
    }

    chief = config.setdefault("chief_engineer", {})
    chief["api_key"] = os.getenv("CHIEF_ENGINEER_API_KEY", chief.get("api_key", ""))
    chief["api_url"] = os.getenv("CHIEF_ENGINEER_API_URL", chief.get("api_url", ""))
    chief["model"] = os.getenv("CHIEF_ENGINEER_MODEL", chief.get("model", ""))
    if chief.get("cache_dir"):
        chief["cache_dir"] = str((root / chief["cache_dir"]).resolve())
    return config


def ensure_dirs(paths: list[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
