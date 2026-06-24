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
    if chief.get("seed_cache_path") and not Path(chief["seed_cache_path"]).is_absolute():
        chief["seed_cache_path"] = str((root / chief["seed_cache_path"]).resolve())

    fusion = config.setdefault("fusion", {})
    process_expert = fusion.setdefault("process_expert", {})
    geometry_expert = fusion.setdefault("geometry_expert", {})
    symbol_expert = fusion.setdefault("symbol_expert", {})
    process_expert["bert_model_name_or_path"] = os.getenv(
        "BERT_MODEL_NAME_OR_PATH", process_expert.get("bert_model_name_or_path", "")
    )
    process_expert["tokenizer_name_or_path"] = os.getenv(
        "BERT_TOKENIZER_NAME_OR_PATH", process_expert.get("tokenizer_name_or_path", "")
    )
    geometry_expert["vit_model_name_or_path"] = os.getenv(
        "VIT_MODEL_NAME_OR_PATH", geometry_expert.get("vit_model_name_or_path", "")
    )
    geometry_expert["image_processor_name_or_path"] = os.getenv(
        "VIT_IMAGE_PROCESSOR_NAME_OR_PATH", geometry_expert.get("image_processor_name_or_path", "")
    )
    symbol_expert["symbol_vocab_path"] = os.getenv("SYMBOL_VOCAB_PATH", symbol_expert.get("symbol_vocab_path", ""))
    if symbol_expert.get("symbol_vocab_path") and not Path(symbol_expert["symbol_vocab_path"]).is_absolute():
        symbol_expert["symbol_vocab_path"] = str((root / symbol_expert["symbol_vocab_path"]).resolve())
    return config


def ensure_dirs(paths: list[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
