"""Model backend loading utilities."""

from __future__ import annotations

from typing import Any


def require_backend(strict: bool, *packages: str) -> None:
    missing = [package for package in packages if not _can_import(package)]
    if missing:
        raise ImportError(f"Missing required model dependencies: {', '.join(missing)}")


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


class ModelLoadError(RuntimeError):
    """Raised when a configured model path or checkpoint cannot be loaded."""


def import_torch() -> Any:
    import torch

    return torch


def import_transformers() -> Any:
    import transformers

    return transformers
