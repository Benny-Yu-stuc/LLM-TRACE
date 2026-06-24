"""Modality-specific expert encoders.

These encoders are deterministic placeholders. They keep the same interfaces
that a production BERT, ViT, symbol encoder, or layout-aware model would use.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

from llm_trace.schemas import EncodedEvidence, Evidence, Modality


class BaseExpert:
    def __init__(self, modality: Modality, vector_dim: int = 64) -> None:
        self.modality = modality
        self.vector_dim = vector_dim

    def encode(self, evidence: Iterable[Evidence]) -> list[EncodedEvidence]:
        encoded: list[EncodedEvidence] = []
        for item in evidence:
            if item.modality != self.modality:
                continue
            encoded.append(
                EncodedEvidence(
                    evidence_id=item.evidence_id,
                    modality=item.modality,
                    op_id=item.op_id,
                    vector=hash_text_vector(
                        text=f"{item.field_name} {item.text} {item.unit}",
                        dim=self.vector_dim,
                        salt=self.modality,
                    ),
                    confidence=item.confidence,
                    text=item.text,
                    metadata=item.metadata,
                )
            )
        return encoded


class ProcessExpert(BaseExpert):
    def __init__(self, vector_dim: int = 64) -> None:
        super().__init__("table", vector_dim)


class GeometryExpert(BaseExpert):
    def __init__(self, vector_dim: int = 64) -> None:
        super().__init__("drawing", vector_dim)


class SymbolExpert(BaseExpert):
    def __init__(self, vector_dim: int = 64) -> None:
        super().__init__("symbol", vector_dim)


def hash_text_vector(text: str, dim: int, salt: str = "") -> list[float]:
    """Create a stable normalized vector from text."""

    vector = [0.0] * dim
    tokens = [token for token in _tokenize(text) if token]
    if not tokens:
        tokens = ["empty"]
    for token in tokens:
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    separators = ",;:/\\|()[]{}<>=\"'\t\r\n"
    normalized = text.lower()
    for separator in separators:
        normalized = normalized.replace(separator, " ")
    return normalized.split()
