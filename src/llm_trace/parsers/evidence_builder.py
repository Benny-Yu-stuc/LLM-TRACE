"""Factory helpers for evidence units."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from llm_trace.schemas import Evidence, Modality, SourceLocation


def stable_id(*parts: object, prefix: str = "ev") -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_evidence(
    *,
    document_id: str,
    op_id: str,
    modality: Modality,
    text: str,
    source_file: str,
    field_name: str = "",
    value: str = "",
    unit: str = "",
    sheet_name: str = "",
    cell: str = "",
    bbox: list[float] | None = None,
    page_index: int | None = None,
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    evidence_id = stable_id(document_id, op_id, modality, source_file, sheet_name, cell, text)
    location = SourceLocation(
        document_id=document_id,
        op_id=op_id,
        source_type=modality,
        source_file=str(Path(source_file)),
        sheet_name=sheet_name,
        cell=cell,
        bbox=bbox or [],
        page_index=page_index,
    )
    return Evidence(
        evidence_id=evidence_id,
        document_id=document_id,
        op_id=op_id,
        modality=modality,
        text=text.strip(),
        location=location,
        field_name=field_name,
        value=value,
        unit=unit,
        confidence=confidence,
        metadata=metadata or {},
    )
