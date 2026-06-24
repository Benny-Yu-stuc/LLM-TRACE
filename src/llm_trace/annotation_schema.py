"""Annotation schema validation for process-card samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_trace.schemas import Evidence


REQUIRED_ENTITY_KEYS = {"entity_type", "text", "source_parser"}
REQUIRED_RELATION_KEYS = {"relation_type", "head_text", "tail_text", "evidence_sources"}


def validate_annotation_sample(sample: dict[str, Any], evidence: list[Evidence]) -> list[str]:
    errors: list[str] = []
    by_field = {item.field_name: item for item in evidence}
    by_cell = {item.location.cell: item for item in evidence if item.location.cell}
    by_region = {item.metadata.get("region_id"): item for item in evidence if item.metadata.get("region_id")}
    by_symbol = {item.metadata.get("symbol_id"): item for item in evidence if item.metadata.get("symbol_id")}

    for index, entity in enumerate(sample.get("entity_spans", []), start=1):
        missing = REQUIRED_ENTITY_KEYS - set(entity)
        if missing:
            errors.append(f"entity_spans[{index}] missing keys: {sorted(missing)}")
        source_parser = entity.get("source_parser")
        if source_parser == "WorkbookParser" and entity.get("source_field") not in by_field:
            errors.append(f"entity_spans[{index}] source_field not parsed: {entity.get('source_field')}")
        if source_parser == "WorkbookParser" and entity.get("cell") not in by_cell:
            errors.append(f"entity_spans[{index}] cell not parsed: {entity.get('cell')}")
        if source_parser == "ImageEvidenceParser" and entity.get("region_id") not in by_region:
            errors.append(f"entity_spans[{index}] region_id not parsed: {entity.get('region_id')}")
        if source_parser == "SymbolParser" and entity.get("symbol_id") not in by_symbol:
            errors.append(f"entity_spans[{index}] symbol_id not parsed: {entity.get('symbol_id')}")

    for index, relation in enumerate(sample.get("relation_triples", []), start=1):
        missing = REQUIRED_RELATION_KEYS - set(relation)
        if missing:
            errors.append(f"relation_triples[{index}] missing keys: {sorted(missing)}")
        if not relation.get("evidence_sources"):
            errors.append(f"relation_triples[{index}] has no evidence sources")

    image_files = [
        item.location.source_file
        for item in evidence
        if item.modality in {"drawing", "symbol"} and item.location.source_file
    ]
    for source_file in image_files:
        if not Path(source_file).exists() and not Path("src", source_file).exists() and not Path.cwd().joinpath(source_file).exists():
            errors.append(f"image evidence source does not exist: {source_file}")
    return errors
