"""Evidence-anchored entity extraction."""

from __future__ import annotations

import re

from llm_trace.schemas import Evidence, ExtractedEntity


class EntityExtractor:
    def __init__(self, confidence_threshold: float = 0.35) -> None:
        self.confidence_threshold = confidence_threshold

    def extract(self, document_id: str, evidence: list[Evidence]) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in evidence:
            for entity_type, text, confidence, attributes in self._candidates(item):
                if confidence < self.confidence_threshold:
                    continue
                key = (item.op_id, entity_type, text.lower(), item.evidence_id)
                if key in seen:
                    continue
                seen.add(key)
                entity_id = f"ent_{len(entities) + 1:05d}"
                entities.append(
                    ExtractedEntity(
                        entity_id=entity_id,
                        document_id=document_id,
                        op_id=item.op_id,
                        entity_type=entity_type,
                        text=text,
                        evidence_id=item.evidence_id,
                        confidence=round(confidence, 4),
                        attributes=attributes,
                    )
                )
        return entities

    def _candidates(self, item: Evidence) -> list[tuple[str, str, float, dict]]:
        text = item.text.strip()
        lower = text.lower()
        candidates: list[tuple[str, str, float, dict]] = []
        field = item.field_name.lower()

        type_by_field = [
            ("operation", "OPERATION"),
            ("equipment", "EQUIPMENT"),
            ("mold", "TOOLING"),
            ("tool", "TOOLING"),
            ("parameter", "PROCESS_PARAMETER"),
            ("pressure", "PROCESS_PARAMETER"),
            ("speed", "PROCESS_PARAMETER"),
            ("material", "MATERIAL"),
            ("part", "PRODUCT_PART"),
            ("quality", "QUALITY_INDICATOR"),
            ("standard", "QUALITY_STANDARD"),
            ("defect", "DEFECT"),
            ("cause", "CAUSE"),
            ("person", "PERSONNEL"),
        ]
        for keyword, entity_type in type_by_field:
            if keyword in field or keyword in lower:
                candidates.append((entity_type, _clean_value(item.value or text), item.confidence, {"source": "keyword"}))
                break

        if item.op_id != "OP_UNKNOWN":
            candidates.append(("OPERATION", item.op_id, max(item.confidence, 0.8), {"source": "op_id"}))

        for number, unit in re.findall(r"([+-]?\d+(?:\.\d+)?)\s*(mm|mpa|kn|n|s|sec|%)", lower):
            candidates.append(
                (
                    "PROCESS_PARAMETER",
                    f"{number} {unit}",
                    min(1.0, item.confidence + 0.1),
                    {"source": "numeric_unit", "unit": unit},
                )
            )

        if item.modality == "symbol":
            symbol_type = item.field_name or "symbol"
            if symbol_type.lower() in {"sc", "cc", "qc", "quality"} or any(token in lower for token in ["sc", "cc", "qc"]):
                candidates.append(("QUALITY_INDICATOR", text, item.confidence, {"source": "symbol"}))
            else:
                candidates.append(("SYMBOL", text, item.confidence, {"source": "symbol"}))

        if item.modality == "manual":
            candidates.append(("QUALITY_STANDARD", text[:120], item.confidence * 0.8, {"source": "manual_clause"}))

        return [(entity_type, value, score, attrs) for entity_type, value, score, attrs in candidates if value]


def _clean_value(value: str) -> str:
    value = " ".join(value.split())
    return value[:160]
