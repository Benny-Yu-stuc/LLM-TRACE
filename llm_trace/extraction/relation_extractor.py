"""Evidence-anchored relation extraction."""

from __future__ import annotations

from collections import defaultdict

from llm_trace.schemas import ExtractedEntity, ExtractedRelation


class RelationExtractor:
    def __init__(self, confidence_threshold: float = 0.35) -> None:
        self.confidence_threshold = confidence_threshold

    def extract(self, document_id: str, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        by_op: dict[str, list[ExtractedEntity]] = defaultdict(list)
        for entity in entities:
            by_op[entity.op_id].append(entity)

        relations: list[ExtractedRelation] = []
        for op_id, op_entities in by_op.items():
            relations.extend(self._within_operation(document_id, op_id, op_entities, len(relations)))

        relations.extend(self._operation_sequence(document_id, entities, len(relations)))
        return [relation for relation in relations if relation.confidence >= self.confidence_threshold]

    def _within_operation(
        self,
        document_id: str,
        op_id: str,
        entities: list[ExtractedEntity],
        offset: int,
    ) -> list[ExtractedRelation]:
        result: list[ExtractedRelation] = []
        operations = [entity for entity in entities if entity.entity_type == "OPERATION"]
        equipment = [entity for entity in entities if entity.entity_type == "EQUIPMENT"]
        parameters = [entity for entity in entities if entity.entity_type == "PROCESS_PARAMETER"]
        quality = [
            entity
            for entity in entities
            if entity.entity_type in {"QUALITY_INDICATOR", "QUALITY_STANDARD", "DEFECT", "CAUSE"}
        ]
        materials = [entity for entity in entities if entity.entity_type == "MATERIAL"]
        parts = [entity for entity in entities if entity.entity_type == "PRODUCT_PART"]

        for left in equipment:
            for right in parameters:
                result.append(self._make(document_id, op_id, "equipment_parameter", left, right, offset + len(result)))

        for left in parameters:
            for right in quality:
                result.append(self._make(document_id, op_id, "parameter_quality", left, right, offset + len(result)))

        for left in materials:
            for right in parts:
                result.append(self._make(document_id, op_id, "material_product", left, right, offset + len(result)))

        for op in operations[:1]:
            for entity in equipment + parameters + quality:
                if entity.entity_id != op.entity_id:
                    result.append(self._make(document_id, op_id, "operation_owns_entity", op, entity, offset + len(result)))
        return result

    def _operation_sequence(
        self,
        document_id: str,
        entities: list[ExtractedEntity],
        offset: int,
    ) -> list[ExtractedRelation]:
        operations = sorted(
            {entity.text: entity for entity in entities if entity.entity_type == "OPERATION"}.values(),
            key=lambda entity: _op_sort(entity.text),
        )
        result: list[ExtractedRelation] = []
        for left, right in zip(operations, operations[1:]):
            result.append(self._make(document_id, left.op_id, "operation_sequence", left, right, offset + len(result), 0.95))
        return result

    def _make(
        self,
        document_id: str,
        op_id: str,
        relation_type: str,
        head: ExtractedEntity,
        tail: ExtractedEntity,
        index: int,
        confidence: float | None = None,
    ) -> ExtractedRelation:
        score = confidence if confidence is not None else round((head.confidence + tail.confidence) / 2.0, 4)
        return ExtractedRelation(
            relation_id=f"rel_{index + 1:05d}",
            document_id=document_id,
            op_id=op_id,
            relation_type=relation_type,
            head_entity_id=head.entity_id,
            tail_entity_id=tail.entity_id,
            evidence_ids=sorted({head.evidence_id, tail.evidence_id}),
            confidence=score,
        )


def _op_sort(op_id: str) -> int:
    digits = "".join(char for char in op_id if char.isdigit())
    return int(digits) if digits else 999999
