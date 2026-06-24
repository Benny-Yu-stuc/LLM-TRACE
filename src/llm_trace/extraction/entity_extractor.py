"""Neural MNER decoding with evidence anchors."""

from __future__ import annotations

from llm_trace.schemas import Evidence, ExtractedEntity, FusedOperation


class EntityExtractor:
    def __init__(self, confidence_threshold: float = 0.35, labels: list[str] | None = None) -> None:
        self.confidence_threshold = confidence_threshold
        self.labels = labels or []

    def extract(
        self,
        document_id: str,
        evidence: list[Evidence],
        fused_operations: list[FusedOperation],
        entity_logits,
    ) -> list[ExtractedEntity]:
        import torch

        evidence_by_id = {item.evidence_id: item for item in evidence}
        entities: list[ExtractedEntity] = []
        offset = 0
        probabilities = torch.softmax(entity_logits, dim=-1)
        label_ids = torch.argmax(probabilities, dim=-1)
        for fused in fused_operations:
            for local_index, evidence_id in enumerate(fused.evidence_ids):
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    continue
                global_index = offset + local_index
                label_id = int(label_ids[global_index].detach().cpu())
                label = self.labels[label_id]
                confidence = float(probabilities[global_index, label_id].detach().cpu())
                if label == "O" or confidence < self.confidence_threshold:
                    continue
                entity_type = _strip_bio(label)
                entity_id = f"ent_{len(entities) + 1:05d}"
                entities.append(
                    ExtractedEntity(
                        entity_id=entity_id,
                        document_id=document_id,
                        op_id=item.op_id,
                        entity_type=entity_type,
                        text=item.value or item.text,
                        evidence_id=item.evidence_id,
                        confidence=round(confidence, 4),
                        attributes={
                            "decoder": "neural_mner",
                            "label": label,
                            "token_index": global_index,
                            "field_name": item.field_name,
                        },
                    )
                )
            offset += len(fused.evidence_ids)
        return entities


def _strip_bio(label: str) -> str:
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label
