"""Neural MRE decoding with evidence anchors."""

from __future__ import annotations

from llm_trace.schemas import ExtractedEntity, ExtractedRelation


class RelationExtractor:
    def __init__(self, confidence_threshold: float = 0.35, labels: list[str] | None = None) -> None:
        self.confidence_threshold = confidence_threshold
        self.labels = labels or []

    def extract(self, document_id: str, entities: list[ExtractedEntity], relation_logits, candidate_pairs: list[tuple[int, int]]) -> list[ExtractedRelation]:
        import torch

        probabilities = torch.softmax(relation_logits, dim=-1)
        label_ids = torch.argmax(probabilities, dim=-1)
        relations: list[ExtractedRelation] = []
        for index, (head_index, tail_index) in enumerate(candidate_pairs):
            label_id = int(label_ids[index].detach().cpu())
            label = self.labels[label_id]
            confidence = float(probabilities[index, label_id].detach().cpu())
            if label == "no_relation" or confidence < self.confidence_threshold:
                continue
            head = entities[head_index]
            tail = entities[tail_index]
            relations.append(
                ExtractedRelation(
                    relation_id=f"rel_{len(relations) + 1:05d}",
                    document_id=document_id,
                    op_id=head.op_id,
                    relation_type=label,
                    head_entity_id=head.entity_id,
                    tail_entity_id=tail.entity_id,
                    evidence_ids=sorted({head.evidence_id, tail.evidence_id}),
                    confidence=round(confidence, 4),
                    attributes={"decoder": "neural_mre", "candidate_pair": [head_index, tail_index]},
                )
            )
        return relations
