"""Combined MNER and MRE extractor."""

from __future__ import annotations

from llm_trace.models.heads import NeuralExtractionHeads
from llm_trace.extraction.entity_extractor import EntityExtractor
from llm_trace.extraction.relation_extractor import RelationExtractor
from llm_trace.schemas import Evidence, ExtractedEntity, ExtractedRelation, FusedOperation


class MultiModalExtractor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.entity_label_schema = list(config.get("entity_label_schema", []))
        self.relation_label_schema = list(config.get("relation_label_schema", []))
        self.entity_extractor = EntityExtractor(
            float(config.get("entity_confidence_threshold", 0.35)),
            labels=self.entity_label_schema,
        )
        self.relation_extractor = RelationExtractor(
            float(config.get("relation_confidence_threshold", 0.35)),
            labels=self.relation_label_schema,
        )
        self.neural_heads = NeuralExtractionHeads(
            hidden_dim=int(config.get("hidden_dim", 768)),
            entity_labels=self.entity_label_schema,
            relation_labels=self.relation_label_schema,
        )

    def extract(
        self,
        document_id: str,
        evidence: list[Evidence],
        fused_operations: list[FusedOperation],
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        import torch

        token_vectors = [vector for fused in fused_operations for vector in fused.token_vectors]
        if not token_vectors:
            return [], []
        fused_tensor = torch.tensor(token_vectors, dtype=torch.float32)
        entity_logits = self.neural_heads.entity_logits(fused_tensor)
        entities = self.entity_extractor.extract(document_id, evidence, fused_operations, entity_logits)

        candidate_pairs = _candidate_pairs(entities)
        if not candidate_pairs:
            return entities, []
        context_repr = fused_tensor.mean(dim=0, keepdim=True)
        head_repr = torch.stack([fused_tensor[_entity_token_index(entities[i])] for i, _ in candidate_pairs])
        tail_repr = torch.stack([fused_tensor[_entity_token_index(entities[j])] for _, j in candidate_pairs])
        context = context_repr.repeat(len(candidate_pairs), 1)
        relation_logits = self.neural_heads.relation_logits(head_repr, tail_repr, context)
        relations = self.relation_extractor.extract(document_id, entities, relation_logits, candidate_pairs)
        return entities, relations


def _candidate_pairs(entities: list[ExtractedEntity]) -> list[tuple[int, int]]:
    pairs = []
    for head_index, head in enumerate(entities):
        for tail_index, tail in enumerate(entities):
            if head_index == tail_index:
                continue
            if head.op_id == tail.op_id or head.entity_type == "OPERATION" or tail.entity_type == "OPERATION":
                pairs.append((head_index, tail_index))
    return pairs


def _entity_token_index(entity: ExtractedEntity) -> int:
    return int(entity.attributes.get("token_index", 0))
