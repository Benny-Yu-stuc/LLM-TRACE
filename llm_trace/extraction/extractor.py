"""Combined MNER and MRE extractor."""

from __future__ import annotations

from llm_trace.extraction.entity_extractor import EntityExtractor
from llm_trace.extraction.relation_extractor import RelationExtractor
from llm_trace.schemas import Evidence, ExtractedEntity, ExtractedRelation


class MultiModalExtractor:
    def __init__(self, config: dict) -> None:
        self.entity_extractor = EntityExtractor(float(config.get("entity_confidence_threshold", 0.35)))
        self.relation_extractor = RelationExtractor(float(config.get("relation_confidence_threshold", 0.35)))

    def extract(self, document_id: str, evidence: list[Evidence]) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        entities = self.entity_extractor.extract(document_id, evidence)
        relations = self.relation_extractor.extract(document_id, entities)
        return entities, relations
