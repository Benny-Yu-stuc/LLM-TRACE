"""Lightweight traceability metrics."""

from __future__ import annotations

from llm_trace.schemas import Evidence, ExtractedEntity, ExtractedRelation, OperationChainGraph


def compute_metrics(
    *,
    evidence: list[Evidence],
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    graph: OperationChainGraph,
    alignment_scores: dict[str, float],
) -> dict:
    evidence_ids = {item.evidence_id for item in evidence}
    anchored_entities = [entity for entity in entities if entity.evidence_id in evidence_ids]
    anchored_relations = [
        relation for relation in relations if relation.evidence_ids and all(eid in evidence_ids for eid in relation.evidence_ids)
    ]
    qc_entities = [
        entity
        for entity in entities
        if entity.entity_type in {"QUALITY_INDICATOR", "QUALITY_STANDARD", "DEFECT", "CAUSE"}
    ]
    op_ids = {item.op_id for item in evidence if item.op_id not in {"GLOBAL", "OP_UNKNOWN"}}

    return {
        "evidence_count": len(evidence),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "traceability": {
            "entity_traceability_rate": round(len(anchored_entities) / max(1, len(entities)), 4),
            "relation_traceability_rate": round(len(anchored_relations) / max(1, len(relations)), 4),
        },
        "business": {
            "quality_control_item_count": len(qc_entities),
            "operation_count": len(op_ids),
            "operation_chain_integrity": round(_chain_integrity(op_ids, graph), 4),
        },
        "tm_cgb": {
            "graph_node_count": len(graph.nodes),
            "graph_edge_count": len(graph.edges),
            "graph_consistency_score": graph.consistency_score,
            "contrastive_alignment": alignment_scores,
        },
    }


def _chain_integrity(op_ids: set[str], graph: OperationChainGraph) -> float:
    if len(op_ids) <= 1:
        return 1.0 if op_ids else 0.0
    sequence_edges = sum(1 for edge in graph.edges if edge.edge_type == "operation_sequence")
    return min(1.0, sequence_edges / (len(op_ids) - 1))
