"""Temporal Multimodal Contrastive Graph Booster scaffold."""

from __future__ import annotations

from collections import defaultdict

from llm_trace.schemas import (
    CandidateConstraint,
    Evidence,
    ExtractedEntity,
    ExtractedRelation,
    GraphEdge,
    GraphNode,
    OperationChainGraph,
    operation_sort_key,
)


class TemporalMultimodalContrastiveGraphBooster:
    """Build and score an operation-chain graph.

    This is a deterministic graph builder. It preserves the manuscript's
    graph types and edge categories while keeping training optional.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def build(
        self,
        *,
        document_id: str,
        evidence: list[Evidence],
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        constraints: list[CandidateConstraint],
    ) -> OperationChainGraph:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_nodes: set[str] = set()

        op_ids = sorted({item.op_id for item in evidence if item.op_id != "GLOBAL"}, key=operation_sort_key)
        for op_id in op_ids:
            node = GraphNode(node_id=f"op:{op_id}", node_type="operation", label=op_id, op_id=op_id)
            _add_node(nodes, seen_nodes, node)

        for left, right in zip(op_ids, op_ids[1:]):
            edges.append(GraphEdge(source=f"op:{left}", target=f"op:{right}", edge_type="operation_sequence", weight=1.0))

        for item in evidence:
            node_id = f"evidence:{item.evidence_id}"
            _add_node(
                nodes,
                seen_nodes,
                GraphNode(
                    node_id=node_id,
                    node_type="evidence",
                    label=item.text[:80],
                    op_id=item.op_id,
                    evidence_id=item.evidence_id,
                    attributes={"modality": item.modality, "confidence": item.confidence},
                ),
            )
            if item.op_id != "GLOBAL":
                edges.append(
                    GraphEdge(
                        source=f"op:{item.op_id}",
                        target=node_id,
                        edge_type="operation_has_evidence",
                        weight=item.confidence,
                        evidence_ids=[item.evidence_id],
                    )
                )

        for entity in entities:
            entity_node = GraphNode(
                node_id=f"entity:{entity.entity_id}",
                node_type="entity",
                label=entity.text,
                op_id=entity.op_id,
                evidence_id=entity.evidence_id,
                attributes={"entity_type": entity.entity_type, "confidence": entity.confidence},
            )
            _add_node(nodes, seen_nodes, entity_node)
            edges.append(
                GraphEdge(
                    source=f"op:{entity.op_id}",
                    target=entity_node.node_id,
                    edge_type="operation_owns_entity",
                    weight=entity.confidence,
                    evidence_ids=[entity.evidence_id],
                )
            )
            edges.append(
                GraphEdge(
                    source=f"evidence:{entity.evidence_id}",
                    target=entity_node.node_id,
                    edge_type="evidence_supports_entity",
                    weight=entity.confidence,
                    evidence_ids=[entity.evidence_id],
                )
            )

        entity_by_id = {entity.entity_id: entity for entity in entities}
        for relation in relations:
            head = entity_by_id.get(relation.head_entity_id)
            tail = entity_by_id.get(relation.tail_entity_id)
            if not head or not tail:
                continue
            edges.append(
                GraphEdge(
                    source=f"entity:{head.entity_id}",
                    target=f"entity:{tail.entity_id}",
                    edge_type=f"candidate_relation:{relation.relation_type}",
                    weight=relation.confidence,
                    evidence_ids=relation.evidence_ids,
                )
            )

        for index, constraint in enumerate(constraints, start=1):
            node_id = f"rule:{index:04d}"
            _add_node(
                nodes,
                seen_nodes,
                GraphNode(
                    node_id=node_id,
                    node_type="rule_constraint",
                    label=constraint.description[:100],
                    attributes={"constraint_type": constraint.constraint_type},
                ),
            )
            for evidence_id in constraint.related_evidence:
                edges.append(
                    GraphEdge(
                        source=node_id,
                        target=f"evidence:{evidence_id}",
                        edge_type="rule_constrains_evidence",
                        weight=1.0,
                        evidence_ids=[evidence_id],
                    )
                )

        graph = OperationChainGraph(document_id=document_id, nodes=nodes, edges=edges)
        graph.consistency_score = self._score(graph)
        return graph

    def contrastive_alignment_score(self, evidence: list[Evidence]) -> dict[str, float]:
        """Compute an interpretable intra-OP multimodal coverage proxy."""

        by_op: dict[str, set[str]] = defaultdict(set)
        for item in evidence:
            if item.modality in {"table", "drawing", "symbol"}:
                by_op[item.op_id].add(item.modality)
        scores = {op_id: len(modalities) / 3.0 for op_id, modalities in by_op.items()}
        if scores:
            scores["macro_average"] = sum(scores.values()) / len(scores)
        else:
            scores["macro_average"] = 0.0
        return scores

    @staticmethod
    def _score(graph: OperationChainGraph) -> float:
        if not graph.nodes:
            return 0.0
        edge_density = min(1.0, len(graph.edges) / max(1, len(graph.nodes) * 2))
        evidence_edges = sum(1 for edge in graph.edges if edge.evidence_ids)
        traceability = evidence_edges / max(1, len(graph.edges))
        sequence_edges = sum(1 for edge in graph.edges if edge.edge_type == "operation_sequence")
        sequence_bonus = min(1.0, sequence_edges / max(1, len({node.op_id for node in graph.nodes if node.op_id}) - 1))
        return round(0.4 * edge_density + 0.4 * traceability + 0.2 * sequence_bonus, 4)


def _add_node(nodes: list[GraphNode], seen_nodes: set[str], node: GraphNode) -> None:
    if node.node_id in seen_nodes:
        return
    seen_nodes.add(node.node_id)
    nodes.append(node)
