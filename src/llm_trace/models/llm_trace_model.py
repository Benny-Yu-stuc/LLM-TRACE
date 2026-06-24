"""Full LLM-TRACE model assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_trace.fusion.dpfc import DomainPriorFusionCorroboration
from llm_trace.models.graph_encoder import RelationAwareGraphEncoder
from llm_trace.models.heads import NeuralExtractionHeads
from llm_trace.models.losses import LlmTraceLosses
from llm_trace.schemas import ChiefEngineerOutput, Evidence


@dataclass(slots=True)
class ModelForwardOutput:
    encoded_evidence: dict[str, Any]
    fused_operations: dict[str, Any]
    entity_logits: Any | None
    relation_logits: Any | None
    graph_node_repr: Any | None
    losses: dict[str, Any]
    total_loss: Any | None


class LlmTraceModel:
    """Complete article-level model.

    Components:
    - BERT Process Expert for table text and manual clauses.
    - ViT Geometry Expert for process sketches and image regions.
    - Structured Symbol Expert for symbols, bbox, confidence, and evidence IDs.
    - PMCA prior-modulated cross-modal corroborative fusion.
    - MNER and MRE neural heads.
    - Relation-aware operation-chain graph encoder.
    - Joint L_MNER, L_MRE, L_contrastive, L_graph objective.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.fusion = DomainPriorFusionCorroboration(config.get("fusion", {}))
        extraction_config = config.get("extraction", {})
        graph_config = config.get("graph", {})
        self.heads = NeuralExtractionHeads(
            hidden_dim=int(extraction_config.get("hidden_dim", config.get("fusion", {}).get("hidden_dim", 768))),
            entity_labels=list(extraction_config.get("entity_label_schema", [])),
            relation_labels=list(extraction_config.get("relation_label_schema", [])),
        )
        self.graph_encoder = RelationAwareGraphEncoder(
            hidden_dim=int(graph_config.get("graph_hidden_dim", 768)),
            edge_types=list(graph_config.get("edge_types", [])),
            layers=int(graph_config.get("graph_layers", 2)),
        )
        self.losses = LlmTraceLosses(temperature=float(config.get("training", {}).get("temperature", 0.1)))

    def parameters(self) -> list[Any]:
        params = []
        params.extend(self.fusion.process_expert.parameters())
        params.extend(self.fusion.geometry_expert.parameters())
        params.extend(self.fusion.symbol_expert.parameters())
        params.extend(self.fusion.pmca.parameters())
        params.extend(self.heads.parameters())
        params.extend(self.graph_encoder.parameters())
        return params

    def encode_operation(
        self,
        *,
        document_id: str,
        op_id: str,
        evidence: list[Evidence],
        chief_output: ChiefEngineerOutput,
    ):
        return self.fusion.fuse_operation(
            document_id=document_id,
            op_id=op_id,
            evidence=evidence,
            chief_output=chief_output,
        )

    def supervised_losses(
        self,
        *,
        entity_logits: Any,
        entity_labels: Any,
        relation_logits: Any,
        relation_labels: Any,
        contrastive_anchor: Any,
        contrastive_positive: Any,
        contrastive_candidates: Any,
        graph_view_a: Any,
        graph_view_b: Any,
    ) -> dict[str, Any]:
        return {
            "entity_loss": self.losses.entity_loss(entity_logits, entity_labels),
            "relation_loss": self.losses.relation_loss(relation_logits, relation_labels),
            "contrastive_loss": self.losses.info_nce(
                contrastive_anchor,
                contrastive_positive,
                contrastive_candidates,
            ),
            "graph_consistency_loss": self.losses.graph_consistency(graph_view_a, graph_view_b),
        }

    def total_loss(self, losses: dict[str, Any]) -> Any:
        training_config = self.config.get("training", {})
        weights = {
            "entity_loss": float(training_config.get("entity_loss_weight", 1.0)),
            "relation_loss": float(training_config.get("relation_loss_weight", 1.0)),
            "contrastive_loss": float(training_config.get("contrastive_loss_weight", 0.001)),
            "graph_consistency_loss": float(training_config.get("graph_consistency_loss_weight", 0.001)),
        }
        return self.losses.total(losses, weights)
