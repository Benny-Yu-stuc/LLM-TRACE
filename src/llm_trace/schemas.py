"""Shared data schemas for the LLM-TRACE pipeline.

The manuscript emphasizes traceability. For that reason, every entity,
relation, route, and graph edge carries evidence IDs or source locations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Modality = Literal["table", "drawing", "symbol", "manual"]


@dataclass(slots=True)
class SourceLocation:
    """Original location of an evidence item."""

    document_id: str
    op_id: str
    source_type: Modality
    source_file: str = ""
    sheet_name: str = ""
    cell: str = ""
    bbox: list[float] = field(default_factory=list)
    page_index: int | None = None


@dataclass(slots=True)
class Evidence:
    """Atomic multimodal evidence item."""

    evidence_id: str
    document_id: str
    op_id: str
    modality: Modality
    text: str
    location: SourceLocation
    field_name: str = ""
    value: str = ""
    unit: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessDocument:
    """All parsed evidence belonging to one workbook."""

    document_id: str
    part_id: str = ""
    part_name: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def op_ids(self) -> list[str]:
        return sorted({item.op_id for item in self.evidence}, key=operation_sort_key)


@dataclass(slots=True)
class ModalityPrior:
    """Prior weights for table, drawing, and symbol evidence."""

    pi_table: float
    pi_drawing: float
    pi_symbol: float

    def normalized(self, minimum: float = 0.0) -> "ModalityPrior":
        values = {
            "pi_table": max(self.pi_table, minimum),
            "pi_drawing": max(self.pi_drawing, minimum),
            "pi_symbol": max(self.pi_symbol, minimum),
        }
        total = sum(values.values()) or 1.0
        return ModalityPrior(
            pi_table=values["pi_table"] / total,
            pi_drawing=values["pi_drawing"] / total,
            pi_symbol=values["pi_symbol"] / total,
        )


@dataclass(slots=True)
class RoutingFocus:
    """One routing instruction for a modality expert."""

    evidence_id: str
    reason: str
    priority: float = 1.0
    field_name: str = ""
    region_id: str = ""
    symbol_id: str = ""


@dataclass(slots=True)
class CandidateConstraint:
    """Rule, unit, type, sequence, or evidence constraint."""

    constraint_type: str
    description: str
    related_evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UncertainItem:
    """Candidate that is intentionally not promoted to final output."""

    item: str
    reason: str
    required_evidence: str = ""


@dataclass(slots=True)
class ChiefEngineerOutput:
    """Schema-constrained Chief-Engineer result."""

    document_id: str
    op_id: str
    modality_prior: ModalityPrior
    table_focus: list[RoutingFocus] = field(default_factory=list)
    drawing_focus: list[RoutingFocus] = field(default_factory=list)
    symbol_focus: list[RoutingFocus] = field(default_factory=list)
    candidate_constraints: list[CandidateConstraint] = field(default_factory=list)
    uncertain_items: list[UncertainItem] = field(default_factory=list)


@dataclass(slots=True)
class EncodedEvidence:
    """Vector-like representation produced by a modality expert."""

    evidence_id: str
    modality: Modality
    op_id: str
    vector: list[float]
    confidence: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusedOperation:
    """Prior-guided fused representation for one operation."""

    document_id: str
    op_id: str
    vector: list[float]
    token_vectors: list[list[float]]
    evidence_ids: list[str]
    modality_prior: ModalityPrior
    route_scores: dict[str, float]
    summary: str


@dataclass(slots=True)
class ExtractedEntity:
    """MNER-style entity with strict evidence anchor."""

    entity_id: str
    document_id: str
    op_id: str
    entity_type: str
    text: str
    evidence_id: str
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedRelation:
    """MRE-style relation with head, tail, type, and evidence anchor."""

    relation_id: str
    document_id: str
    op_id: str
    relation_type: str
    head_entity_id: str
    tail_entity_id: str
    evidence_ids: list[str]
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    op_id: str = ""
    evidence_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    evidence_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OperationChainGraph:
    document_id: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    consistency_score: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    document_id: str
    evidence: list[Evidence]
    chief_engineer_outputs: list[ChiefEngineerOutput]
    fused_operations: list[FusedOperation]
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    graph: OperationChainGraph
    metrics: dict[str, Any]


def operation_sort_key(op_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in op_id if ch.isdigit())
    return (int(digits) if digits else 999999, op_id)


def to_plain_data(value: Any) -> Any:
    """Convert nested dataclasses into JSON-serializable dictionaries."""

    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    return value
