"""Training loop for the full LLM-TRACE objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_trace.models.llm_trace_model import LlmTraceModel


@dataclass(slots=True)
class TrainingBatch:
    document_ids: list[str]
    op_ids: list[str]
    evidence_ids: list[list[str]]
    entity_logits: Any
    entity_labels: Any
    relation_logits: Any
    relation_labels: Any
    contrastive_anchor: Any
    contrastive_positive: Any
    contrastive_candidates: Any
    graph_view_a: Any
    graph_view_b: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingStepOutput:
    loss: float
    losses: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)


class LlmTraceTrainer:
    """AdamW trainer for BERT + ViT + PMCA + MNER/MRE + TM-CGB."""

    def __init__(self, model: LlmTraceModel, config: dict[str, Any]) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.config = config
        self.optimizer = self.build_optimizer()

    def build_optimizer(self) -> Any:
        lr = float(self.config.get("learning_rate", 2e-5))
        return self.torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train_step(self, batch: TrainingBatch) -> TrainingStepOutput:
        self.optimizer.zero_grad()
        losses = self.model.supervised_losses(
            entity_logits=batch.entity_logits,
            entity_labels=batch.entity_labels,
            relation_logits=batch.relation_logits,
            relation_labels=batch.relation_labels,
            contrastive_anchor=batch.contrastive_anchor,
            contrastive_positive=batch.contrastive_positive,
            contrastive_candidates=batch.contrastive_candidates,
            graph_view_a=batch.graph_view_a,
            graph_view_b=batch.graph_view_b,
        )
        total_loss = self.model.total_loss(losses)
        total_loss.backward()
        self.optimizer.step()
        detached = {name: float(value.detach().cpu()) for name, value in losses.items()}
        return TrainingStepOutput(loss=float(total_loss.detach().cpu()), losses=detached)

    def fit(self, batches: list[TrainingBatch], max_epochs: int | None = None) -> list[TrainingStepOutput]:
        epochs = int(max_epochs or self.config.get("max_epochs", 30))
        history: list[TrainingStepOutput] = []
        for _ in range(epochs):
            for batch in batches:
                history.append(self.train_step(batch))
        return history
