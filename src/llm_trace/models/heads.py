"""MNER and MRE neural head interfaces."""

from __future__ import annotations

from typing import Any


class NeuralExtractionHeads:
    """Token classifier and relation classifier described in the article."""

    def __init__(self, hidden_dim: int, entity_labels: list[str], relation_labels: list[str]) -> None:
        import torch

        self.torch = torch
        nn = torch.nn
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.entity_classifier = nn.Linear(hidden_dim, len(entity_labels))
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(relation_labels)),
        )

    def entity_logits(self, fused_sequence: Any) -> Any:
        return self.entity_classifier(fused_sequence)

    def relation_logits(self, head_repr: Any, tail_repr: Any, context_repr: Any) -> Any:
        return self.relation_classifier(self.torch.cat([head_repr, tail_repr, context_repr], dim=-1))

    def parameters(self) -> list[Any]:
        return list(self.entity_classifier.parameters()) + list(self.relation_classifier.parameters())
