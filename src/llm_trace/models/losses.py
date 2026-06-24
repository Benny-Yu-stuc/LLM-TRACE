"""Training losses for the article-level objective."""

from __future__ import annotations

from typing import Any


class LlmTraceLosses:
    """L = L_MNER + L_MRE + lambda_c L_contrastive + lambda_g L_graph."""

    def __init__(self, temperature: float = 0.1) -> None:
        import torch

        self.torch = torch
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def entity_loss(self, logits: Any, labels: Any) -> Any:
        return self.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

    def relation_loss(self, logits: Any, labels: Any) -> Any:
        return self.cross_entropy(logits, labels)

    def info_nce(self, anchors: Any, positives: Any, candidates: Any) -> Any:
        anchors = self.torch.nn.functional.normalize(anchors, dim=-1)
        positives = self.torch.nn.functional.normalize(positives, dim=-1)
        candidates = self.torch.nn.functional.normalize(candidates, dim=-1)
        positive_logits = (anchors * positives).sum(dim=-1, keepdim=True) / self.temperature
        negative_logits = self.torch.matmul(anchors.unsqueeze(1), candidates.transpose(-1, -2)).squeeze(1)
        negative_logits = negative_logits / self.temperature
        logits = self.torch.cat([positive_logits, negative_logits], dim=-1)
        labels = self.torch.zeros(logits.shape[0], dtype=self.torch.long, device=logits.device)
        return self.cross_entropy(logits, labels)

    def graph_consistency(self, view_a: Any, view_b: Any) -> Any:
        view_a = self.torch.nn.functional.normalize(view_a, dim=-1)
        view_b = self.torch.nn.functional.normalize(view_b, dim=-1)
        return ((view_a - view_b) ** 2).sum(dim=-1).mean()

    def total(self, losses: dict[str, Any], weights: dict[str, float]) -> Any:
        total_loss = 0
        for name, value in losses.items():
            total_loss = total_loss + weights.get(name, 1.0) * value
        return total_loss
