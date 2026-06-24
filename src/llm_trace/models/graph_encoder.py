"""Relation-aware operation-chain graph encoder interface."""

from __future__ import annotations

from typing import Any

from llm_trace.schemas import OperationChainGraph


class RelationAwareGraphEncoder:
    """R-GCN style graph encoder for OP sequence and evidence graph edges."""

    def __init__(self, hidden_dim: int, edge_types: list[str], layers: int = 2) -> None:
        import torch

        self.torch = torch
        nn = torch.nn
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types
        self.layers = layers
        self.edge_transforms = {
            edge_type: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for edge_type in edge_types
        }
        self.self_loop = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def parameters(self) -> list[Any]:
        params = []
        for layer in self.edge_transforms.values():
            params.extend(list(layer.parameters()))
        params.extend(list(self.self_loop.parameters()))
        return params

    def encode(self, node_features: Any, edge_index: Any, edge_type_ids: Any, edge_weights: Any | None = None) -> Any:
        hidden = node_features
        for _ in range(self.layers):
            hidden = self._message_passing(hidden, edge_index, edge_type_ids, edge_weights)
        return hidden

    def _message_passing(self, hidden: Any, edge_index: Any, edge_type_ids: Any, edge_weights: Any | None) -> Any:
        output = self.self_loop(hidden)
        for edge_type_id, edge_type in enumerate(self.edge_types):
            mask = edge_type_ids == edge_type_id
            if not mask.any():
                continue
            source = edge_index[0][mask]
            target = edge_index[1][mask]
            messages = self.edge_transforms[edge_type](hidden[source])
            if edge_weights is not None:
                messages = messages * edge_weights[mask].unsqueeze(-1)
            output.index_add_(0, target, messages)
        return self.activation(output)

    def featurize_graph(self, graph: OperationChainGraph) -> dict[str, Any]:
        edge_type_to_id = {edge_type: index for index, edge_type in enumerate(self.edge_types)}
        node_to_id = {node.node_id: index for index, node in enumerate(graph.nodes)}
        edges = []
        type_ids = []
        weights = []
        for edge in graph.edges:
            source = node_to_id.get(edge.source)
            target = node_to_id.get(edge.target)
            if source is None or target is None:
                continue
            normalized_type = edge.edge_type.split(":", 1)[0]
            edges.append((source, target))
            type_ids.append(edge_type_to_id.get(normalized_type, 0))
            weights.append(edge.weight)
        return {"node_to_id": node_to_id, "edges": edges, "edge_type_ids": type_ids, "edge_weights": weights}
