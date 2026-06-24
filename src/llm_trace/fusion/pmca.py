"""Prior-Modulated Corroborative Attention implementation."""

from __future__ import annotations

from collections import defaultdict

from llm_trace.schemas import ChiefEngineerOutput, EncodedEvidence, FusedOperation, ModalityPrior


class PriorModulatedCorroborativeAttention:
    """PMCA with modality-specific Query and prior-modulated Key/Value."""

    def __init__(self, hidden_dim: int, attention_heads: int, dropout: float = 0.1, routing_bias: float = 0.15) -> None:
        import torch

        self.torch = torch
        nn = torch.nn
        self.hidden_dim = hidden_dim
        self.routing_bias = routing_bias
        self.q_table = nn.Linear(hidden_dim, hidden_dim)
        self.q_drawing = nn.Linear(hidden_dim, hidden_dim)
        self.q_symbol = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def parameters(self):
        params = []
        for module in [
            self.q_table,
            self.q_drawing,
            self.q_symbol,
            self.k_proj,
            self.v_proj,
            self.attention,
            self.norm,
            self.ffn,
        ]:
            params.extend(list(module.parameters()))
        return params

    def fuse(
        self,
        *,
        document_id: str,
        op_id: str,
        encoded: list[EncodedEvidence],
        chief_output: ChiefEngineerOutput,
    ) -> FusedOperation:
        if not encoded:
            raise ValueError(f"No encoded evidence available for {document_id}/{op_id}.")

        route_scores = self._route_scores(chief_output)
        memory = self.torch.tensor([_fit_dim(item.vector, self.hidden_dim) for item in encoded], dtype=self.torch.float32).unsqueeze(0)
        prior_weights = self.torch.tensor(
            [self._modality_weight(item.modality, chief_output.modality_prior) for item in encoded],
            dtype=self.torch.float32,
        ).view(1, -1, 1)
        key = self.k_proj(memory) * prior_weights
        value = self.v_proj(memory) * prior_weights
        updated_tokens = []

        by_modality: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(encoded):
            by_modality[item.modality].append(index)

        for modality in ["table", "drawing", "symbol"]:
            indices = by_modality.get(modality, [])
            if not indices:
                continue
            query_input = memory[:, indices, :]
            query = self._query_projection(modality)(query_input)
            attn_mask = self._attention_bias(indices, encoded, route_scores)
            attended, _ = self.attention(query=query, key=key, value=value, attn_mask=attn_mask)
            updated = self.norm(query_input + attended)
            updated = self.norm(updated + self.ffn(updated))
            updated_tokens.append(updated)

        if not updated_tokens:
            raise ValueError(f"Encoded evidence for {document_id}/{op_id} does not contain table, drawing, or symbol tokens.")

        fused_sequence = self.torch.cat(updated_tokens, dim=1)
        fused_vector = fused_sequence.mean(dim=1).squeeze(0).detach().cpu().tolist()
        token_vectors = fused_sequence.squeeze(0).detach().cpu().tolist()
        used_ids = [item.evidence_id for item in encoded]

        route_scores = self._route_scores(chief_output)
        modality_counts: dict[str, int] = defaultdict(int)
        for item in encoded:
            modality_counts[item.modality] += 1
        summary = (
            f"Fused OP {op_id} with {len(used_ids)} evidence units; "
            f"table={modality_counts['table']}, drawing={modality_counts['drawing']}, "
            f"symbol={modality_counts['symbol']}."
        )
        return FusedOperation(
            document_id=document_id,
            op_id=op_id,
            vector=fused_vector,
            token_vectors=token_vectors,
            evidence_ids=used_ids,
            modality_prior=chief_output.modality_prior,
            route_scores=route_scores,
            summary=summary,
        )

    def _route_scores(self, chief_output: ChiefEngineerOutput) -> dict[str, float]:
        scores: dict[str, float] = {}
        for focus in chief_output.table_focus + chief_output.drawing_focus + chief_output.symbol_focus:
            if not focus.evidence_id:
                continue
            scores[focus.evidence_id] = max(scores.get(focus.evidence_id, 0.0), self.routing_bias * focus.priority)
        return scores

    def _attention_bias(self, query_indices: list[int], encoded: list[EncodedEvidence], route_scores: dict[str, float]):
        bias = self.torch.zeros((len(query_indices), len(encoded)), dtype=self.torch.float32)
        for row_index, _ in enumerate(query_indices):
            for col_index, item in enumerate(encoded):
                score = route_scores.get(item.evidence_id, 0.0)
                if score:
                    bias[row_index, col_index] = -score
        return bias

    def _query_projection(self, modality: str):
        if modality == "table":
            return self.q_table
        if modality == "drawing":
            return self.q_drawing
        if modality == "symbol":
            return self.q_symbol
        raise ValueError(f"Unsupported modality for PMCA query projection: {modality}")

    @staticmethod
    def _modality_weight(modality: str, prior: ModalityPrior) -> float:
        if modality == "table":
            return prior.pi_table
        if modality == "drawing":
            return prior.pi_drawing
        if modality == "symbol":
            return prior.pi_symbol
        raise ValueError(f"Unsupported modality for PMCA prior: {modality}")


def _fit_dim(vector: list[float], dim: int) -> list[float]:
    if len(vector) == dim:
        return vector
    if len(vector) > dim:
        return vector[:dim]
    return vector + [0.0] * (dim - len(vector))
