"""Prior-Modulated Corroborative Attention style fusion."""

from __future__ import annotations

from collections import defaultdict

from llm_trace.schemas import ChiefEngineerOutput, EncodedEvidence, FusedOperation, ModalityPrior


class PriorModulatedCorroborativeAttention:
    """Deterministic PMCA-inspired weighted fusion.

    A neural implementation would compute Query, Key, and Value matrices. This
    scaffold implements the same control idea: modality priors and routing
    guidance modulate how much each evidence vector contributes.
    """

    def __init__(self, routing_bias: float = 0.15) -> None:
        self.routing_bias = routing_bias

    def fuse(
        self,
        *,
        document_id: str,
        op_id: str,
        encoded: list[EncodedEvidence],
        chief_output: ChiefEngineerOutput,
    ) -> FusedOperation:
        if not encoded:
            prior = chief_output.modality_prior
            return FusedOperation(
                document_id=document_id,
                op_id=op_id,
                vector=[],
                evidence_ids=[],
                modality_prior=prior,
                route_scores={},
                summary="No evidence available for fusion.",
            )

        route_scores = self._route_scores(chief_output)
        dim = len(encoded[0].vector)
        fused = [0.0] * dim
        used_ids: list[str] = []
        total_weight = 0.0
        for item in encoded:
            weight = self._modality_weight(item.modality, chief_output.modality_prior)
            weight *= max(0.0, min(1.0, item.confidence))
            weight += route_scores.get(item.evidence_id, 0.0)
            if weight <= 0:
                continue
            used_ids.append(item.evidence_id)
            total_weight += weight
            for index, value in enumerate(item.vector):
                fused[index] += weight * value

        if total_weight:
            fused = [value / total_weight for value in fused]

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
            vector=fused,
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

    @staticmethod
    def _modality_weight(modality: str, prior: ModalityPrior) -> float:
        if modality == "table":
            return prior.pi_table
        if modality == "drawing":
            return prior.pi_drawing
        if modality == "symbol":
            return prior.pi_symbol
        return 0.0
