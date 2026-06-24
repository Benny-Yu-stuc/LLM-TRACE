"""Domain Prior-guided Fusion and Corroboration module."""

from __future__ import annotations

from llm_trace.fusion.experts import GeometryExpert, ProcessExpert, SymbolExpert
from llm_trace.fusion.pmca import PriorModulatedCorroborativeAttention
from llm_trace.schemas import ChiefEngineerOutput, Evidence, FusedOperation


class DomainPriorFusionCorroboration:
    def __init__(self, config: dict) -> None:
        vector_dim = int(config.get("vector_dim", 64))
        self.process_expert = ProcessExpert(vector_dim)
        self.geometry_expert = GeometryExpert(vector_dim)
        self.symbol_expert = SymbolExpert(vector_dim)
        self.pmca = PriorModulatedCorroborativeAttention(float(config.get("routing_bias", 0.15)))

    def fuse_operation(
        self,
        *,
        document_id: str,
        op_id: str,
        evidence: list[Evidence],
        chief_output: ChiefEngineerOutput,
    ) -> FusedOperation:
        encoded = []
        encoded.extend(self.process_expert.encode(evidence))
        encoded.extend(self.geometry_expert.encode(evidence))
        encoded.extend(self.symbol_expert.encode(evidence))
        return self.pmca.fuse(
            document_id=document_id,
            op_id=op_id,
            encoded=encoded,
            chief_output=chief_output,
        )
