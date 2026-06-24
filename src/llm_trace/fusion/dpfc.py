"""Domain Prior-guided Fusion and Corroboration module."""

from __future__ import annotations

from llm_trace.fusion.pmca import PriorModulatedCorroborativeAttention
from llm_trace.models.neural_experts import BertProcessExpert, StructuredSymbolExpert, VitGeometryExpert
from llm_trace.schemas import ChiefEngineerOutput, Evidence, FusedOperation


class DomainPriorFusionCorroboration:
    def __init__(self, config: dict) -> None:
        vector_dim = int(config.get("vector_dim", 768))
        hidden_dim = int(config.get("hidden_dim", vector_dim))
        attention_heads = int(config.get("attention_heads", 8))
        dropout = float(config.get("dropout", 0.1))
        strict = bool(config.get("strict_model_loading", False))
        self.process_expert = BertProcessExpert(config.get("process_expert", {}), vector_dim, strict=strict)
        self.geometry_expert = VitGeometryExpert(config.get("geometry_expert", {}), vector_dim, strict=strict)
        self.symbol_expert = StructuredSymbolExpert(config.get("symbol_expert", {}), vector_dim, strict=strict)
        self.pmca = PriorModulatedCorroborativeAttention(
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            dropout=dropout,
            routing_bias=float(config.get("routing_bias", 0.15)),
        )

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
