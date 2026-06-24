"""End-to-end LLM-TRACE pipeline."""

from __future__ import annotations

from collections import defaultdict

from llm_trace.chief_engineer.router import ChiefEngineerRouter
from llm_trace.eval.metrics import compute_metrics
from llm_trace.extraction.extractor import MultiModalExtractor
from llm_trace.fusion.dpfc import DomainPriorFusionCorroboration
from llm_trace.graph.tm_cgb import TemporalMultimodalContrastiveGraphBooster
from llm_trace.parsers.document_parser import DocumentParser
from llm_trace.schemas import CandidateConstraint, Evidence, PipelineResult, ProcessDocument


class LlmTracePipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.parser = DocumentParser()
        chief_config = dict(config.get("chief_engineer", {}))
        chief_config["minimum_prior"] = config.get("fusion", {}).get("minimum_prior", 0.05)
        self.router = ChiefEngineerRouter(chief_config)
        self.fusion = DomainPriorFusionCorroboration(config.get("fusion", {}))
        self.extractor = MultiModalExtractor(config.get("extraction", {}))
        self.graph_booster = TemporalMultimodalContrastiveGraphBooster(config.get("graph", {}))

    def load_documents(self) -> list[ProcessDocument]:
        documents = self.parser.parse(self.config.get("data", {}))
        if documents:
            return documents
        raise FileNotFoundError("No process documents parsed from rawdata.")

    def run_document(self, document: ProcessDocument) -> PipelineResult:
        manual_evidence = [item for item in document.evidence if item.modality == "manual"]
        operation_evidence = [item for item in document.evidence if item.modality != "manual"]
        by_op: dict[str, list[Evidence]] = defaultdict(list)
        for item in operation_evidence:
            by_op[item.op_id].append(item)

        chief_outputs = []
        fused_operations = []
        constraints: list[CandidateConstraint] = []
        for op_id in document.op_ids:
            op_items = by_op.get(op_id, [])
            if not op_items:
                continue
            chief = self.router.route(
                document_id=document.document_id,
                op_id=op_id,
                operation_evidence=op_items,
                manual_evidence=manual_evidence,
            )
            chief_outputs.append(chief)
            constraints.extend(chief.candidate_constraints)
            fused_operations.append(
                self.fusion.fuse_operation(
                    document_id=document.document_id,
                    op_id=op_id,
                    evidence=op_items,
                    chief_output=chief,
                )
            )

        entities, relations = self.extractor.extract(document.document_id, document.evidence, fused_operations)
        graph = self.graph_booster.build(
            document_id=document.document_id,
            evidence=document.evidence,
            entities=entities,
            relations=relations,
            constraints=constraints,
        )
        alignment_scores = self.graph_booster.contrastive_alignment_score(document.evidence)
        metrics = compute_metrics(
            evidence=document.evidence,
            entities=entities,
            relations=relations,
            graph=graph,
            alignment_scores=alignment_scores,
        )
        return PipelineResult(
            document_id=document.document_id,
            evidence=document.evidence,
            chief_engineer_outputs=chief_outputs,
            fused_operations=fused_operations,
            entities=entities,
            relations=relations,
            graph=graph,
            metrics=metrics,
        )
