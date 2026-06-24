"""Chief-Engineer routing implementation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_trace.chief_engineer.client import ChiefEngineerApiClient
from llm_trace.chief_engineer.prompt import build_chief_engineer_prompt
from llm_trace.schemas import (
    CandidateConstraint,
    ChiefEngineerOutput,
    Evidence,
    ModalityPrior,
    RoutingFocus,
    UncertainItem,
)
from llm_trace.utils.io import read_json, write_json


class ChiefEngineerRouter:
    """Generate modality priors, evidence routes, and constraints."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.minimum_prior = float(config.get("minimum_prior", 0.05))
        self.cache_enabled = bool(config.get("cache_enabled", True))
        self.cache_dir = Path(config.get("cache_dir", ""))
        self.client = ChiefEngineerApiClient(config)
        if self.cache_enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def route(
        self,
        *,
        document_id: str,
        op_id: str,
        operation_evidence: list[Evidence],
        manual_evidence: list[Evidence],
    ) -> ChiefEngineerOutput:
        prompt = build_chief_engineer_prompt(
            document_id=document_id,
            op_id=op_id,
            operation_evidence=operation_evidence,
            manual_evidence=manual_evidence,
        )
        cache_path = self._cache_path(document_id, op_id, prompt)
        if cache_path and cache_path.exists():
            return self._from_payload(read_json(cache_path), document_id=document_id, op_id=op_id)

        if self.client.enabled:
            try:
                payload = self.client.complete_json(prompt)
                result = self._from_payload(payload, document_id=document_id, op_id=op_id)
            except Exception as exc:
                result = self._local_route(document_id, op_id, operation_evidence, manual_evidence)
                result.uncertain_items.append(
                    UncertainItem(
                        item="chief_engineer_api",
                        reason=f"External routing failed, local routing used: {exc}",
                        required_evidence="api_response",
                    )
                )
        else:
            result = self._local_route(document_id, op_id, operation_evidence, manual_evidence)

        if cache_path:
            write_json(cache_path, _chief_to_payload(result))
        return result

    def _local_route(
        self,
        document_id: str,
        op_id: str,
        operation_evidence: list[Evidence],
        manual_evidence: list[Evidence],
    ) -> ChiefEngineerOutput:
        table = [item for item in operation_evidence if item.modality == "table"]
        drawing = [item for item in operation_evidence if item.modality == "drawing"]
        symbol = [item for item in operation_evidence if item.modality == "symbol"]

        table_score = _quality_score(table, important_words=["op", "equipment", "parameter", "quality", "material", "part"])
        drawing_score = _quality_score(drawing, important_words=["region", "drawing", "dimension", "surface", "hole", "line"])
        symbol_score = _quality_score(symbol, important_words=["sc", "cc", "arrow", "dimension", "tolerance", "qc"])
        prior = ModalityPrior(table_score, drawing_score, symbol_score).normalized(minimum=self.minimum_prior)

        constraints: list[CandidateConstraint] = []
        for item in manual_evidence[:8]:
            constraints.append(
                CandidateConstraint(
                    constraint_type=_guess_constraint_type(item.text),
                    description=item.text,
                    related_evidence=[item.evidence_id],
                )
            )

        uncertain: list[UncertainItem] = []
        if not table:
            uncertain.append(UncertainItem("table_evidence", "No table evidence found for this OP.", "table cells"))
        if not drawing:
            uncertain.append(UncertainItem("drawing_evidence", "No drawing region evidence found for this OP.", "image regions"))
        if not symbol:
            uncertain.append(UncertainItem("symbol_evidence", "No symbol evidence found for this OP.", "symbol boxes"))

        return ChiefEngineerOutput(
            document_id=document_id,
            op_id=op_id,
            modality_prior=prior,
            table_focus=_top_focus(table, "table"),
            drawing_focus=_top_focus(drawing, "drawing"),
            symbol_focus=_top_focus(symbol, "symbol"),
            candidate_constraints=constraints,
            uncertain_items=uncertain,
        )

    def _from_payload(self, payload: dict[str, Any], *, document_id: str, op_id: str) -> ChiefEngineerOutput:
        prior_payload = payload.get("modality_prior", {})
        prior = ModalityPrior(
            pi_table=float(prior_payload.get("pi_table", 0.34)),
            pi_drawing=float(prior_payload.get("pi_drawing", 0.33)),
            pi_symbol=float(prior_payload.get("pi_symbol", 0.33)),
        ).normalized(minimum=self.minimum_prior)

        guidance = payload.get("routing_guidance", {})
        return ChiefEngineerOutput(
            document_id=str(payload.get("document_id") or document_id),
            op_id=str(payload.get("op_id") or op_id),
            modality_prior=prior,
            table_focus=[_focus_from_dict(item) for item in guidance.get("table_focus", [])],
            drawing_focus=[_focus_from_dict(item) for item in guidance.get("drawing_focus", [])],
            symbol_focus=[_focus_from_dict(item) for item in guidance.get("symbol_focus", [])],
            candidate_constraints=[
                CandidateConstraint(
                    constraint_type=str(item.get("constraint_type", "evidence")),
                    description=str(item.get("description", "")),
                    related_evidence=[str(value) for value in item.get("related_evidence", [])],
                )
                for item in payload.get("candidate_constraints", [])
                if isinstance(item, dict)
            ],
            uncertain_items=[
                UncertainItem(
                    item=str(item.get("item", "")),
                    reason=str(item.get("reason", "")),
                    required_evidence=str(item.get("required_evidence", "")),
                )
                for item in payload.get("uncertain_items", [])
                if isinstance(item, dict)
            ],
        )

    def _cache_path(self, document_id: str, op_id: str, prompt: str) -> Path | None:
        if not self.cache_enabled or not self.cache_dir:
            return None
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{document_id}_{op_id}_{digest}.json"


def _quality_score(items: list[Evidence], important_words: list[str]) -> float:
    if not items:
        return 0.0
    confidence = sum(max(0.0, min(1.0, item.confidence)) for item in items) / len(items)
    text = " ".join(item.text.lower() for item in items)
    keyword_hits = sum(1 for word in important_words if word in text)
    coverage = min(1.0, len(items) / 8.0)
    return 0.55 * confidence + 0.25 * coverage + 0.20 * (keyword_hits / max(1, len(important_words)))


def _top_focus(items: list[Evidence], modality: str) -> list[RoutingFocus]:
    sorted_items = sorted(items, key=lambda item: (item.confidence, len(item.text)), reverse=True)
    focus: list[RoutingFocus] = []
    for rank, item in enumerate(sorted_items[:10], start=1):
        focus.append(
            RoutingFocus(
                evidence_id=item.evidence_id,
                reason=f"High-confidence {modality} evidence selected by local routing.",
                priority=max(0.1, 1.0 - (rank - 1) * 0.08),
                field_name=item.field_name if modality == "table" else "",
                region_id=item.metadata.get("region_id", item.evidence_id) if modality == "drawing" else "",
                symbol_id=item.metadata.get("symbol_id", item.evidence_id) if modality == "symbol" else "",
            )
        )
    return focus


def _guess_constraint_type(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["before", "after", "op", "sequence"]):
        return "sequence"
    if any(token in lower for token in ["mm", "mpa", "kn", "unit", "range"]):
        return "unit"
    if any(token in lower for token in ["must", "shall", "quality", "standard"]):
        return "type"
    return "evidence"


def _focus_from_dict(item: dict[str, Any]) -> RoutingFocus:
    return RoutingFocus(
        evidence_id=str(item.get("evidence_id") or item.get("cell_id") or item.get("region_id") or item.get("symbol_id") or ""),
        reason=str(item.get("reason", "")),
        priority=float(item.get("priority", 1.0)),
        field_name=str(item.get("field_name", "")),
        region_id=str(item.get("region_id", "")),
        symbol_id=str(item.get("symbol_id", "")),
    )


def _chief_to_payload(output: ChiefEngineerOutput) -> dict[str, Any]:
    return {
        "document_id": output.document_id,
        "op_id": output.op_id,
        "modality_prior": {
            "pi_table": output.modality_prior.pi_table,
            "pi_drawing": output.modality_prior.pi_drawing,
            "pi_symbol": output.modality_prior.pi_symbol,
        },
        "routing_guidance": {
            "table_focus": [asdict(focus) for focus in output.table_focus],
            "drawing_focus": [asdict(focus) for focus in output.drawing_focus],
            "symbol_focus": [asdict(focus) for focus in output.symbol_focus],
        },
        "candidate_constraints": [asdict(item) for item in output.candidate_constraints],
        "uncertain_items": [asdict(item) for item in output.uncertain_items],
    }
