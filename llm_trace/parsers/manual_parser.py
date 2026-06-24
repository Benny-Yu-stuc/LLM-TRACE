"""External process rule and manual parser."""

from __future__ import annotations

from pathlib import Path

from llm_trace.parsers.evidence_builder import build_evidence
from llm_trace.schemas import Evidence
from llm_trace.utils.io import read_json


class ManualParser:
    def parse_dir(self, directory: str | Path) -> list[Evidence]:
        path = Path(directory)
        if not path.exists():
            return []
        evidence: list[Evidence] = []
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in {".txt", ".md"}:
                evidence.extend(self._parse_text(file_path))
            elif file_path.suffix.lower() == ".json":
                evidence.extend(self._parse_json(file_path))
        return evidence

    def _parse_text(self, file_path: Path) -> list[Evidence]:
        evidence: list[Evidence] = []
        for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            evidence.append(
                build_evidence(
                    document_id=file_path.stem,
                    op_id="GLOBAL",
                    modality="manual",
                    text=text,
                    source_file=str(file_path),
                    cell=f"clause_{index}",
                    field_name="manual_clause",
                    value=text,
                    metadata={"clause_index": index},
                )
            )
        return evidence

    def _parse_json(self, file_path: Path) -> list[Evidence]:
        payload = read_json(file_path)
        clauses = payload.get("clauses", payload) if isinstance(payload, dict) else payload
        evidence: list[Evidence] = []
        for index, clause in enumerate(clauses if isinstance(clauses, list) else [], start=1):
            if isinstance(clause, str):
                clause = {"text": clause}
            if not isinstance(clause, dict):
                continue
            text = str(clause.get("text") or clause.get("clause") or "")
            if not text:
                continue
            evidence.append(
                build_evidence(
                    document_id=str(clause.get("document_id") or file_path.stem),
                    op_id=str(clause.get("op_id") or "GLOBAL"),
                    modality="manual",
                    text=text,
                    source_file=str(file_path),
                    cell=str(clause.get("clause_id") or f"clause_{index}"),
                    field_name=str(clause.get("rule_type") or "manual_clause"),
                    value=text,
                    confidence=float(clause.get("confidence", 1.0)),
                    metadata=clause,
                )
            )
        return evidence
