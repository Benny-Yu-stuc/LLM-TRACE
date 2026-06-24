"""Symbol detection parser."""

from __future__ import annotations

from pathlib import Path

from llm_trace.parsers.evidence_builder import build_evidence
from llm_trace.schemas import Evidence
from llm_trace.utils.io import read_json


class SymbolParser:
    def parse_dir(self, directory: str | Path) -> list[Evidence]:
        path = Path(directory)
        if not path.exists():
            return []
        evidence: list[Evidence] = []
        for file_path in sorted(path.glob("*.json")):
            evidence.extend(self.parse_file(file_path))
        return evidence

    def parse_file(self, path: str | Path) -> list[Evidence]:
        file_path = Path(path)
        payload = read_json(file_path)
        symbols = payload.get("symbols", payload) if isinstance(payload, dict) else payload
        evidence: list[Evidence] = []
        for index, symbol in enumerate(symbols if isinstance(symbols, list) else [], start=1):
            if not isinstance(symbol, dict):
                continue
            category = str(symbol.get("category") or symbol.get("symbol_type") or "symbol")
            text = str(symbol.get("text") or symbol.get("label") or category)
            document_id = str(symbol.get("document_id") or file_path.stem)
            op_id = str(symbol.get("op_id") or "OP_UNKNOWN")
            evidence.append(
                build_evidence(
                    document_id=document_id,
                    op_id=op_id,
                    modality="symbol",
                    text=text,
                    source_file=str(symbol.get("image_file") or file_path),
                    bbox=[float(x) for x in symbol.get("bbox", [])],
                    field_name=category,
                    value=text,
                    confidence=float(symbol.get("confidence", 1.0)),
                    metadata={"symbol_index": index, **symbol},
                )
            )
        return evidence
