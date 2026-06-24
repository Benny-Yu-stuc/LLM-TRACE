"""Top-level document parsing orchestration."""

from __future__ import annotations

from collections import defaultdict

from llm_trace.parsers.image_parser import ImageEvidenceParser
from llm_trace.parsers.manual_parser import ManualParser
from llm_trace.parsers.symbol_parser import SymbolParser
from llm_trace.parsers.workbook_parser import WorkbookParser
from llm_trace.schemas import Evidence, ProcessDocument


class DocumentParser:
    def __init__(self) -> None:
        self.workbook_parser = WorkbookParser()
        self.image_parser = ImageEvidenceParser()
        self.symbol_parser = SymbolParser()
        self.manual_parser = ManualParser()

    def parse(self, data_config: dict[str, str]) -> list[ProcessDocument]:
        evidence: list[Evidence] = []
        evidence.extend(self.workbook_parser.parse_dir(data_config.get("raw_workbook_dir", "")))
        evidence.extend(self.image_parser.parse_dir(data_config.get("raw_image_dir", "")))
        evidence.extend(self.symbol_parser.parse_dir(data_config.get("raw_symbol_dir", "")))

        manuals = self.manual_parser.parse_dir(data_config.get("raw_manual_dir", ""))
        evidence.extend(manuals)

        if not evidence:
            return []

        grouped: dict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            if item.modality == "manual" and item.op_id == "GLOBAL":
                continue
            grouped[item.document_id].append(item)

        if not grouped and manuals:
            grouped["manual_only"].extend(manuals)

        documents: list[ProcessDocument] = []
        for document_id, items in grouped.items():
            related_manuals = [item for item in manuals if item.document_id == document_id or item.op_id == "GLOBAL"]
            documents.append(ProcessDocument(document_id=document_id, evidence=items + related_manuals))
        return documents
