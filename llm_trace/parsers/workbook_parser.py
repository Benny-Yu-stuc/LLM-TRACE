"""Workbook parser for process cards.

The production system can replace this file with a richer Excel parser.
This scaffold supports:

- .xlsx when openpyxl is installed
- .csv with columns such as op_id, field_name, value, unit
- .json arrays of evidence-like rows

All parsed cells become table evidence with sheet and cell anchors.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm_trace.parsers.evidence_builder import build_evidence
from llm_trace.schemas import Evidence


class WorkbookParser:
    def parse_dir(self, directory: str | Path) -> list[Evidence]:
        path = Path(directory)
        if not path.exists():
            return []
        evidence: list[Evidence] = []
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in {".xlsx", ".csv", ".json", ".txt"}:
                evidence.extend(self.parse_file(file_path))
        return evidence

    def parse_file(self, path: str | Path) -> list[Evidence]:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".xlsx":
            return self._parse_xlsx(file_path)
        if suffix == ".csv":
            return self._parse_csv(file_path)
        if suffix == ".json":
            return self._parse_json(file_path)
        if suffix == ".txt":
            return self._parse_text_table(file_path)
        return []

    def _parse_xlsx(self, file_path: Path) -> list[Evidence]:
        try:
            import openpyxl  # type: ignore
        except Exception:
            return [
                build_evidence(
                    document_id=file_path.stem,
                    op_id="OP_UNKNOWN",
                    modality="table",
                    text=f"Workbook placeholder for {file_path.name}; install openpyxl to parse cells.",
                    source_file=str(file_path),
                    field_name="workbook_placeholder",
                    value=file_path.name,
                    confidence=0.2,
                )
            ]

        workbook = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        evidence: list[Evidence] = []
        for sheet in workbook.worksheets:
            op_id = infer_op_id(sheet.title)
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is None:
                        continue
                    text = str(cell.value).strip()
                    if not text:
                        continue
                    evidence.append(
                        build_evidence(
                            document_id=file_path.stem,
                            op_id=op_id,
                            modality="table",
                            text=text,
                            source_file=str(file_path),
                            sheet_name=sheet.title,
                            cell=cell.coordinate,
                            field_name=guess_field_name(sheet.title, cell.coordinate),
                            value=text,
                            metadata={"row": cell.row, "column": cell.column},
                        )
                    )
        workbook.close()
        return evidence

    def _parse_csv(self, file_path: Path) -> list[Evidence]:
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        evidence: list[Evidence] = []
        for index, row in enumerate(rows, start=2):
            document_id = row.get("document_id") or file_path.stem
            op_id = row.get("op_id") or infer_op_id(row.get("sheet_name") or "")
            field_name = row.get("field_name") or row.get("field") or ""
            value = row.get("value") or row.get("text") or ""
            unit = row.get("unit") or ""
            text = " ".join(part for part in [field_name, value, unit] if part)
            if not text:
                continue
            evidence.append(
                build_evidence(
                    document_id=document_id,
                    op_id=op_id,
                    modality="table",
                    text=text,
                    source_file=str(file_path),
                    sheet_name=row.get("sheet_name", ""),
                    cell=row.get("cell") or f"R{index}",
                    field_name=field_name,
                    value=value,
                    unit=unit,
                    metadata=dict(row),
                )
            )
        return evidence

    def _parse_json(self, file_path: Path) -> list[Evidence]:
        with file_path.open("r", encoding="utf-8") as file:
            payload: Any = json.load(file)
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        evidence: list[Evidence] = []
        for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
            if not isinstance(row, dict):
                continue
            document_id = str(row.get("document_id") or file_path.stem)
            op_id = str(row.get("op_id") or infer_op_id(str(row.get("sheet_name", ""))))
            text = str(row.get("text") or row.get("value") or "")
            field_name = str(row.get("field_name") or row.get("field") or "")
            value = str(row.get("value") or text)
            unit = str(row.get("unit") or "")
            combined = " ".join(part for part in [field_name, text, unit] if part)
            if not combined:
                continue
            evidence.append(
                build_evidence(
                    document_id=document_id,
                    op_id=op_id,
                    modality="table",
                    text=combined,
                    source_file=str(file_path),
                    sheet_name=str(row.get("sheet_name", "")),
                    cell=str(row.get("cell", f"row_{index}")),
                    field_name=field_name,
                    value=value,
                    unit=unit,
                    confidence=float(row.get("confidence", 1.0)),
                    metadata=row,
                )
            )
        return evidence

    def _parse_text_table(self, file_path: Path) -> list[Evidence]:
        evidence: list[Evidence] = []
        for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            evidence.append(
                build_evidence(
                    document_id=file_path.stem,
                    op_id=infer_op_id(text),
                    modality="table",
                    text=text,
                    source_file=str(file_path),
                    cell=f"L{index}",
                    field_name="line_text",
                    value=text,
                )
            )
        return evidence


def infer_op_id(text: str) -> str:
    upper = text.upper()
    for token in upper.replace("_", " ").replace("-", " ").split():
        if token.startswith("OP") and any(char.isdigit() for char in token):
            digits = "".join(char for char in token if char.isdigit())
            return f"OP{digits}"
    return "OP_UNKNOWN"


def guess_field_name(sheet_name: str, cell: str) -> str:
    if sheet_name:
        return f"{sheet_name}:{cell}"
    return cell
