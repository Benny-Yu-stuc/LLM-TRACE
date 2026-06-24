"""Built-in empty-safe demo data."""

from __future__ import annotations

from llm_trace.parsers.evidence_builder import build_evidence
from llm_trace.schemas import ProcessDocument


def build_demo_document() -> ProcessDocument:
    document_id = "demo_stamping_process_card"
    evidence = [
        build_evidence(
            document_id=document_id,
            op_id="OP10",
            modality="table",
            text="Operation OP10 drawing process, equipment Press-1200T, pressure 450 kN.",
            source_file="demo/table.csv",
            field_name="operation_equipment_parameter",
            value="Press-1200T pressure 450 kN",
            unit="kN",
            sheet_name="OP10",
            cell="B4",
        ),
        build_evidence(
            document_id=document_id,
            op_id="OP10",
            modality="drawing",
            text="Drawing region marks forming area and draw bead location.",
            source_file="demo/op10.png",
            field_name="forming_region",
            bbox=[10, 20, 240, 180],
            confidence=0.86,
        ),
        build_evidence(
            document_id=document_id,
            op_id="OP10",
            modality="symbol",
            text="SC quality control symbol near draw bead.",
            source_file="demo/op10_symbols.json",
            field_name="SC",
            bbox=[80, 120, 112, 145],
            confidence=0.91,
        ),
        build_evidence(
            document_id=document_id,
            op_id="OP20",
            modality="table",
            text="Operation OP20 trimming process, equipment Trim-800T, clearance 0.8 mm.",
            source_file="demo/table.csv",
            field_name="operation_equipment_parameter",
            value="Trim-800T clearance 0.8 mm",
            unit="mm",
            sheet_name="OP20",
            cell="B4",
        ),
        build_evidence(
            document_id=document_id,
            op_id="OP20",
            modality="drawing",
            text="Drawing region highlights trimming line and piercing holes.",
            source_file="demo/op20.png",
            field_name="trim_region",
            bbox=[12, 18, 260, 190],
            confidence=0.83,
        ),
        build_evidence(
            document_id=document_id,
            op_id="OP20",
            modality="manual",
            text="Clearance values must use mm units and remain consistent with trimming quality standards.",
            source_file="demo/manual.txt",
            field_name="quality_standard",
            value="Clearance values must use mm units.",
        ),
    ]
    return ProcessDocument(document_id=document_id, part_id="DEMO_PART", part_name="Demo stamping part", evidence=evidence)
