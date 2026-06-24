import json
import unittest
from pathlib import Path

from llm_trace.annotation_schema import validate_annotation_sample
from llm_trace.parsers.document_parser import DocumentParser
from llm_trace.utils.config import load_json_config


class RawDataParserAlignmentTest(unittest.TestCase):
    def test_rawdata_matches_parser_and_annotation_interfaces(self):
        root = Path(__file__).resolve().parents[1]
        config = load_json_config(root / "configs" / "default_config.json")
        documents = DocumentParser().parse(config["data"])

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document.document_id, "stamping_process_sheet_OP40")

        by_field = {item.field_name: item for item in document.evidence}
        by_metadata_id = {}
        for item in document.evidence:
            for key in ["region_id", "symbol_id"]:
                if key in item.metadata:
                    by_metadata_id[item.metadata[key]] = item

        self.assertIn("process_parameter_calibration_force", by_field)
        self.assertIn("equipment_press", by_field)
        self.assertIn("img_op40_stamping_mark", by_metadata_id)
        self.assertIn("sym_op40_sc_remark", by_metadata_id)

        annotation_path = root / "rawdata" / "annotations" / "stamping_process_sheet_OP40.jsonl"
        sample = json.loads(annotation_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(validate_annotation_sample(sample, document.evidence), [])

        chief_path = root / "rawdata" / "annotations" / "chief_engineer_OP40.cached.json"
        chief = json.loads(chief_path.read_text(encoding="utf-8"))
        parsed_evidence_ids = {item.evidence_id for item in document.evidence}
        for focus_group in chief["routing_guidance"].values():
            for focus in focus_group:
                self.assertIn(focus["evidence_id"], parsed_evidence_ids)
        for constraint in chief["candidate_constraints"]:
            for evidence_id in constraint["related_evidence"]:
                self.assertIn(evidence_id, parsed_evidence_ids)

        for entity in sample["entity_spans"]:
            if entity["source_parser"] == "WorkbookParser":
                self.assertIn(entity["source_field"], by_field)
            if entity["source_parser"] == "ImageEvidenceParser":
                self.assertIn(entity["region_id"], by_metadata_id)
            if entity["source_parser"] == "SymbolParser":
                self.assertIn(entity["symbol_id"], by_metadata_id)

        self.assertTrue(sample["relation_triples"])
        self.assertTrue(sample["graph_annotations"]["edge_types"])


if __name__ == "__main__":
    unittest.main()
