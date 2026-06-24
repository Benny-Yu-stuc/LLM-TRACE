"""CLI for checking rawdata and annotation consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_trace.annotation_schema import validate_annotation_sample
from llm_trace.parsers.document_parser import DocumentParser
from llm_trace.utils.config import load_json_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate parser outputs against annotation JSONL files.")
    parser.add_argument("--config", default="configs/default_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    documents = DocumentParser().parse(config["data"])
    by_document = {document.document_id: document for document in documents}
    annotation_dir = Path(config["data"]["annotation_dir"])
    errors: list[str] = []
    for path in sorted(annotation_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            document_id = sample.get("document_id", "")
            document = by_document.get(document_id)
            if document is None:
                errors.append(f"{path}:{line_no} document not parsed: {document_id}")
                continue
            for error in validate_annotation_sample(sample, document.evidence):
                errors.append(f"{path}:{line_no} {error}")
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"Annotation check passed for {len(documents)} parsed document(s).")


if __name__ == "__main__":
    main()
