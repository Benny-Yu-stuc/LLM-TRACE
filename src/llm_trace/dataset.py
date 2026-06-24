"""Dataset interfaces for annotated MNER, MRE, and graph training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from llm_trace.schemas import Evidence
from llm_trace.utils.io import read_json


@dataclass(slots=True)
class AnnotatedSample:
    document_id: str
    op_id: str
    evidence: list[Evidence]
    entity_spans: list[dict[str, Any]] = field(default_factory=list)
    relation_triples: list[dict[str, Any]] = field(default_factory=list)
    graph_annotations: dict[str, Any] = field(default_factory=dict)


class ProcessCardDataset:
    """Reader for article-style document-level splits.

    Expected JSONL fields:
    - document_id
    - op_id
    - evidence
    - entity_spans
    - relation_triples
    - graph_annotations
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield read_json_line(line)

    def __len__(self) -> int:
        with self.jsonl_path.open("r", encoding="utf-8") as file:
            return sum(1 for line in file if line.strip())


class LlmTraceCollator:
    """Collator interface for BERT, ViT, symbol, PMCA, and TM-CGB tensors."""

    def __init__(self, entity_label_to_id: dict[str, int], relation_label_to_id: dict[str, int]) -> None:
        self.entity_label_to_id = entity_label_to_id
        self.relation_label_to_id = relation_label_to_id

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "document_ids": [sample["document_id"] for sample in samples],
            "op_ids": [sample["op_id"] for sample in samples],
            "evidence": [sample.get("evidence", []) for sample in samples],
            "entity_spans": [sample.get("entity_spans", []) for sample in samples],
            "relation_triples": [sample.get("relation_triples", []) for sample in samples],
            "graph_annotations": [sample.get("graph_annotations", {}) for sample in samples],
            "entity_label_to_id": self.entity_label_to_id,
            "relation_label_to_id": self.relation_label_to_id,
        }


def read_json_line(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)
