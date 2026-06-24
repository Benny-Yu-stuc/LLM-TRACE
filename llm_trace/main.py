"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_trace.pipeline import LlmTracePipeline
from llm_trace.schemas import to_plain_data
from llm_trace.utils.config import ensure_dirs, load_json_config
from llm_trace.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-TRACE extraction scaffold.")
    parser.add_argument("--config", default="configs/default_config.json", help="Path to config JSON.")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo data.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    data_config = config.get("data", {})
    ensure_dirs(
        [
            data_config.get("processed_dir", "data/processed"),
            data_config.get("output_dir", "data/outputs"),
            config.get("chief_engineer", {}).get("cache_dir", "data/processed/chief_engineer_cache"),
        ]
    )

    pipeline = LlmTracePipeline(config)
    documents = pipeline.load_documents(demo=args.demo)
    results = [pipeline.run_document(document) for document in documents]

    output_path = Path(args.output) if args.output else Path(data_config.get("output_dir", "data/outputs")) / "run_result.json"
    payload = {
        "project": config.get("project_name", "LLM-TRACE"),
        "demo": bool(args.demo),
        "document_count": len(results),
        "results": [to_plain_data(result) for result in results],
    }
    write_json(output_path, payload)

    for result in results:
        metrics = result.metrics
        print(
            f"[{result.document_id}] evidence={metrics['evidence_count']} "
            f"entities={metrics['entity_count']} relations={metrics['relation_count']} "
            f"graph_nodes={metrics['tm_cgb']['graph_node_count']} "
            f"graph_edges={metrics['tm_cgb']['graph_edge_count']}"
        )
    print(f"Output written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
