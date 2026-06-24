"""Checkpoint save/load helpers for LLM-TRACE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_trace.models.llm_trace_model import LlmTraceModel


def save_checkpoint(model: LlmTraceModel, config: dict[str, Any], path: str | Path) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "state_dict": collect_state_dict(model),
        },
        output_path,
    )
    output_path.with_suffix(".config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location="cpu")


def collect_state_dict(model: LlmTraceModel) -> dict[str, Any]:
    state: dict[str, Any] = {}
    modules = {
        "process_bert": model.fusion.process_expert.model,
        "geometry_vit": model.fusion.geometry_expert.model,
        "symbol_category_embedding": model.fusion.symbol_expert.category_embedding,
        "symbol_bbox_projection": model.fusion.symbol_expert.bbox_projection,
        "symbol_confidence_projection": model.fusion.symbol_expert.confidence_projection,
        "symbol_output_projection": model.fusion.symbol_expert.output_projection,
        "pmca_q_table": model.fusion.pmca.q_table,
        "pmca_q_drawing": model.fusion.pmca.q_drawing,
        "pmca_q_symbol": model.fusion.pmca.q_symbol,
        "pmca_k_proj": model.fusion.pmca.k_proj,
        "pmca_v_proj": model.fusion.pmca.v_proj,
        "pmca_attention": model.fusion.pmca.attention,
        "pmca_ffn": model.fusion.pmca.ffn,
        "entity_classifier": model.heads.entity_classifier,
        "relation_classifier": model.heads.relation_classifier,
        "graph_self_loop": model.graph_encoder.self_loop,
    }
    for name, module in modules.items():
        if hasattr(module, "state_dict"):
            state[name] = module.state_dict()
    for edge_type, layer in model.graph_encoder.edge_transforms.items():
        state[f"graph_edge_transform::{edge_type}"] = layer.state_dict()
    return state
