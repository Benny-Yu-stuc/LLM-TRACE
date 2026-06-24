"""Prompt construction for the Chief-Engineer layer."""

from __future__ import annotations

from llm_trace.schemas import Evidence


def build_chief_engineer_prompt(
    *,
    document_id: str,
    op_id: str,
    operation_evidence: list[Evidence],
    manual_evidence: list[Evidence],
) -> str:
    """Build an evidence-constrained prompt.

    The prompt asks for routing priors only. It explicitly prevents direct
    final entity or relation generation, matching the manuscript appendix.
    """

    table_items = [item for item in operation_evidence if item.modality == "table"]
    drawing_items = [item for item in operation_evidence if item.modality == "drawing"]
    symbol_items = [item for item in operation_evidence if item.modality == "symbol"]

    def render(items: list[Evidence], limit: int = 30) -> str:
        lines = []
        for item in items[:limit]:
            lines.append(
                f"- evidence_id={item.evidence_id}; field={item.field_name}; "
                f"text={item.text}; confidence={item.confidence:.3f}"
            )
        return "\n".join(lines) if lines else "- none"

    return f"""
You are the Chief-Engineer routing module for industrial process document extraction.
Return JSON only. Do not output final entities or final relations.

Task:
- Produce modality_prior with pi_table, pi_drawing, and pi_symbol summing to 1.
- Produce routing_guidance for table_focus, drawing_focus, and symbol_focus.
- Produce candidate_constraints using only evidence-backed rules.
- Put unsupported or conflicting candidates into uncertain_items.

Document metadata:
- document_id: {document_id}
- op_id: {op_id}

Tabular text evidence:
{render(table_items)}

Engineering image evidence:
{render(drawing_items)}

Symbol evidence:
{render(symbol_items)}

External process rules and manuals:
{render(manual_evidence, limit=20)}

Required JSON schema:
{{
  "document_id": "string",
  "op_id": "string",
  "modality_prior": {{
    "pi_table": 0.0,
    "pi_drawing": 0.0,
    "pi_symbol": 0.0
  }},
  "routing_guidance": {{
    "table_focus": [{{"field_name": "string", "cell_id": "string", "evidence_id": "string", "reason": "string", "priority": 1.0}}],
    "drawing_focus": [{{"region_id": "string", "evidence_id": "string", "reason": "string", "priority": 1.0}}],
    "symbol_focus": [{{"symbol_id": "string", "evidence_id": "string", "reason": "string", "priority": 1.0}}]
  }},
  "candidate_constraints": [
    {{"constraint_type": "type | sequence | unit | evidence", "description": "string", "related_evidence": ["evidence_id"]}}
  ],
  "uncertain_items": [
    {{"item": "string", "reason": "string", "required_evidence": "string"}}
  ]
}}
""".strip()
