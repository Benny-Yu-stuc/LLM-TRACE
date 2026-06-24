# LLM-TRACE Engineering Document Extraction


## Technical Chain

1. Document Parsing Layer
   - Parses Excel process cards, page image metadata, symbol detection files, and manual clauses.
   - Converts every source item into traceable evidence units with stable evidence IDs.

2. Chief-Engineer Layer
   - Builds evidence-constrained routing context.
   - Produces modality priors for table, drawing, and symbol evidence.
   - Falls back to deterministic local routing when no key is configured.

3. DPFC and PMCA Fusion Layer
   - Encodes table text, drawing evidence, and symbol evidence with separate experts.
   - Applies prior-modulated corroborative attention style weighting.
   - Produces fused operation representations for extraction.

4. Extraction Layer
   - Runs MNER-style entity extraction with evidence anchors.
   - Runs MRE-style relation extraction with strict head, tail, relation, and evidence links.

5. TM-CGB Process Chain Layer
   - Builds an operation-chain graph from operations, evidence, entities, relations, and constraints.
   - Computes a deterministic graph consistency score that can be replaced by a trainable graph encoder later.

6. Output Layer
   - Writes traceable JSON results for downstream retrieval, audit, and knowledge graph construction.

## Project Layout

```text
src/
  configs/
    default_config.json
    .env.example
  data/
    raw/
      workbooks/
      images/
      symbols/
      manuals/
    processed/
    outputs/
  llm_trace/
    chief_engineer/
    parsers/
    fusion/
    extraction/
    graph/
    eval/
    utils/
    main.py
    pipeline.py
    schemas.py
  tests/
```

## Quick Start

From the `src` folder:

```bash
python -m llm_trace.main --demo
```

This creates a small built-in demo document and writes outputs to:

```text
src/data/outputs/run_result.json
```

To run with your own data:

```bash
python -m llm_trace.main --config configs/default_config.json
```

Put files in:

```text
data/raw/workbooks/   Excel process cards, .xlsx
data/raw/images/      Page images and optional .ocr.json sidecars
data/raw/symbols/     Symbol detection JSON files
data/raw/manuals/     Manual clauses, .txt, .md, or .json
```

## API Key

The external Chief-Engineer endpoint is optional. Keys are blank by default.

Copy:

```text
configs/.env.example
```

Set:

```text
CHIEF_ENGINEER_API_KEY=
CHIEF_ENGINEER_API_URL=
CHIEF_ENGINEER_MODEL=
```

When the API key or URL is empty, the system uses the local deterministic router.

## Output

The main output contains:

- parsed evidence
- Chief-Engineer modality priors and routing guidance
- fused operation summaries
- extracted entities and relations
- operation-chain graph
- quality and traceability metrics

## Notes

The repository contains no real enterprise data. The data folders are placeholders. Replace or extend the parsers and expert encoders according to your deployment environment.
