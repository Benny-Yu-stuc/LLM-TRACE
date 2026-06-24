# LLM-TRACE

LLM-TRACE is a traceable, rule-aware, and process-chain-enhanced cross-modal extraction framework for industrial process documents. The project implements the full pipeline for parsing process cards, generating Chief-Engineer routing priors, encoding table/image/symbol evidence, PMCA fusion, MNER/MRE extraction, TM-CGB graph modeling, and traceable structured output.

## Environment

```bash
pip install -r requirements.txt
```

Configure API keys and model paths in:

```text
configs/.env.example
configs/default_config.json
```

Required fields:

```text
CHIEF_ENGINEER_API_KEY=
CHIEF_ENGINEER_API_URL=
CHIEF_ENGINEER_MODEL=
BERT_MODEL_NAME_OR_PATH=
BERT_TOKENIZER_NAME_OR_PATH=
VIT_MODEL_NAME_OR_PATH=
VIT_IMAGE_PROCESSOR_NAME_OR_PATH=
SYMBOL_VOCAB_PATH=
```

## Data Layout

```text
rawdata/
  workbooks/       process-card table files
  images/          process sheet images and image-region annotations
  symbols/         symbol annotations and symbol vocabulary
  manuals/         enterprise process rules and manual clauses
  annotations/     entity, relation, graph, and Chief-Engineer cache annotations
```

Current sample data is based on one OP40 stamping process sheet.

## Check Data

```bash
python -m llm_trace.check_annotations --config configs/default_config.json
```

Run parser-alignment tests:

```bash
python -m unittest discover -s tests -v
```

## Run Extraction

```bash
python -m llm_trace.main --config configs/default_config.json
```

Default output path:

```text
data/outputs/run_result.json
```

Custom output path:

```bash
python -m llm_trace.main --config configs/default_config.json --output data/outputs/op40_result.json
```

## Main Modules

```text
llm_trace/parsers/          document, table, image, symbol, and manual parsing
llm_trace/chief_engineer/   modality prior and routing guidance
llm_trace/fusion/           DPFC and PMCA fusion
llm_trace/models/           BERT, ViT, Symbol Encoder, heads, losses, graph encoder
llm_trace/extraction/       MNER and MRE decoding
llm_trace/graph/            TM-CGB operation-chain graph modeling
llm_trace/training.py       training loop
llm_trace/checkpointing.py  checkpoint save/load
```

## Training Entry

Use `LlmTraceModel` and `LlmTraceTrainer`:

```python
from llm_trace.models.llm_trace_model import LlmTraceModel
from llm_trace.training import LlmTraceTrainer
from llm_trace.utils.config import load_json_config

config = load_json_config("configs/default_config.json")
model = LlmTraceModel(config)
trainer = LlmTraceTrainer(model, config["training"])
```

The training objective is:

```text
L = L_MNER + L_MRE + lambda_c * L_contrastive + lambda_g * L_graph
```
