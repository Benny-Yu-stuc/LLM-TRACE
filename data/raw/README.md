# Raw Data Interface

No real data is included. Put your enterprise files under these folders.

## workbooks

Supported placeholders:

- `.xlsx`: parsed with `openpyxl` if installed; otherwise creates a workbook placeholder evidence item.
- `.csv`: use columns such as:

```csv
document_id,op_id,sheet_name,cell,field_name,value,unit,confidence
part_001,OP10,OP10,B4,equipment,Press-1200T,,1.0
part_001,OP10,OP10,C8,pressure,450,kN,1.0
```

- `.json`: array or `{ "rows": [...] }` with similar fields.

## images

Use OCR or image-region sidecar JSON:

```json
{
  "regions": [
    {
      "document_id": "part_001",
      "op_id": "OP10",
      "image_file": "part_001_op10.png",
      "region_type": "forming_region",
      "text": "forming area and draw bead",
      "bbox": [10, 20, 240, 180],
      "confidence": 0.86
    }
  ]
}
```

## symbols

Use symbol detector JSON:

```json
{
  "symbols": [
    {
      "document_id": "part_001",
      "op_id": "OP10",
      "symbol_type": "SC",
      "text": "SC quality control symbol",
      "bbox": [80, 120, 112, 145],
      "confidence": 0.91
    }
  ]
}
```

## manuals

Use `.txt`, `.md`, or JSON:

```json
{
  "clauses": [
    {
      "document_id": "part_001",
      "op_id": "GLOBAL",
      "rule_type": "quality_standard",
      "text": "Clearance values must use mm units."
    }
  ]
}
```
