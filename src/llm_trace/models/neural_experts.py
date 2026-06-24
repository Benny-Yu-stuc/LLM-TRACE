"""BERT, ViT, and structured symbol expert implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

from llm_trace.models.backends import ModelLoadError, require_backend
from llm_trace.schemas import EncodedEvidence, Evidence, Modality


class TorchBackedExpert(ABC):
    """Base class for real model-backed experts."""

    def __init__(self, modality: Modality, config: dict[str, Any], vector_dim: int, strict: bool = False) -> None:
        self.modality = modality
        self.config = config
        self.vector_dim = vector_dim
        self.strict = strict
        self.backend_name = config.get("backend", "")
        self._load_backend()

    def parameters(self) -> list[Any]:
        params: list[Any] = []
        for value in self.__dict__.values():
            if hasattr(value, "parameters"):
                params.extend(list(value.parameters()))
        return params

    @abstractmethod
    def _load_backend(self) -> None:
        """Load tokenizer, processor, model, and projection layers."""

    def encode(self, evidence: list[Evidence]) -> list[EncodedEvidence]:
        items = [item for item in evidence if item.modality == self.modality]
        return self._encode_with_model(items)

    @abstractmethod
    def _encode_with_model(self, items: list[Evidence]) -> list[EncodedEvidence]:
        """Encode modality-specific evidence into neural vectors."""

class BertProcessExpert(TorchBackedExpert):
    """Text/table expert corresponding to the BERT side in the article."""

    def __init__(self, config: dict[str, Any], vector_dim: int, strict: bool = False) -> None:
        self.tokenizer = None
        self.model = None
        self.torch = None
        super().__init__("table", config, vector_dim, strict)

    def _load_backend(self) -> None:
        model_path = self.config.get("bert_model_name_or_path", "")
        tokenizer_path = self.config.get("tokenizer_name_or_path") or model_path
        require_backend(True, "torch", "transformers")
        if not model_path:
            raise ModelLoadError("bert_model_name_or_path is empty.")
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.torch = torch

    def _encode_with_model(self, items: list[Evidence]) -> list[EncodedEvidence]:
        if not items:
            return []
        max_length = int(self.config.get("max_length", 512))
        texts = [f"{item.field_name} {item.text} {item.unit}".strip() for item in items]
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        outputs = self.model(**encoded_inputs)
        hidden = outputs.last_hidden_state[:, 0, :]
        vectors = hidden.detach().cpu().tolist()
        return [self._to_encoded(item, vector, "bert") for item, vector in zip(items, vectors)]

    def _to_encoded(self, item: Evidence, vector: list[float], backend: str) -> EncodedEvidence:
        return EncodedEvidence(
            evidence_id=item.evidence_id,
            modality=item.modality,
            op_id=item.op_id,
            vector=_fit_dim(vector, self.vector_dim),
            confidence=item.confidence,
            text=item.text,
            metadata={**item.metadata, "encoder_backend": backend},
        )


class VitGeometryExpert(TorchBackedExpert):
    """Engineering drawing expert corresponding to ViT-B/16 in the article."""

    def __init__(self, config: dict[str, Any], vector_dim: int, strict: bool = False) -> None:
        self.processor = None
        self.model = None
        self.torch = None
        super().__init__("drawing", config, vector_dim, strict)

    def _load_backend(self) -> None:
        model_path = self.config.get("vit_model_name_or_path", "")
        processor_path = self.config.get("image_processor_name_or_path") or model_path
        require_backend(True, "torch", "transformers", "PIL")
        if not model_path:
            raise ModelLoadError("vit_model_name_or_path is empty.")
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self.processor = AutoImageProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.torch = torch

    def _encode_with_model(self, items: list[Evidence]) -> list[EncodedEvidence]:
        if not items:
            return []
        from PIL import Image

        images = []
        for item in items:
            source_file = item.location.source_file
            if not source_file or not Path(source_file).exists():
                raise FileNotFoundError(f"Image evidence source not found: {source_file}")
            images.append(Image.open(source_file).convert("RGB"))
        encoded_inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**encoded_inputs)
        hidden = outputs.last_hidden_state[:, 0, :]
        vectors = hidden.detach().cpu().tolist()
        return [
            EncodedEvidence(
                evidence_id=item.evidence_id,
                modality=item.modality,
                op_id=item.op_id,
                vector=_fit_dim(vector, self.vector_dim),
                confidence=item.confidence,
                text=item.text,
                metadata={**item.metadata, "encoder_backend": "vit"},
            )
            for item, vector in zip(items, vectors)
        ]


class StructuredSymbolExpert(TorchBackedExpert):
    """Symbol expert using category, bbox, confidence, and evidence ID fields."""

    def __init__(self, config: dict[str, Any], vector_dim: int, strict: bool = False) -> None:
        self.symbol_vocab = self._load_vocab(config.get("symbol_vocab_path", ""))
        self.torch = None
        self.category_embedding = None
        self.bbox_projection = None
        self.confidence_projection = None
        self.output_projection = None
        super().__init__("symbol", config, vector_dim, strict)

    def _load_backend(self) -> None:
        require_backend(True, "torch")
        import torch

        nn = torch.nn
        self.torch = torch
        category_dim = int(self.config.get("category_embedding_dim", 128))
        position_dim = int(self.config.get("position_embedding_dim", 128))
        confidence_dim = int(self.config.get("confidence_embedding_dim", 32))
        self.category_embedding = nn.Embedding(max(self.symbol_vocab.values(), default=0) + 2, category_dim)
        self.bbox_projection = nn.Linear(int(self.config.get("bbox_dim", 4)), position_dim)
        self.confidence_projection = nn.Linear(1, confidence_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(category_dim + position_dim + confidence_dim, self.vector_dim),
            nn.GELU(),
            nn.Linear(self.vector_dim, self.vector_dim),
        )

    def _encode_with_model(self, items: list[Evidence]) -> list[EncodedEvidence]:
        if not items:
            return []
        vectors = []
        vocab_ids = []
        for item in items:
            symbol_key = item.field_name or item.text
            vocab_ids.append(self.symbol_vocab.get(symbol_key, self.symbol_vocab.get(symbol_key.lower(), 0)))
        category_ids = self.torch.tensor(vocab_ids, dtype=self.torch.long)
        bboxes = self.torch.tensor([_bbox4(item.location.bbox) for item in items], dtype=self.torch.float32)
        confidences = self.torch.tensor([[float(item.confidence)] for item in items], dtype=self.torch.float32)
        category_repr = self.category_embedding(category_ids)
        bbox_repr = self.bbox_projection(bboxes)
        confidence_repr = self.confidence_projection(confidences)
        output = self.output_projection(self.torch.cat([category_repr, bbox_repr, confidence_repr], dim=-1))
        vectors = output.detach().cpu().tolist()
        return [
            EncodedEvidence(
                evidence_id=item.evidence_id,
                modality=item.modality,
                op_id=item.op_id,
                vector=_fit_dim(vector, self.vector_dim),
                confidence=item.confidence,
                text=item.text,
                metadata={**item.metadata, "encoder_backend": "structured_symbol_encoder", "symbol_vocab_id": vocab_id},
            )
            for item, vector, vocab_id in zip(items, vectors, vocab_ids)
        ]

    @staticmethod
    def _load_vocab(path: str) -> dict[str, int]:
        if not path or not Path(path).exists():
            return {"SC": 1, "CC": 2, "QC": 3, "arrow": 4, "dimension": 5, "tolerance": 6}
        vocab: dict[str, int] = {}
        for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            token = line.strip()
            if token:
                vocab[token] = index
        return vocab


def _fit_dim(vector: list[float], dim: int) -> list[float]:
    if len(vector) == dim:
        return vector
    if len(vector) > dim:
        return vector[:dim]
    return vector + [0.0] * (dim - len(vector))


def _bbox4(values: list[float]) -> list[float]:
    padded = [float(value) for value in values[:4]]
    return padded + [0.0] * (4 - len(padded))
