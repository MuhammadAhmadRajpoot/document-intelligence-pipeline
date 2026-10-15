"""Shared offline fakes: a scripted LLM and an OCR engine that never loads models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from docintel.config import Settings
from docintel.llm.client import ChatMessage
from docintel.models import TextLine
from docintel.pipeline import Pipeline

INVOICE_JSON: dict[str, Any] = {
    "vendor_name": "Northwind Analytics LLC",
    "vendor_address": None,
    "customer_name": "Acme Retail Group",
    "invoice_number": "INV-2025-1001",
    "invoice_date": "2025-03-05",
    "due_date": "2025-04-04",
    "currency": "USD",
    "line_items": [
        {"description": "Consulting", "quantity": 2, "unit_price": 100.0, "amount": 200.0},
        {"description": "Hosting", "quantity": 1, "unit_price": 50.0, "amount": 50.0},
    ],
    "subtotal": 250.0,
    "tax": 20.0,
    "total": 270.0,
}

INVOICE_TEXT_LINES = [
    "Northwind Analytics LLC",
    "INVOICE",
    "Invoice No: INV-2025-1001",
    "Invoice Date: March 05, 2025",
    "Due Date: April 04, 2025",
    "Bill To: Acme Retail Group",
    "Consulting 2 $100.00 $200.00",
    "Hosting 1 $50.00 $50.00",
    "Subtotal $250.00",
    "Sales Tax (8%) $20.00",
    "Total Due $270.00",
]


class FakeLLM:
    """Returns scripted outputs. Items may be dicts or exceptions (raised when reached)."""

    model = "fake-model"

    def __init__(self, extractions: list[Any], document_type: str = "invoice") -> None:
        self.extractions = list(extractions)
        self.document_type = document_type
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def complete_json(
        self,
        messages: list[ChatMessage],
        *,
        schema_name: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((schema_name, messages))
        if schema_name == "classification":
            return {"document_type": self.document_type, "confidence": 0.97, "reason": "test"}
        item = self.extractions.pop(0) if len(self.extractions) > 1 else self.extractions[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeOCR:
    """Emits the given lines stacked vertically, ignoring image content."""

    def __init__(self, lines: list[str] | None = None, confidence: float = 0.9) -> None:
        self.lines = lines if lines is not None else INVOICE_TEXT_LINES
        self.confidence = confidence
        self.calls = 0

    def recognize(self, image_rgb: np.ndarray) -> list[TextLine]:
        self.calls += 1
        return [
            TextLine(
                text=t, bbox=(10.0, 10.0 + 30 * i, 400.0, 34.0 + 30 * i), confidence=self.confidence
            )
            for i, t in enumerate(self.lines)
        ]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, groq_api_key="test-key", max_repair_attempts=2)


@pytest.fixture
def fake_ocr() -> FakeOCR:
    return FakeOCR()


@pytest.fixture
def make_pipeline(settings: Settings, fake_ocr: FakeOCR):
    def factory(extractions: list[Any], document_type: str = "invoice") -> Pipeline:
        return Pipeline(settings, llm=FakeLLM(extractions, document_type), ocr=fake_ocr)

    return factory


@pytest.fixture
def invoice_png_bytes() -> bytes:
    import cv2

    image = np.full((600, 500, 3), 255, dtype=np.uint8)
    cv2.putText(image, "INVOICE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()
