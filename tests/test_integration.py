"""Real Groq calls. Skipped by default; run with `pytest -m integration`."""

from pathlib import Path

import pytest

from docintel.classify import classify_document
from docintel.config import Settings
from docintel.pipeline import Pipeline, build_llm
from docintel.schemas import DocumentType
from tests.conftest import FakeOCR

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    settings = Settings()
    if settings.groq_api_key is None:
        pytest.skip("GROQ_API_KEY not configured")
    return settings


def test_groq_classifies_a_receipt(real_settings):
    text = "CAFE MERIDIAN\nReceipt #207151\nEspresso 2.90\nTOTAL €2.90\nPaid by VISA"
    result = classify_document(text, build_llm(real_settings))
    assert result.document_type is DocumentType.RECEIPT


def test_groq_extracts_digital_invoice_end_to_end(real_settings):
    # Digital PDF: text layer only, so the OCR engine must never be called.
    ocr = FakeOCR()
    pipeline = Pipeline(real_settings, ocr=ocr)
    result = pipeline.process_path(EXAMPLES / "invoice_digital.pdf")
    assert ocr.calls == 0
    assert result.error is None
    assert result.document_type is DocumentType.INVOICE
    assert result.is_valid, result.issues
    assert result.data["invoice_number"].startswith("INV-2025-")
    assert result.data["currency"] == "GBP"
