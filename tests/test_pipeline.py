import pytest

from docintel.config import Settings
from docintel.pipeline import ConfigurationError, build_llm
from docintel.schemas import DocumentType
from tests.conftest import INVOICE_JSON
from tests.test_ingest import text_pdf

PDF_TEXT = (
    "INVOICE Northwind Analytics LLC Invoice No: INV-2025-1001 Date: 2025-03-05 "
    "Bill To: Acme Retail Group Subtotal $250.00 Tax $20.00 Total $270.00"
)


def test_end_to_end_valid_invoice(make_pipeline):
    pipeline = make_pipeline([INVOICE_JSON])
    result, loaded = pipeline.process("inv.pdf", text_pdf(PDF_TEXT))
    assert result.document_type is DocumentType.INVOICE
    assert result.is_valid
    assert result.data["total"] == 270.0
    assert result.field_confidence["invoice_number"].grounded
    assert set(result.timings_ms) == {"ingest_ms", "classify_ms", "extract_ms", "total_ms"}
    assert loaded.pages[0].source.value == "text_layer"


def test_inconsistent_document_is_flagged_for_review(make_pipeline):
    pipeline = make_pipeline([{**INVOICE_JSON, "total": 300.0}])  # model never fixes it
    result, _ = pipeline.process("inv.pdf", text_pdf(PDF_TEXT))
    assert not result.is_valid
    assert result.needs_review
    assert len(result.attempts) == 3
    assert any("total" in reason for reason in result.review_reasons)


def test_forced_type_skips_classification(make_pipeline):
    pipeline = make_pipeline([INVOICE_JSON])
    result, _ = pipeline.process("inv.pdf", text_pdf(PDF_TEXT), DocumentType.INVOICE)
    assert result.classification_confidence == 1.0
    assert all(name != "classification" for name, _ in pipeline.llm.calls)


def test_process_many_reports_errors_without_raising(make_pipeline, tmp_path):
    good = tmp_path / "a.pdf"
    good.write_bytes(text_pdf(PDF_TEXT))
    bad = tmp_path / "b.txt"
    bad.write_text("x")
    done = []
    results = make_pipeline([INVOICE_JSON]).process_many(
        [good, bad], concurrency=2, on_done=done.append
    )
    assert results[0].error is None and results[0].is_valid
    assert results[1].error and "unsupported" in results[1].error
    assert len(done) == 2


def test_missing_api_key_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        build_llm(Settings(_env_file=None, groq_api_key=None))
