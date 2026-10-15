from docintel.confidence import (
    RULE_FAILURE_PENALTY,
    UNGROUNDED_CONFIDENCE,
    review_reasons,
    score_fields,
)
from docintel.models import Page, Severity, TextLine, TextSource, ValidationIssue
from docintel.schemas import Invoice
from tests.conftest import INVOICE_JSON, INVOICE_TEXT_LINES


def make_page(lines, confidence=1.0, source=TextSource.TEXT_LAYER):
    return Page(
        index=0,
        width=500,
        height=800,
        source=source,
        lines=[
            TextLine(text=t, bbox=(0, 30.0 * i, 300, 30.0 * i + 20), confidence=confidence)
            for i, t in enumerate(lines)
        ],
    )


def test_grounded_fields_inherit_line_confidence_and_bbox():
    page = make_page(INVOICE_TEXT_LINES, confidence=0.9, source=TextSource.OCR)
    evidence = score_fields(Invoice.model_validate(INVOICE_JSON), [page], [])
    assert evidence["invoice_number"].grounded
    assert evidence["invoice_number"].confidence == 0.9
    assert evidence["invoice_date"].grounded  # "March 05, 2025" matches 2025-03-05
    total = evidence["total"]
    assert total.bbox is not None
    assert page.lines[int(total.bbox[1] // 30)].text == "Total Due $270.00"


def test_ungrounded_and_penalised_fields():
    page = make_page(["INVOICE", "Total Due $270.00"])
    issue = ValidationIssue(
        code="total_mismatch", message="x", severity=Severity.ERROR, fields=["total"]
    )
    evidence = score_fields(Invoice.model_validate(INVOICE_JSON), [page], [issue])
    assert not evidence["vendor_name"].grounded
    assert evidence["vendor_name"].confidence == UNGROUNDED_CONFIDENCE
    assert evidence["total"].confidence == RULE_FAILURE_PENALTY


def test_fuzzy_match_tolerates_ocr_noise():
    page = make_page(["Northwind Analytlcs LLC"], confidence=0.8)
    evidence = score_fields(Invoice.model_validate(INVOICE_JSON), [page], [])
    assert evidence["vendor_name"].grounded
    assert 0.6 < evidence["vendor_name"].confidence < 0.8


def test_review_reasons():
    doc = Invoice.model_validate({**INVOICE_JSON, "invoice_number": None})
    page = make_page(INVOICE_TEXT_LINES)
    evidence = score_fields(doc, [page], [])
    reasons = review_reasons(doc, evidence, [], classification_confidence=0.4, threshold=0.6)
    assert "missing key field: invoice_number" in reasons
    assert any("uncertain document type" in r for r in reasons)
    clean = Invoice.model_validate(INVOICE_JSON)
    assert review_reasons(clean, score_fields(clean, [page], []), [], 0.95, 0.6) == []


def test_page_text_keeps_rows_together():
    page = Page(
        index=0,
        width=100,
        height=100,
        source=TextSource.OCR,
        lines=[
            TextLine(text="$50.00", bbox=(80, 10, 99, 20), confidence=1.0),
            TextLine(text="Total", bbox=(0, 11, 30, 21), confidence=1.0),
            TextLine(text="Thanks", bbox=(0, 40, 30, 50), confidence=1.0),
        ],
    )
    assert page.text == "Total   $50.00\nThanks"
