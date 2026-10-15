from datetime import date

from docintel.models import Severity
from docintel.schemas import GenericDocument, Invoice, Receipt
from docintel.validation import Tolerance, validate_document
from tests.conftest import INVOICE_JSON

TODAY = date(2025, 10, 1)


def codes(issues, severity=Severity.ERROR):
    return {i.code for i in issues if i.severity is severity}


def test_consistent_invoice_has_no_errors():
    issues = validate_document(Invoice.model_validate(INVOICE_JSON), today=TODAY)
    assert codes(issues) == set()
    assert codes(issues, Severity.WARNING) == set()


def test_subtotal_and_total_mismatch_are_errors():
    doc = Invoice.model_validate({**INVOICE_JSON, "subtotal": 240.0, "total": 300.0})
    assert {"subtotal_mismatch", "total_mismatch"} <= codes(validate_document(doc, today=TODAY))


def test_line_item_math_error():
    items = [{"description": "Consulting", "quantity": 2, "unit_price": 100.0, "amount": 210.0}]
    doc = Invoice.model_validate(
        {**INVOICE_JSON, "line_items": items, "subtotal": 210.0, "tax": 0, "total": 210.0}
    )
    assert codes(validate_document(doc, today=TODAY)) == {"line_item_math"}


def test_tolerance_absorbs_rounding():
    doc = Invoice.model_validate({**INVOICE_JSON, "total": 270.01})
    assert codes(validate_document(doc, Tolerance(0.02, 0.0), today=TODAY)) == set()


def test_due_before_issue_and_missing_total():
    doc = Invoice.model_validate({**INVOICE_JSON, "due_date": "2025-01-01", "total": None})
    assert {"due_before_issue", "total_missing"} <= codes(validate_document(doc, today=TODAY))


def test_total_without_subtotal_uses_line_items():
    doc = Invoice.model_validate({**INVOICE_JSON, "subtotal": None})
    assert codes(validate_document(doc, today=TODAY)) == set()


def test_receipt_includes_tip_in_total():
    receipt = Receipt.model_validate(
        {
            "merchant_name": "Cafe",
            "transaction_date": "2025-02-02",
            "currency": "EUR",
            "line_items": [{"description": "Espresso", "amount": 3.0}],
            "subtotal": 3.0,
            "tax": 0.27,
            "tip": 0.3,
            "total": 3.57,
        }
    )
    assert codes(validate_document(receipt, today=TODAY)) == set()


def test_missing_fields_are_warnings():
    doc = Invoice.model_validate({"total": 10.0, "line_items": [], "subtotal": 10.0})
    warnings = codes(validate_document(doc, today=TODAY), Severity.WARNING)
    assert {
        "vendor_missing",
        "invoice_number_missing",
        "invoice_date_missing",
        "currency_missing",
    } <= warnings


def test_implausible_date_warning():
    doc = Invoice.model_validate(
        {**INVOICE_JSON, "invoice_date": "1970-01-01", "due_date": "1970-02-01"}
    )
    assert "invoice_date_implausible" in codes(
        validate_document(doc, today=TODAY), Severity.WARNING
    )


def test_generic_without_fields_warns():
    assert codes(validate_document(GenericDocument()), Severity.WARNING) == {"no_fields"}
