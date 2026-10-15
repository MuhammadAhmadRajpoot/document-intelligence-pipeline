from datetime import date

import pytest
from pydantic import ValidationError

from docintel.llm.prompts import schema_outline
from docintel.schemas import GenericDocument, Invoice, Receipt
from tests.conftest import INVOICE_JSON


def test_invoice_normalises_loose_llm_output():
    raw = {
        **INVOICE_JSON,
        "invoice_date": "March 5, 2025",
        "total": "$270.00",
        "currency": "usd",
        "due_date": "n/a",
    }
    invoice = Invoice.model_validate(raw)
    assert invoice.invoice_date == date(2025, 3, 5)
    assert invoice.total == 270.0
    assert invoice.currency == "USD"
    assert invoice.due_date is None


def test_invoice_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValidationError):
        Invoice.model_validate({**INVOICE_JSON, "surprise": 1})
    with pytest.raises(ValidationError):
        Invoice.model_validate({**INVOICE_JSON, "total": "lots"})


def test_receipt_and_generic_schemas():
    receipt = Receipt.model_validate({"merchant_name": "Cafe", "total": "4,50", "tip": None})
    assert receipt.total == 4.5
    generic = GenericDocument.model_validate({"fields": [{"key": "Room", "value": "B"}]})
    assert generic.fields[0].value == "B"


def test_schema_outline_is_compact_and_typed():
    outline = schema_outline(Invoice)
    assert outline["invoice_date"] == "YYYY-MM-DD | null"
    assert outline["total"] == "number | null"
    assert outline["line_items"][0]["description"] == "string"
