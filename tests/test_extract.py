from docintel.extract import extract_document
from docintel.llm.client import LLMError
from docintel.schemas import DocumentType, Invoice
from docintel.validation import validate_document
from tests.conftest import INVOICE_JSON, FakeLLM

TEXT = "INVOICE Northwind Analytics LLC Total $270.00"


def run(llm, max_repairs=2):
    return extract_document(llm, DocumentType.INVOICE, TEXT, validate_document, max_repairs)


def test_clean_output_needs_one_attempt():
    llm = FakeLLM([INVOICE_JSON])
    outcome = run(llm)
    assert isinstance(outcome.document, Invoice)
    assert len(outcome.attempts) == 1
    assert len(llm.calls) == 1


def test_business_rule_failure_triggers_repair_with_feedback():
    broken = {**INVOICE_JSON, "total": 999.0}
    llm = FakeLLM([broken, INVOICE_JSON])
    outcome = run(llm)
    assert outcome.document.total == 270.0
    assert len(outcome.attempts) == 2
    repair_prompt = llm.calls[1][1][-1]["content"]
    assert "total_mismatch" in repair_prompt
    assert llm.calls[1][1][-2]["role"] == "assistant"


def test_schema_failure_triggers_repair():
    llm = FakeLLM([{**INVOICE_JSON, "total": "lots"}, INVOICE_JSON])
    outcome = run(llm)
    assert outcome.attempts[0].schema_error and "total" in outcome.attempts[0].schema_error
    assert outcome.document.total == 270.0


def test_llm_error_is_retried():
    llm = FakeLLM([LLMError("json_validate_failed"), INVOICE_JSON])
    outcome = run(llm)
    assert outcome.document is not None
    assert outcome.attempts[0].schema_error == "json_validate_failed"


def test_returns_best_attempt_when_never_clean():
    bad_total = {**INVOICE_JSON, "total": 999.0}
    worse = {**INVOICE_JSON, "total": 999.0, "subtotal": 1.0}
    llm = FakeLLM([bad_total, worse, worse])
    outcome = run(llm)
    assert len(outcome.attempts) == 3
    assert outcome.document.subtotal == 250.0  # fewest errors wins
    assert any(i.code == "total_mismatch" for i in outcome.issues)


def test_no_valid_output_returns_none():
    outcome = run(FakeLLM([{"total": "x"}]), max_repairs=1)
    assert outcome.document is None
    assert len(outcome.attempts) == 2


def test_missing_currency_is_inferred_from_text():
    llm = FakeLLM([{**INVOICE_JSON, "currency": None}])
    outcome = run(llm)
    assert outcome.document.currency == "USD"
    assert any(i.code == "currency_inferred" for i in outcome.issues)
