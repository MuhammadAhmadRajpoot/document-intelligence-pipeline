from docintel.classify import classify_document, heuristic_classify
from docintel.llm.client import LLMError
from docintel.schemas import DocumentType
from tests.conftest import FakeLLM


def test_heuristic_classifier():
    assert (
        heuristic_classify("INVOICE  Bill To: ACME  Due Date: 2025-01-01").document_type
        is DocumentType.INVOICE
    )
    assert (
        heuristic_classify("RECEIPT cashier: Ann  VISA  change 0.00").document_type
        is DocumentType.RECEIPT
    )
    assert (
        heuristic_classify("Dear team, see you at the meeting.").document_type is DocumentType.OTHER
    )


def test_llm_classification_is_used():
    result = classify_document("anything", FakeLLM([{}], document_type="receipt"))
    assert result.document_type is DocumentType.RECEIPT
    assert result.confidence == 0.97


def test_falls_back_to_heuristic_when_llm_fails():
    class Broken(FakeLLM):
        def complete_json(self, messages, *, schema_name, json_schema=None):
            raise LLMError("down")

    result = classify_document("INVOICE Bill To due date", Broken([{}]))
    assert result.document_type is DocumentType.INVOICE


def test_falls_back_when_llm_returns_bad_label():
    class Weird(FakeLLM):
        def complete_json(self, messages, *, schema_name, json_schema=None):
            return {"document_type": "spaceship", "confidence": 2}

    assert classify_document("receipt cashier", Weird([{}])).document_type is DocumentType.RECEIPT
