import json

import pytest

from docintel.evaluation.metrics import field_matches, score_document, summarise
from docintel.export import to_csv, write_results
from docintel.models import ExtractionResult
from docintel.schemas import DocumentType
from tests.conftest import INVOICE_JSON


def make_result(**overrides):
    base = {
        "file": "inv.pdf",
        "document_type": DocumentType.INVOICE,
        "classification_confidence": 0.9,
        "data": dict(INVOICE_JSON),
        "is_valid": True,
        "needs_review": False,
        "timings_ms": {"total_ms": 1500.0},
    }
    return ExtractionResult(**{**base, **overrides})


def test_write_results_formats(tmp_path):
    results = [make_result()]
    write_results(results, tmp_path / "r.jsonl")
    line = json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["data"]["total"] == 270.0
    write_results(results, tmp_path / "r.xlsx")
    assert (tmp_path / "r.xlsx").stat().st_size > 0
    with pytest.raises(ValueError):
        write_results(results, tmp_path / "r.txt")


def test_csv_summarises_line_items():
    header, row = to_csv([make_result()]).splitlines()[:2]
    assert "line_item_count" in header
    assert ",2," in row


def test_field_matches():
    assert field_matches("vendor_name", "Acme  Ltd", "acme ltd.")
    assert not field_matches("vendor_name", "Acme Ltd", "AcmeLtd")
    assert field_matches("total", 270.0, 270.004)
    assert not field_matches("total", 270.0, 270.5)
    assert field_matches("tip", None, None)
    assert not field_matches("tip", None, 1.0)
    items = INVOICE_JSON["line_items"]
    assert field_matches("line_items", items, list(reversed(items)))
    assert not field_matches("line_items", items, items[:1])


def test_score_and_summarise():
    truth = {"file": "inv.pdf", "variant": "digital", **INVOICE_JSON, "document_type": "invoice"}
    good = score_document(truth, make_result())
    assert good.all_fields_correct
    wrong_type = score_document(truth, make_result(document_type=DocumentType.RECEIPT))
    assert not any(wrong_type.field_hits.values())
    bad = score_document(truth, make_result(data={**INVOICE_JSON, "total": 1.0}, needs_review=True))
    report = summarise([good, bad])
    assert report["by_field"]["total"]["accuracy"] == 0.5
    assert report["overall"]["review_recall_on_wrong_docs"] == 1.0
    assert report["overall"]["false_review_rate_on_correct_docs"] == 0.0
    assert report["documents"][1]["wrong_fields"] == ["total"]
