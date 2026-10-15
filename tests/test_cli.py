import json

from typer.testing import CliRunner

from docintel import cli
from tests.conftest import INVOICE_JSON
from tests.test_ingest import text_pdf
from tests.test_pipeline import PDF_TEXT

runner = CliRunner()


def test_extract_directory_to_jsonl(tmp_path, monkeypatch, make_pipeline):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (docs / name).write_bytes(text_pdf(PDF_TEXT))
    (docs / "ignored.txt").write_text("not a document")
    monkeypatch.setattr(cli, "_pipeline", lambda: make_pipeline([INVOICE_JSON]))

    out = tmp_path / "results.jsonl"
    result = runner.invoke(cli.app, ["extract", str(docs), "--out", str(out), "-c", "2"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["is_valid"] for row in rows)


def test_extract_with_no_documents_exits_1(tmp_path):
    result = runner.invoke(cli.app, ["extract", str(tmp_path)])
    assert result.exit_code == 1


def test_generate_dataset(tmp_path):
    out = tmp_path / "ds"
    result = runner.invoke(cli.app, ["generate-dataset", "--out", str(out)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))
    assert len(manifest) == 28
    assert {m["variant"] for m in manifest} == {"digital", "scanned"}
    assert all((out / m["file"]).exists() for m in manifest)
