"""Export extraction results as JSONL, CSV or Excel."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from docintel.models import ExtractionResult

SUMMARY_COLUMNS = ["file", "document_type", "is_valid", "needs_review", "review_reasons", "error"]
LINE_ITEM_COLUMNS = ["file", "line", "description", "quantity", "unit_price", "amount"]


def result_to_json(result: ExtractionResult, include_ocr: bool = False) -> dict[str, Any]:
    exclude = None if include_ocr else {"pages"}
    return result.model_dump(mode="json", exclude=exclude)


def write_jsonl(results: Iterable[ExtractionResult], path: Path, include_ocr: bool = False) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result_to_json(result, include_ocr), ensure_ascii=False) + "\n")


def summary_rows(results: Iterable[ExtractionResult]) -> tuple[list[str], list[dict[str, Any]]]:
    """One flat row per document. Line items are summarised as a count."""
    rows: list[dict[str, Any]] = []
    data_columns: list[str] = []
    for result in results:
        row: dict[str, Any] = {
            "file": result.file,
            "document_type": result.document_type.value,
            "is_valid": result.is_valid,
            "needs_review": result.needs_review,
            "review_reasons": " | ".join(result.review_reasons),
            "error": result.error or "",
        }
        for key, value in (result.data or {}).items():
            if key == "line_items":
                value = len(value)
                key = "line_item_count"
            elif key == "fields":
                value = "; ".join(f"{kv['key']}={kv['value']}" for kv in value)
            if key not in data_columns:
                data_columns.append(key)
            row[key] = value
        rows.append(row)
    return SUMMARY_COLUMNS + data_columns, rows


def line_item_rows(results: Iterable[ExtractionResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for i, item in enumerate((result.data or {}).get("line_items", []), start=1):
            rows.append({"file": result.file, "line": i, **item})
    return rows


def to_csv(results: list[ExtractionResult]) -> str:
    columns, rows = summary_rows(results)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def to_xlsx(results: list[ExtractionResult]) -> bytes:
    """Workbook with a `documents` sheet and a `line_items` sheet."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "documents"
    columns, rows = summary_rows(results)
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(c) for c in columns])
    items = workbook.create_sheet("line_items")
    items.append(LINE_ITEM_COLUMNS)
    for row in line_item_rows(results):
        items.append([row.get(c) for c in LINE_ITEM_COLUMNS])
    for ws in (sheet, items):
        ws.freeze_panes = "A2"
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def write_results(results: list[ExtractionResult], path: Path, include_ocr: bool = False) -> None:
    """Write results in the format implied by the file extension."""
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".jsonl", ".ndjson"}:
        write_jsonl(results, path, include_ocr)
    elif suffix == ".json":
        payload = [result_to_json(r, include_ocr) for r in results]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix == ".csv":
        path.write_text(to_csv(results), encoding="utf-8", newline="")
    elif suffix == ".xlsx":
        path.write_bytes(to_xlsx(results))
    else:
        raise ValueError(f"unsupported output format {suffix!r} (use .jsonl, .json, .csv, .xlsx)")
