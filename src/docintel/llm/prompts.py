"""Prompt templates and a compact schema outline for the extraction model."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from docintel.llm.client import ChatMessage
from docintel.models import ValidationIssue
from docintel.schemas import DocumentType

CLASSIFY_SYSTEM = """You classify business documents from their OCR/text content.
Return a JSON object: {"document_type": "invoice" | "receipt" | "other",
"confidence": number between 0 and 1, "reason": short string}.
- invoice: a bill requesting payment (invoice number, bill-to, due date, payment terms).
- receipt: proof of a completed purchase (store/merchant, items paid, payment method, change).
- other: anything else (letters, contracts, statements, forms)."""

EXTRACT_SYSTEM = """You extract structured data from a {doc_type} for an accounting system.
Return ONE JSON object that matches this outline exactly (no extra keys):
{outline}

Rules:
- Copy values exactly as printed. Never guess or invent values; use null when absent.
- Dates must be ISO format YYYY-MM-DD. Resolve ambiguous day/month order using other
  dates in the document and the vendor's country when possible.
- Amounts are plain numbers (no currency symbols, no thousands separators).
- currency is an ISO 4217 code (USD, EUR, GBP, PKR, ...), inferred from symbols if needed.
- line_items: one entry per product/service row; amount is the line total as printed.
- tax is the total tax amount (sum of all tax lines), not a rate.
- The text comes from OCR and may contain recognition errors; use layout and arithmetic
  to read numbers correctly."""

REPAIR_TEMPLATE = """Your previous JSON failed validation:
{problems}

Re-read the document text and return the corrected full JSON object.
If the document itself really prints these values (for example its totals do not add up),
keep the printed values instead of inventing numbers to make the checks pass."""


def schema_outline(model: type[BaseModel]) -> dict[str, Any]:
    """Turn a Pydantic model into a small, readable JSON outline for the prompt."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def describe(prop: dict[str, Any]) -> Any:
        if "$ref" in prop:
            return outline(defs[prop["$ref"].split("/")[-1]])
        variants = prop.get("anyOf", [prop])
        nullable = any(v.get("type") == "null" for v in variants)
        concrete = next((v for v in variants if v.get("type") != "null"), {})
        if concrete.get("type") == "array":
            return [describe(concrete.get("items", {}))]
        if "$ref" in concrete:
            return outline(defs[concrete["$ref"].split("/")[-1]])
        kind = {"number": "number", "integer": "number", "string": "string"}.get(
            concrete.get("type", ""), "string"
        )
        if concrete.get("format") == "date":
            kind = "YYYY-MM-DD"
        if "enum" in concrete:
            kind = " | ".join(concrete["enum"])
        return f"{kind} | null" if nullable else kind

    def outline(node: dict[str, Any]) -> dict[str, Any]:
        return {name: describe(prop) for name, prop in node.get("properties", {}).items()}

    return outline(schema)


def classify_messages(text: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": f"Document text:\n\n{text}"},
    ]


def extraction_messages(
    doc_type: DocumentType, schema: type[BaseModel], text: str
) -> list[ChatMessage]:
    label = "business document" if doc_type is DocumentType.OTHER else doc_type.value
    system = EXTRACT_SYSTEM.format(
        doc_type=label, outline=json.dumps(schema_outline(schema), indent=1)
    )
    if doc_type is DocumentType.OTHER:
        system += "\n- fields: the important key/value pairs printed on the document."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Document text:\n\n{text}"},
    ]


def repair_messages(
    previous: list[ChatMessage],
    raw_output: dict[str, Any],
    schema_error: str | None,
    issues: list[ValidationIssue],
) -> list[ChatMessage]:
    problems = [f"- schema: {schema_error}"] if schema_error else []
    problems += [f"- {issue.code}: {issue.message}" for issue in issues]
    return [
        *previous[:2],
        {"role": "assistant", "content": json.dumps(raw_output, default=str)},
        {"role": "user", "content": REPAIR_TEMPLATE.format(problems="\n".join(problems))},
    ]
