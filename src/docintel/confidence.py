"""Per-field confidence by grounding extracted values in the document text.

A value that literally appears on the page inherits the confidence of the text
line it was found in (1.0 for a PDF text layer, the OCR score otherwise). Values
the model produced but that cannot be located get a low prior, and fields involved
in a failed business rule are penalised.
"""

from __future__ import annotations

import re
from contextlib import suppress
from datetime import date
from difflib import SequenceMatcher
from typing import Any

from docintel.models import FieldEvidence, Page, Severity, TextLine, ValidationIssue
from docintel.normalize import CURRENCY_ALIASES, CURRENCY_SYMBOLS, canonical_text
from docintel.schemas import DocumentSchema, GenericDocument, Invoice, Receipt

UNGROUNDED_CONFIDENCE = 0.45
RULE_FAILURE_PENALTY = 0.5
_FUZZY_MIN_RATIO = 0.85
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

KEY_FIELDS: dict[type, tuple[str, ...]] = {
    Invoice: ("vendor_name", "invoice_number", "invoice_date", "total", "currency"),
    Receipt: ("merchant_name", "transaction_date", "total", "currency"),
    GenericDocument: (),
}


def _amount_variants(value: float) -> list[str]:
    us = f"{value:,.2f}"
    eu = us.replace(",", "_").replace(".", ",").replace("_", ".")
    return list(dict.fromkeys([us, f"{value:.2f}", eu, f"{value:.2f}".replace(".", ",")]))


def _date_variants(value: date) -> list[str]:
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d-%b-%Y",
        "%b %d %Y",
    ]
    variants = {value.strftime(fmt).lower() for fmt in formats}
    # Unpadded day forms such as "March 5, 2025" / "5 Mar 2025".
    variants |= {v.replace(f"0{value.day} ", f"{value.day} ") for v in variants}
    variants |= {v.replace(f" 0{value.day},", f" {value.day},") for v in variants}
    return sorted(variants, key=len, reverse=True)


def _variants(name: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float | int):
        return _amount_variants(float(value))
    if isinstance(value, date):
        return _date_variants(value)
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        with suppress(ValueError):
            return [value, *_date_variants(date.fromisoformat(value))]
    if name == "currency":
        symbols = [s for s, code in CURRENCY_SYMBOLS.items() if code == value]
        aliases = [a.lower() for a in CURRENCY_ALIASES.get(str(value), ())]
        return [str(value).lower(), *symbols, *aliases]
    return [canonical_text(value)]


# Label words that usually sit on the same row as a field's value; used to pick the
# right occurrence when a value (e.g. "12.00") appears several times on the page.
_LABEL_HINTS: dict[str, tuple[str, ...]] = {
    "total": ("total", "amount due", "balance due"),
    "subtotal": ("subtotal", "sub-total", "sub total", "net"),
    "tax": ("tax", "vat", "gst"),
    "tip": ("tip", "gratuity"),
    "invoice_date": ("date",),
    "due_date": ("due",),
    "transaction_date": ("date",),
    "invoice_number": ("invoice", "inv"),
    "receipt_number": ("receipt", "trans", "#"),
}


def _row_label_text(page: Page, line: TextLine, variant: str) -> str:
    """Text of the matched line (minus the value) plus its neighbours on the same row."""
    centre = (line.bbox[1] + line.bbox[3]) / 2
    height = max(line.bbox[3] - line.bbox[1], 1.0)
    parts = [canonical_text(line.text).replace(variant, " ")]
    parts += [
        other.text.lower()
        for other in page.lines
        if other is not line and abs((other.bbox[1] + other.bbox[3]) / 2 - centre) <= 0.5 * height
    ]
    return " ".join(parts)


def _match_quality(variant: str, haystack: str) -> float:
    if variant.isalpha() and len(variant) <= 4:
        # Short codes ("usd", "rs") must be whole words, not parts of "hours" or "partners".
        return 1.0 if re.search(rf"\b{re.escape(variant)}\b", haystack) else 0.0
    if variant in haystack:
        return 1.0
    if len(variant) < 6:
        return 0.0
    # Share of the value covered by matching runs of 3+ characters: tolerant to a few
    # OCR substitutions ("Analytlcs") without rewarding scattered single-letter hits.
    blocks = SequenceMatcher(None, variant, haystack, autojunk=False).get_matching_blocks()
    quality = sum(b.size for b in blocks if b.size >= 3) / len(variant)
    return quality if quality >= _FUZZY_MIN_RATIO else 0.0


def _best_match(
    name: str, variants: list[str], pages: list[Page]
) -> tuple[float, Page, TextLine] | None:
    """Return (match quality, page, line) for the best-ranked match across all pages."""
    hints = _LABEL_HINTS.get(name, ())
    best: tuple[float, float, Page, TextLine] | None = None
    for page in pages:
        for line in page.lines:
            haystack = canonical_text(line.text)
            scored = [(_match_quality(v, haystack), v) for v in variants if v]
            quality, variant = max(scored, default=(0.0, ""))
            if quality == 0.0:
                continue
            rank = quality + 0.01 * line.confidence
            if hints and any(h in _row_label_text(page, line, variant) for h in hints):
                rank += 0.1
            if best is None or rank > best[0]:
                best = (rank, quality, page, line)
    return None if best is None else (best[1], best[2], best[3])


def _evidence(name: str, value: Any, pages: list[Page]) -> FieldEvidence:
    match = _best_match(name, _variants(name, value), pages)
    if match is None:
        return FieldEvidence(field=name, value=value, confidence=UNGROUNDED_CONFIDENCE)
    quality, page, line = match
    return FieldEvidence(
        field=name,
        value=value,
        confidence=round(quality * line.confidence, 3),
        page=page.index,
        bbox=line.bbox,
        grounded=True,
    )


def score_fields(
    doc: DocumentSchema, pages: list[Page], issues: list[ValidationIssue]
) -> dict[str, FieldEvidence]:
    """Compute a `FieldEvidence` for every non-empty top-level field."""
    failing = {f for issue in issues if issue.severity is Severity.ERROR for f in issue.fields}
    evidence: dict[str, FieldEvidence] = {}
    if isinstance(doc, GenericDocument):
        for kv in doc.fields:
            if kv.value:
                evidence[kv.key] = _evidence(kv.key, kv.value, pages)
        return evidence

    for name in type(doc).model_fields:
        value = getattr(doc, name)
        if value is None or value == []:
            continue
        if name == "line_items":
            item_scores = [
                _evidence("description", item.description, pages).confidence for item in value
            ]
            ev = FieldEvidence(
                field=name,
                value=len(value),
                confidence=round(sum(item_scores) / len(item_scores), 3),
                grounded=all(s > UNGROUNDED_CONFIDENCE for s in item_scores),
            )
        else:
            ev = _evidence(name, value, pages)
        if name in failing:
            ev.confidence = round(ev.confidence * RULE_FAILURE_PENALTY, 3)
        evidence[name] = ev
    return evidence


def review_reasons(
    doc: DocumentSchema,
    evidence: dict[str, FieldEvidence],
    issues: list[ValidationIssue],
    classification_confidence: float,
    threshold: float,
) -> list[str]:
    """Human-readable reasons why a document should be checked by a person."""
    reasons = [f"rule failed: {i.message}" for i in issues if i.severity is Severity.ERROR]
    if classification_confidence < threshold:
        reasons.append(f"uncertain document type ({classification_confidence:.2f})")
    for name in KEY_FIELDS.get(type(doc), ()):
        ev = evidence.get(name)
        if ev is None:
            reasons.append(f"missing key field: {name}")
        elif ev.confidence < threshold:
            reasons.append(f"low confidence: {name} ({ev.confidence:.2f})")
    if isinstance(doc, GenericDocument) and not doc.fields:
        reasons.append("no fields extracted")
    return reasons
