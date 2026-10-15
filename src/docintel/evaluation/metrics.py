"""Field-level exact-match scoring against ground truth."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from docintel.models import ExtractionResult

FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "invoice": (
        "vendor_name",
        "customer_name",
        "invoice_number",
        "invoice_date",
        "due_date",
        "currency",
        "subtotal",
        "tax",
        "total",
        "line_items",
    ),
    "receipt": (
        "merchant_name",
        "receipt_number",
        "transaction_date",
        "currency",
        "subtotal",
        "tax",
        "tip",
        "total",
        "line_items",
    ),
    "other": (),
}
AMOUNT_TOLERANCE = 0.01


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().strip(".,;:").lower()


def field_matches(name: str, truth: Any, predicted: Any) -> bool:
    """Exact match after light normalisation (case/whitespace; amounts to the cent)."""
    if name == "line_items":
        truth_amounts = sorted(round(item["amount"], 2) for item in truth or [])
        pred_amounts = sorted(round(item.get("amount") or 0.0, 2) for item in predicted or [])
        return len(truth_amounts) == len(pred_amounts) and all(
            abs(a - b) <= AMOUNT_TOLERANCE for a, b in zip(truth_amounts, pred_amounts, strict=True)
        )
    if truth is None or predicted is None:
        return truth is None and predicted is None
    if isinstance(truth, int | float):
        try:
            return abs(float(truth) - float(predicted)) <= AMOUNT_TOLERANCE
        except (TypeError, ValueError):
            return False
    return _norm_text(truth) == _norm_text(predicted)


@dataclass
class DocScore:
    file: str
    variant: str
    true_type: str
    predicted_type: str
    field_hits: dict[str, bool] = field(default_factory=dict)
    needs_review: bool = True
    is_valid: bool = False
    attempts: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def type_correct(self) -> bool:
        return self.true_type == self.predicted_type

    @property
    def all_fields_correct(self) -> bool:
        return self.type_correct and all(self.field_hits.values())


def score_document(truth: dict[str, Any], result: ExtractionResult) -> DocScore:
    true_type = truth["document_type"]
    score = DocScore(
        file=truth["file"],
        variant=truth["variant"],
        true_type=true_type,
        predicted_type=result.document_type.value,
        needs_review=result.needs_review,
        is_valid=result.is_valid,
        attempts=len(result.attempts),
        latency_ms=result.timings_ms.get("total_ms", 0.0),
        error=result.error,
    )
    data = result.data if score.type_correct and result.data else {}
    for name in FIELDS_BY_TYPE[true_type]:
        score.field_hits[name] = bool(data) and field_matches(name, truth.get(name), data.get(name))
    return score


def _accuracy(hits: list[bool]) -> float | None:
    return round(sum(hits) / len(hits), 4) if hits else None


def summarise(scores: list[DocScore]) -> dict[str, Any]:
    """Aggregate per-field, per-variant and per-type accuracy plus review-flag quality."""

    def block(subset: list[DocScore]) -> dict[str, Any]:
        hits = [hit for s in subset for hit in s.field_hits.values()]
        wrong = [s for s in subset if s.field_hits and not s.all_fields_correct]
        right = [s for s in subset if s.field_hits and s.all_fields_correct]
        return {
            "documents": len(subset),
            "field_accuracy": _accuracy(hits),
            "fields_scored": len(hits),
            "classification_accuracy": _accuracy([s.type_correct for s in subset]),
            "documents_fully_correct": _accuracy(
                [s.all_fields_correct for s in subset if s.field_hits]
            ),
            "flagged_for_review": _accuracy([s.needs_review for s in subset]),
            "review_recall_on_wrong_docs": _accuracy([s.needs_review for s in wrong]),
            "false_review_rate_on_correct_docs": _accuracy([s.needs_review for s in right]),
            "mean_latency_s": round(mean(s.latency_ms for s in subset) / 1000, 2)
            if subset
            else None,
            "mean_llm_extraction_attempts": round(mean(s.attempts for s in subset), 2)
            if subset
            else None,
        }

    per_field: dict[str, list[bool]] = defaultdict(list)
    per_field_variant: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        for name, hit in s.field_hits.items():
            per_field[name].append(hit)
            per_field_variant[name][s.variant].append(hit)

    variants = sorted({s.variant for s in scores})
    types = sorted({s.true_type for s in scores})
    return {
        "overall": block(scores),
        "by_variant": {v: block([s for s in scores if s.variant == v]) for v in variants},
        "by_type": {t: block([s for s in scores if s.true_type == t]) for t in types},
        "by_field": {
            name: {
                "accuracy": _accuracy(hits),
                "n": len(hits),
                **{v: _accuracy(per_field_variant[name][v]) for v in variants},
            }
            for name, hits in sorted(per_field.items())
        },
        "documents": [
            {
                "file": s.file,
                "variant": s.variant,
                "true_type": s.true_type,
                "predicted_type": s.predicted_type,
                "needs_review": s.needs_review,
                "is_valid": s.is_valid,
                "attempts": s.attempts,
                "wrong_fields": [k for k, v in s.field_hits.items() if not v],
                "latency_s": round(s.latency_ms / 1000, 2),
                "error": s.error,
            }
            for s in scores
        ],
    }
