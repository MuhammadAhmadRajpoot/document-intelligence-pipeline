"""Business-rule validation for extracted documents.

Errors mean the data is internally inconsistent (and trigger a repair attempt);
warnings mean something is missing or unusual and should be looked at by a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from docintel.models import Severity, ValidationIssue
from docintel.normalize import amounts_close
from docintel.schemas import DocumentSchema, GenericDocument, Invoice, LineItem, Receipt

_EARLIEST_DATE = date(1990, 1, 1)


@dataclass(frozen=True)
class Tolerance:
    absolute: float = 0.02
    relative: float = 0.005

    def close(self, a: float, b: float) -> bool:
        return amounts_close(a, b, self.absolute, self.relative)


def _error(code: str, message: str, *fields: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity=Severity.ERROR, fields=list(fields))


def _warning(code: str, message: str, *fields: str) -> ValidationIssue:
    return ValidationIssue(
        code=code, message=message, severity=Severity.WARNING, fields=list(fields)
    )


def _check_line_items(items: list[LineItem], tol: Tolerance) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for i, item in enumerate(items):
        if item.amount is None:
            issues.append(
                _warning("line_item_amount_missing", f"line {i + 1} has no amount", "line_items")
            )
            continue
        if item.quantity is not None and item.unit_price is not None:
            expected = item.quantity * item.unit_price
            if not tol.close(expected, item.amount):
                issues.append(
                    _error(
                        "line_item_math",
                        f"line {i + 1} '{item.description}': quantity {item.quantity:g} x "
                        f"unit price {item.unit_price:.2f} = {expected:.2f}, "
                        f"but amount is {item.amount:.2f}",
                        "line_items",
                    )
                )
    return issues


def _check_totals(
    items: list[LineItem],
    subtotal: float | None,
    tax: float | None,
    extra: float | None,
    total: float | None,
    tol: Tolerance,
    extra_name: str = "tip",
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if total is None:
        return [_error("total_missing", "total amount is missing", "total")]
    if total < 0:
        issues.append(_warning("total_negative", "total is negative (credit note?)", "total"))

    amounts = [item.amount for item in items if item.amount is not None]
    if amounts and subtotal is not None:
        items_sum = round(sum(amounts), 2)
        if not tol.close(items_sum, subtotal):
            issues.append(
                _error(
                    "subtotal_mismatch",
                    f"line items sum to {items_sum:.2f} but subtotal is {subtotal:.2f}",
                    "line_items",
                    "subtotal",
                )
            )

    base = subtotal if subtotal is not None else (round(sum(amounts), 2) if amounts else None)
    if base is None:
        issues.append(
            _warning("subtotal_missing", "no subtotal or line items to check total", "subtotal")
        )
        return issues
    expected = base + (tax or 0.0) + (extra or 0.0)
    if not tol.close(expected, total):
        parts = f"{base:.2f} + tax {tax or 0.0:.2f}"
        if extra:
            parts += f" + {extra_name} {extra:.2f}"
        issues.append(
            _error(
                "total_mismatch",
                f"{parts} = {expected:.2f}, but total is {total:.2f}",
                "subtotal",
                "tax",
                "total",
            )
        )
    return issues


def _check_date(value: date | None, name: str, today: date) -> list[ValidationIssue]:
    if value is None:
        return [_warning(f"{name}_missing", f"{name} is missing", name)]
    if not _EARLIEST_DATE <= value <= today + timedelta(days=366):
        return [
            _warning(f"{name}_implausible", f"{name} {value.isoformat()} looks implausible", name)
        ]
    return []


def validate_document(
    doc: DocumentSchema, tol: Tolerance | None = None, today: date | None = None
) -> list[ValidationIssue]:
    """Run all business rules that apply to the document's schema."""
    tol = tol or Tolerance()
    today = today or date.today()
    issues: list[ValidationIssue] = []

    if isinstance(doc, Invoice):
        if not doc.vendor_name:
            issues.append(_warning("vendor_missing", "vendor name is missing", "vendor_name"))
        if not doc.invoice_number:
            issues.append(
                _warning("invoice_number_missing", "invoice number is missing", "invoice_number")
            )
        issues += _check_date(doc.invoice_date, "invoice_date", today)
        if doc.invoice_date and doc.due_date and doc.due_date < doc.invoice_date:
            issues.append(
                _error(
                    "due_before_issue",
                    "due date is before the invoice date",
                    "invoice_date",
                    "due_date",
                )
            )
        issues += _check_line_items(doc.line_items, tol)
        issues += _check_totals(doc.line_items, doc.subtotal, doc.tax, None, doc.total, tol)
    elif isinstance(doc, Receipt):
        if not doc.merchant_name:
            issues.append(_warning("merchant_missing", "merchant name is missing", "merchant_name"))
        issues += _check_date(doc.transaction_date, "transaction_date", today)
        issues += _check_line_items(doc.line_items, tol)
        issues += _check_totals(doc.line_items, doc.subtotal, doc.tax, doc.tip, doc.total, tol)
    elif isinstance(doc, GenericDocument) and not doc.fields:
        issues.append(_warning("no_fields", "no key/value fields were extracted", "fields"))

    if isinstance(doc, Invoice | Receipt) and doc.currency is None:
        issues.append(_warning("currency_missing", "currency could not be determined", "currency"))
    return issues
