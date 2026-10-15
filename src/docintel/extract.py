"""Schema-constrained LLM extraction with a validate-and-repair loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from docintel.llm.client import ChatMessage, JSONLLMClient, LLMError
from docintel.llm.prompts import extraction_messages, repair_messages
from docintel.models import ExtractionAttempt, Severity, ValidationIssue
from docintel.normalize import detect_currency
from docintel.schemas import SCHEMA_FOR_TYPE, DocumentSchema, DocumentType, Invoice, Receipt

logger = logging.getLogger(__name__)

Validator = Callable[[DocumentSchema], list[ValidationIssue]]


@dataclass
class ExtractionOutcome:
    document: DocumentSchema | None
    issues: list[ValidationIssue] = field(default_factory=list)
    attempts: list[ExtractionAttempt] = field(default_factory=list)


def _summarise(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:8]
    )


def _infer_currency(doc: DocumentSchema, text: str) -> list[ValidationIssue]:
    if isinstance(doc, Invoice | Receipt) and doc.currency is None:
        detected = detect_currency(text)
        if detected:
            doc.currency = detected
            return [
                ValidationIssue(
                    code="currency_inferred",
                    severity=Severity.WARNING,
                    message=f"currency {detected} inferred from symbols in text",
                    fields=["currency"],
                )
            ]
    return []


def _n_errors(issues: list[ValidationIssue]) -> int:
    return sum(issue.severity is Severity.ERROR for issue in issues)


def extract_document(
    llm: JSONLLMClient,
    doc_type: DocumentType,
    text: str,
    validate: Validator,
    max_repair_attempts: int = 2,
) -> ExtractionOutcome:
    """Extract, validate, and ask the model to repair until clean or out of attempts.

    The best schema-valid attempt (fewest rule errors, latest on ties) is returned.
    """
    schema = SCHEMA_FOR_TYPE[doc_type]
    messages: list[ChatMessage] = extraction_messages(doc_type, schema, text)
    outcome = ExtractionOutcome(document=None)
    best_errors: int | None = None

    for attempt in range(max_repair_attempts + 1):
        try:
            raw: dict[str, Any] = llm.complete_json(
                messages,
                schema_name=schema.__name__.lower(),
                json_schema=schema.model_json_schema(),
            )
        except LLMError as exc:
            logger.warning("extraction call failed", extra={"attempt": attempt, "error": str(exc)})
            outcome.attempts.append(ExtractionAttempt(attempt=attempt, schema_error=str(exc)))
            continue

        try:
            doc = schema.model_validate(raw)
        except ValidationError as exc:
            summary = _summarise(exc)
            outcome.attempts.append(ExtractionAttempt(attempt=attempt, schema_error=summary))
            logger.info("schema validation failed", extra={"attempt": attempt, "error": summary})
            messages = repair_messages(messages, raw, summary, [])
            continue

        issues = _infer_currency(doc, text) + validate(doc)
        outcome.attempts.append(ExtractionAttempt(attempt=attempt, issues=issues))
        errors = _n_errors(issues)
        if best_errors is None or errors <= best_errors:
            outcome.document, outcome.issues, best_errors = doc, issues, errors
        if errors == 0:
            break
        logger.info(
            "business rules failed, requesting repair", extra={"attempt": attempt, "errors": errors}
        )
        messages = repair_messages(
            messages, raw, None, [i for i in issues if i.severity is Severity.ERROR]
        )
    return outcome
