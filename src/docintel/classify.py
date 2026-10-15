"""Document-type classification: LLM first, keyword heuristic as a fallback."""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from docintel.llm.client import JSONLLMClient, LLMError
from docintel.llm.prompts import classify_messages
from docintel.schemas import Classification, DocumentType

logger = logging.getLogger(__name__)

_KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.INVOICE: (
        r"\binvoice\b",
        r"\bbill to\b",
        r"\bdue date\b",
        r"\bpayment terms\b",
        r"\binv[-#]",
        r"\bremit\b",
        r"\bnet \d+\b",
        r"\bpurchase order\b",
    ),
    DocumentType.RECEIPT: (
        r"\breceipt\b",
        r"\bcashier\b",
        r"\bchange\b",
        r"\bthank you\b",
        r"\bvisa\b",
        r"\bmastercard\b",
        r"\bcash\b",
        r"\btip\b",
        r"\bstore\b",
        r"\bregister\b",
    ),
}


def heuristic_classify(text: str) -> Classification:
    """Cheap keyword scoring. Used offline and whenever the LLM call fails."""
    lowered = text.lower()
    scores = {
        doc_type: sum(len(re.findall(pattern, lowered)) for pattern in patterns)
        for doc_type, patterns in _KEYWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    if scores[best] == 0:
        return Classification(
            document_type=DocumentType.OTHER,
            confidence=0.5,
            reason="no invoice/receipt keywords found",
        )
    return Classification(
        document_type=best,
        confidence=round(scores[best] / (total + 1), 2),
        reason=f"keyword scores { ({k.value: v for k, v in scores.items()}) }",
    )


def classify_document(text: str, llm: JSONLLMClient | None) -> Classification:
    if llm is not None:
        try:
            raw = llm.complete_json(
                classify_messages(text),
                schema_name="classification",
                json_schema=Classification.model_json_schema(),
            )
            return Classification.model_validate(raw)
        except (LLMError, ValidationError) as exc:
            logger.warning(
                "LLM classification failed, using heuristic", extra={"error": str(exc)[:200]}
            )
    return heuristic_classify(text)
