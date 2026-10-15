"""Data models that flow through the pipeline and make up the public result."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from docintel.schemas import DocumentType

BBox = tuple[float, float, float, float]
"""Axis-aligned box (x0, y0, x1, y1) in page pixel coordinates of the rendered page image."""


class TextSource(StrEnum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class TextLine(BaseModel):
    text: str
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)


class Page(BaseModel):
    """One page of a document after ingestion. `lines` carry geometry for grounding."""

    index: int
    width: int
    height: int
    source: TextSource
    lines: list[TextLine] = Field(default_factory=list)
    preprocessing: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Reading-order text that keeps table rows on one line (helps the LLM read tables)."""
        rows: list[list[TextLine]] = []
        for line in sorted(self.lines, key=lambda ln: ((ln.bbox[1] + ln.bbox[3]) / 2, ln.bbox[0])):
            centre = (line.bbox[1] + line.bbox[3]) / 2
            height = max(line.bbox[3] - line.bbox[1], 1.0)
            if rows:
                last = rows[-1][-1]
                if abs(centre - (last.bbox[1] + last.bbox[3]) / 2) <= 0.5 * height:
                    rows[-1].append(line)
                    continue
            rows.append([line])
        return "\n".join(
            "   ".join(ln.text for ln in sorted(row, key=lambda ln: ln.bbox[0])) for row in rows
        )

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: Severity
    fields: list[str] = Field(default_factory=list)


class FieldEvidence(BaseModel):
    """Where (and how confidently) a field value was found in the document."""

    field: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    page: int | None = None
    bbox: BBox | None = None
    grounded: bool = False


class ExtractionAttempt(BaseModel):
    attempt: int
    issues: list[ValidationIssue] = Field(default_factory=list)
    schema_error: str | None = None


class ExtractionResult(BaseModel):
    """Final, serialisable output for one document."""

    file: str
    document_type: DocumentType
    classification_confidence: float
    data: dict[str, Any] | None = None
    field_confidence: dict[str, FieldEvidence] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)
    is_valid: bool = False
    needs_review: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    attempts: list[ExtractionAttempt] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    model: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
