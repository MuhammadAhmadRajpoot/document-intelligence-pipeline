"""End-to-end orchestration: ingest -> classify -> extract/repair -> validate -> score."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docintel.classify import classify_document
from docintel.confidence import review_reasons, score_fields
from docintel.config import Settings
from docintel.extract import extract_document
from docintel.ingest import LoadedDocument, UnsupportedFileError, load_document
from docintel.ingest.ocr import OCREngine, RapidOCREngine
from docintel.llm.client import JSONLLMClient, LLMError, OpenAICompatibleClient
from docintel.models import ExtractionResult, Severity, ValidationIssue
from docintel.schemas import DocumentSchema, DocumentType
from docintel.validation import Tolerance, validate_document

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when required configuration (such as the API key) is missing."""


def build_llm(settings: Settings) -> OpenAICompatibleClient:
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        raise ConfigurationError("GROQ_API_KEY is not set (see .env.example)")
    return OpenAICompatibleClient(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.model,
        base_url=settings.llm_base_url,
        temperature=settings.temperature,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        response_format=settings.response_format,
    )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Pipeline:
    """Reusable, thread-safe document pipeline (OCR models and HTTP client are shared)."""

    def __init__(
        self,
        settings: Settings,
        llm: JSONLLMClient | None = None,
        ocr: OCREngine | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm if llm is not None else build_llm(settings)
        self.ocr = ocr if ocr is not None else RapidOCREngine(settings.ocr_min_confidence)
        self._tolerance = Tolerance(settings.amount_abs_tolerance, settings.amount_rel_tolerance)

    def _validate(self, doc: DocumentSchema) -> list[ValidationIssue]:
        return validate_document(doc, self._tolerance)

    def load(self, name: str, data: bytes) -> LoadedDocument:
        s = self.settings
        return load_document(
            name,
            data,
            self.ocr,
            dpi=s.pdf_render_dpi,
            min_text_layer_chars=s.min_text_layer_chars,
            max_pages=s.max_pages,
            do_preprocess=s.preprocess,
        )

    def process(
        self, name: str, data: bytes, doc_type: DocumentType | None = None
    ) -> tuple[ExtractionResult, LoadedDocument]:
        """Run the full pipeline on raw bytes. Raises `UnsupportedFileError` for bad input."""
        started = time.perf_counter()
        timings: dict[str, float] = {}

        t = time.perf_counter()
        loaded = self.load(name, data)
        timings["ingest_ms"] = _ms(t)
        text = loaded.text[: self.settings.max_prompt_chars]

        t = time.perf_counter()
        if doc_type is None:
            classification = classify_document(text, self.llm)
            doc_type, class_conf = classification.document_type, classification.confidence
        else:
            class_conf = 1.0
        timings["classify_ms"] = _ms(t)

        t = time.perf_counter()
        outcome = extract_document(
            self.llm, doc_type, text, self._validate, self.settings.max_repair_attempts
        )
        timings["extract_ms"] = _ms(t)

        result = ExtractionResult(
            file=name,
            document_type=doc_type,
            classification_confidence=class_conf,
            attempts=outcome.attempts,
            pages=loaded.pages,
            model=self.llm.model,
            timings_ms=timings,
        )
        if outcome.document is None:
            result.error = "extraction failed: no schema-valid output after all attempts"
            result.review_reasons = [result.error]
        else:
            doc = outcome.document
            evidence = score_fields(doc, loaded.pages, outcome.issues)
            reasons = review_reasons(
                doc, evidence, outcome.issues, class_conf, self.settings.review_confidence_threshold
            )
            result.data = doc.model_dump(mode="json")
            result.field_confidence = evidence
            result.issues = outcome.issues
            result.is_valid = not any(i.severity is Severity.ERROR for i in outcome.issues)
            result.review_reasons = reasons
            result.needs_review = bool(reasons)
        result.timings_ms["total_ms"] = _ms(started)
        logger.info(
            "document processed",
            extra={
                "file": name,
                "type": doc_type.value,
                "valid": result.is_valid,
                "review": result.needs_review,
                "attempts": len(outcome.attempts),
                "ms": result.timings_ms["total_ms"],
            },
        )
        return result, loaded

    def process_path(self, path: Path, doc_type: DocumentType | None = None) -> ExtractionResult:
        """Like `process`, but never raises: failures are reported in `result.error`."""
        try:
            result, _ = self.process(path.name, path.read_bytes(), doc_type)
            result.file = str(path)
            return result
        except (UnsupportedFileError, LLMError, OSError) as exc:
            logger.error("document failed", extra={"file": str(path), "error": str(exc)})
            return ExtractionResult(
                file=str(path),
                document_type=doc_type or DocumentType.OTHER,
                classification_confidence=0.0,
                error=str(exc),
                review_reasons=[str(exc)],
            )

    def process_many(
        self,
        paths: list[Path],
        concurrency: int = 4,
        doc_type: DocumentType | None = None,
        on_done: Callable[[ExtractionResult], None] | None = None,
    ) -> list[ExtractionResult]:
        """Process documents concurrently, preserving input order."""

        def run(path: Path) -> ExtractionResult:
            result = self.process_path(path, doc_type)
            if on_done is not None:
                on_done(result)
            return result

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            return list(pool.map(run, paths))
