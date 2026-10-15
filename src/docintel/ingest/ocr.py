"""OCR backed by RapidOCR (PP-OCR models on ONNX Runtime; CPU-only, no torch)."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from docintel.models import TextLine

logger = logging.getLogger(__name__)


class OCREngine(Protocol):
    def recognize(self, image_rgb: np.ndarray) -> list[TextLine]: ...


class RapidOCREngine:
    """Lazily-initialised, thread-safe wrapper returning text lines with boxes and scores."""

    def __init__(self, min_confidence: float = 0.3) -> None:
        self._min_confidence = min_confidence
        self._engine: object | None = None
        self._lock = threading.Lock()

    def _get_engine(self) -> object:
        if self._engine is None:
            from rapidocr import RapidOCR  # heavy import, deferred

            logger.info("loading OCR models")
            self._engine = RapidOCR(params={"Global.log_level": "warning"})
        return self._engine

    def recognize(self, image_rgb: np.ndarray) -> list[TextLine]:
        # RapidOCR follows the OpenCV convention and expects BGR arrays.
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        with self._lock:
            output = self._get_engine()(bgr)  # type: ignore[operator]
        if output is None or output.boxes is None:
            return []
        lines: list[TextLine] = []
        for box, text, score in zip(output.boxes, output.txts, output.scores, strict=True):
            confidence = float(score)
            if confidence < self._min_confidence or not str(text).strip():
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            lines.append(
                TextLine(
                    text=str(text).strip(),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=round(min(max(confidence, 0.0), 1.0), 4),
                )
            )
        return lines
