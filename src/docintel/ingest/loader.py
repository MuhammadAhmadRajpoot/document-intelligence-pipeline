"""Load PDFs and images into `Page` objects plus RGB page images for previews."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from docintel.ingest.ocr import OCREngine
from docintel.ingest.preprocess import preprocess
from docintel.models import Page, TextLine, TextSource

logger = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES


class UnsupportedFileError(ValueError):
    """Raised for unknown file types or files that cannot be decoded."""


@dataclass
class LoadedDocument:
    name: str
    pages: list[Page] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"--- page {p.index + 1} ---\n{p.text}" for p in self.pages)


def _ocr_page(
    index: int, image_rgb: np.ndarray, ocr: OCREngine, do_preprocess: bool
) -> tuple[Page, np.ndarray]:
    if do_preprocess:
        prepared = preprocess(image_rgb)
        ocr_image, display, scale, steps = (
            prepared.ocr_image,
            prepared.display_image,
            prepared.scale,
            prepared.steps,
        )
    else:
        ocr_image, display, scale, steps = image_rgb, image_rgb, 1.0, []
    lines = [
        line.model_copy(update={"bbox": tuple(v / scale for v in line.bbox)})
        for line in ocr.recognize(ocr_image)
    ]
    h, w = display.shape[:2]
    page = Page(
        index=index, width=w, height=h, source=TextSource.OCR, lines=lines, preprocessing=steps
    )
    return page, display


def _text_layer_lines(page: pymupdf.Page, zoom: float) -> list[TextLine]:
    lines: dict[tuple[int, int], list[tuple[float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words", sort=True):
        lines.setdefault((block, line), []).append((x0, y0, x1, y1, word))
    out: list[TextLine] = []
    for words in lines.values():
        out.append(
            TextLine(
                text=" ".join(w[4] for w in words),
                bbox=(
                    min(w[0] for w in words) * zoom,
                    min(w[1] for w in words) * zoom,
                    max(w[2] for w in words) * zoom,
                    max(w[3] for w in words) * zoom,
                ),
                confidence=1.0,
            )
        )
    return out


def _load_pdf(
    data: bytes, ocr: OCREngine, dpi: int, min_chars: int, max_pages: int, do_preprocess: bool
) -> tuple[list[Page], list[np.ndarray]]:
    pages: list[Page] = []
    images: list[np.ndarray] = []
    zoom = dpi / 72.0
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise UnsupportedFileError(f"cannot open PDF: {exc}") from exc
    with doc:
        for index, pdf_page in enumerate(doc):
            if index >= max_pages:
                logger.warning("page limit reached", extra={"max_pages": max_pages})
                break
            pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            text_chars = len(pdf_page.get_text("text").strip())
            if text_chars >= min_chars:
                lines = _text_layer_lines(pdf_page, zoom)
                pages.append(
                    Page(
                        index=index,
                        width=pix.width,
                        height=pix.height,
                        source=TextSource.TEXT_LAYER,
                        lines=lines,
                    )
                )
                images.append(image.copy())
            else:
                logger.info("no text layer, running OCR", extra={"page": index})
                page, display = _ocr_page(index, image.copy(), ocr, do_preprocess)
                pages.append(page)
                images.append(display)
    return pages, images


def _decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise UnsupportedFileError("cannot decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_document(
    name: str,
    data: bytes,
    ocr: OCREngine,
    *,
    dpi: int = 200,
    min_text_layer_chars: int = 25,
    max_pages: int = 10,
    do_preprocess: bool = True,
) -> LoadedDocument:
    """Turn raw bytes into pages. PDFs use the text layer when present, else OCR."""
    suffix = Path(name).suffix.lower()
    if suffix in PDF_SUFFIXES:
        pages, images = _load_pdf(data, ocr, dpi, min_text_layer_chars, max_pages, do_preprocess)
    elif suffix in IMAGE_SUFFIXES:
        page, display = _ocr_page(0, _decode_image(data), ocr, do_preprocess)
        pages, images = [page], [display]
    else:
        raise UnsupportedFileError(
            f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    if not pages:
        raise UnsupportedFileError("document has no pages")
    return LoadedDocument(name=name, pages=pages, images=images)
