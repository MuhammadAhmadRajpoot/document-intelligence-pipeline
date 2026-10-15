import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from docintel.evaluation.synthetic import _image_pdf
from docintel.ingest import UnsupportedFileError, load_document
from docintel.ingest.preprocess import estimate_skew, preprocess, rotate
from docintel.models import TextSource
from tests.conftest import FakeOCR


def text_pdf(text: str = "INVOICE INV-1 Total $10.00 thank you for your business") -> bytes:
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        return doc.tobytes()


def test_pdf_text_layer_skips_ocr():
    ocr = FakeOCR()
    loaded = load_document("a.pdf", text_pdf(), ocr)
    assert loaded.pages[0].source is TextSource.TEXT_LAYER
    assert ocr.calls == 0
    assert "INV-1" in loaded.text
    assert loaded.pages[0].lines[0].confidence == 1.0
    assert loaded.images[0].shape[2] == 3


def test_scanned_pdf_falls_back_to_ocr():
    image = Image.new("RGB", (400, 500), "white")
    ocr = FakeOCR(["Total 10.00"])
    loaded = load_document("scan.pdf", _image_pdf(image), ocr)
    assert loaded.pages[0].source is TextSource.OCR
    assert ocr.calls == 1
    assert loaded.text.endswith("Total 10.00")


def test_image_input_uses_ocr(invoice_png_bytes):
    ocr = FakeOCR(["INVOICE"])
    loaded = load_document("doc.png", invoice_png_bytes, ocr)
    assert ocr.calls == 1
    assert loaded.pages[0].source is TextSource.OCR


def test_ocr_boxes_are_mapped_back_to_display_scale(invoice_png_bytes):
    # The 500px-wide input is upscaled for OCR; boxes must come back in display coordinates.
    loaded = load_document("doc.png", invoice_png_bytes, FakeOCR(["INVOICE"]))
    page = loaded.pages[0]
    assert any(step.startswith("upscale") for step in page.preprocessing)
    assert page.width == loaded.images[0].shape[1]
    assert page.lines[0].bbox[2] < 400 / 2


def test_unsupported_and_corrupt_inputs():
    with pytest.raises(UnsupportedFileError):
        load_document("notes.txt", b"hello", FakeOCR())
    with pytest.raises(UnsupportedFileError):
        load_document("broken.png", b"not an image", FakeOCR())
    with pytest.raises(UnsupportedFileError):
        load_document("broken.pdf", b"%PDF-garbage", FakeOCR())


def test_deskew_estimates_rotation():
    page = np.full((900, 700), 255, dtype=np.uint8)
    for y in range(80, 820, 40):
        cv2.putText(
            page, "Lorem ipsum dolor sit amet 123.45", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2
        )
    skewed = rotate(page, 3.0)
    assert abs(estimate_skew(skewed) + 3.0) < 0.6
    result = preprocess(cv2.cvtColor(skewed, cv2.COLOR_GRAY2RGB))
    assert any(step.startswith("deskew") for step in result.steps)


def test_denoise_runs_only_on_noisy_images():
    clean = np.full((400, 1500, 3), 255, dtype=np.uint8)
    assert "denoise" not in preprocess(clean).steps
    rng = np.random.default_rng(0)
    noisy = np.clip(clean + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8)
    assert "denoise" in preprocess(noisy).steps
