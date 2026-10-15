"""OpenCV pre-processing for scanned pages: upscale, denoise and deskew."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_MIN_OCR_WIDTH = 1400
_MAX_SKEW_DEG = 8.0


@dataclass
class Preprocessed:
    """`ocr_image` is fed to OCR; `display_image` is what users see (same geometry / `scale`)."""

    ocr_image: np.ndarray
    display_image: np.ndarray
    scale: float
    steps: list[str] = field(default_factory=list)


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate text skew in degrees with a projection-profile search (robust to borders)."""
    small_w = 800
    factor = small_w / gray.shape[1] if gray.shape[1] > small_w else 1.0
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    centre = (w / 2, h / 2)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = max(np.arange(-_MAX_SKEW_DEG, _MAX_SKEW_DEG + 0.01, 1.0), key=score)
    fine = max(np.arange(coarse - 1.0, coarse + 1.01, 0.1), key=score)
    return round(float(fine), 2)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def noise_level(gray: np.ndarray) -> float:
    """Rough noise estimate: mean absolute difference from a median-filtered copy."""
    return float(np.mean(cv2.absdiff(gray, cv2.medianBlur(gray, 3))))


def preprocess(image_rgb: np.ndarray) -> Preprocessed:
    """Clean a page image for OCR. Steps are only applied when they are likely to help."""
    steps: list[str] = []
    display = image_rgb
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    angle = estimate_skew(gray)
    if abs(angle) >= 0.3:
        display = rotate(display, angle)
        gray = rotate(gray, angle)
        steps.append(f"deskew({angle:+.1f}deg)")

    scale = 1.0
    if gray.shape[1] < _MIN_OCR_WIDTH:
        scale = _MIN_OCR_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        steps.append(f"upscale(x{scale:.2f})")

    if noise_level(gray) > 4.0:
        gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        steps.append("denoise")

    return Preprocessed(
        ocr_image=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        display_image=display,
        scale=scale,
        steps=steps,
    )
