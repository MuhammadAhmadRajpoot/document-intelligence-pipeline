"""Time ingestion only (text layer / pre-processing / OCR, no LLM) for a labelled dataset.

Usage: python scripts/benchmark_ingest.py data/synthetic
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np

from docintel.config import get_settings
from docintel.ingest import load_document
from docintel.ingest.ocr import RapidOCREngine


def main(data_dir: Path) -> None:
    settings = get_settings()
    manifest = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    ocr = RapidOCREngine(settings.ocr_min_confidence)
    ocr.recognize(np.full((32, 32, 3), 255, np.uint8))  # load models before timing
    timings: dict[str, list[float]] = {}
    confidences: list[float] = []
    for item in manifest:
        data = (data_dir / item["file"]).read_bytes()
        started = time.perf_counter()
        loaded = load_document(item["file"], data, ocr, dpi=settings.pdf_render_dpi)
        timings.setdefault(item["variant"], []).append(time.perf_counter() - started)
        if item["variant"] == "scanned":
            confidences.append(loaded.pages[0].mean_confidence)
    for variant, values in timings.items():
        print(
            f"{variant:8s} n={len(values):2d} mean={mean(values):6.2f}s "
            f"median={median(values):6.2f}s max={max(values):6.2f}s"
        )
    if confidences:
        print(f"scanned mean OCR line confidence: {mean(confidences):.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"))
