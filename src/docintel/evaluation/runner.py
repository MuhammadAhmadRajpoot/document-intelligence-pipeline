"""Run the pipeline over a labelled dataset and produce a report."""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docintel.evaluation.metrics import score_document, summarise
from docintel.pipeline import Pipeline

logger = logging.getLogger(__name__)


def run_evaluation(data_dir: Path, pipeline: Pipeline, concurrency: int = 2) -> dict[str, Any]:
    manifest = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    paths = [data_dir / item["file"] for item in manifest]
    started = time.perf_counter()
    results = pipeline.process_many(paths, concurrency=concurrency)
    wall_s = round(time.perf_counter() - started, 1)
    scores = [
        score_document(truth, result) for truth, result in zip(manifest, results, strict=True)
    ]
    report = summarise(scores)
    report["run"] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": pipeline.llm.model,
        "max_repair_attempts": pipeline.settings.max_repair_attempts,
        "concurrency": concurrency,
        "wall_time_s": wall_s,
        "platform": f"{platform.system()} {platform.release()} / {platform.processor()}",
        "python": platform.python_version(),
    }
    return report


def to_markdown(report: dict[str, Any]) -> str:
    """Render the headline tables of a report as Markdown (used in the README)."""

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines = [
        "| Slice | Docs | Field accuracy | Classification | Docs fully correct | "
        "Flagged for review | Mean latency |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = [("All", report["overall"])]
    rows += [(f"{k} (variant)", v) for k, v in report["by_variant"].items()]
    rows += [(f"{k} (type)", v) for k, v in report["by_type"].items()]
    for name, b in rows:
        lines.append(
            f"| {name} | {b['documents']} | {pct(b['field_accuracy'])} | "
            f"{pct(b['classification_accuracy'])} | {pct(b['documents_fully_correct'])} | "
            f"{pct(b['flagged_for_review'])} | {b['mean_latency_s']} s |"
        )
    lines += ["", "| Field | Overall | Digital | Scanned |", "|---|---|---|---|"]
    for name, f in report["by_field"].items():
        lines.append(
            f"| {name} | {pct(f['accuracy'])} | {pct(f.get('digital'))} | {pct(f.get('scanned'))} |"
        )
    return "\n".join(lines)
