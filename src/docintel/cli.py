"""Command-line interface: `docintel extract | serve | generate-dataset | eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from docintel.config import get_settings
from docintel.ingest.loader import SUPPORTED_SUFFIXES
from docintel.logging_setup import configure_logging
from docintel.models import ExtractionResult
from docintel.pipeline import ConfigurationError, Pipeline
from docintel.schemas import DocumentType

app = typer.Typer(add_completion=False, help="Turn business documents into validated JSON.")
console = Console(stderr=True)


def _collect(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files += sorted(p for p in item.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        elif item.is_file():
            files.append(item)
        else:
            raise typer.BadParameter(f"no such file or directory: {item}")
    return files


def _pipeline() -> Pipeline:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return Pipeline(settings)
    except ConfigurationError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _summary_table(results: list[ExtractionResult]) -> Table:
    table = Table(title="Extraction summary")
    for col in ("file", "type", "valid", "review", "total", "notes"):
        table.add_column(col)
    for r in results:
        total = (r.data or {}).get("total")
        notes = r.error or "; ".join(r.review_reasons[:2])
        table.add_row(
            Path(r.file).name,
            r.document_type.value,
            "[green]yes[/green]" if r.is_valid else "[red]no[/red]",
            "[yellow]yes[/yellow]" if r.needs_review else "no",
            "" if total is None else f"{total:,.2f}",
            notes[:70],
        )
    return table


@app.command()
def extract(
    inputs: Annotated[list[Path], typer.Argument(help="Files and/or directories to process.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output file: .jsonl, .json, .csv or .xlsx")
    ] = Path("results.jsonl"),
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1, max=16)] = 4,
    doc_type: Annotated[
        DocumentType | None,
        typer.Option("--type", help="Skip classification and force a document type."),
    ] = None,
    include_ocr: Annotated[
        bool, typer.Option(help="Include OCR lines/boxes in JSON output.")
    ] = False,
) -> None:
    """Extract structured data from documents (batch, concurrent)."""
    from docintel.export import write_results

    files = _collect(inputs)
    if not files:
        console.print("[yellow]no supported documents found[/yellow]")
        raise typer.Exit(1)
    pipeline = _pipeline()
    with Progress(console=console) as progress:
        task = progress.add_task("extracting", total=len(files))
        results = pipeline.process_many(
            files,
            concurrency=concurrency,
            doc_type=doc_type,
            on_done=lambda _: progress.advance(task),
        )
    write_results(results, out, include_ocr=include_ocr)
    console.print(_summary_table(results))
    failed = sum(r.error is not None for r in results)
    console.print(f"wrote {len(results)} result(s) to {out} ({failed} failed)")
    if failed == len(results):
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for the API and web UI.")] = 8104,
) -> None:
    """Run the FastAPI server with the web UI."""
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    uvicorn.run("docintel.api.app:create_app", factory=True, host=host, port=port)


@app.command("generate-dataset")
def generate_dataset_cmd(
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("data/synthetic"),
    seed: int = 7,
) -> None:
    """Generate the synthetic invoice/receipt evaluation set with ground truth."""
    from docintel.evaluation.synthetic import generate_dataset

    manifest = generate_dataset(out, seed=seed)
    console.print(f"wrote {len(manifest)} documents + ground_truth.json to {out}")


@app.command("eval")
def evaluate(
    data: Annotated[Path, typer.Option("--data", "-d")] = Path("data/synthetic"),
    report: Annotated[Path, typer.Option("--report", "-r")] = Path("reports/eval_report.json"),
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1, max=16)] = 2,
) -> None:
    """Score the pipeline against ground truth (field-level exact match)."""
    from docintel.evaluation.runner import run_evaluation, to_markdown

    if not (data / "ground_truth.json").exists():
        console.print(f"[red]{data}/ground_truth.json not found; run generate-dataset[/red]")
        raise typer.Exit(2)
    result = run_evaluation(data, _pipeline(), concurrency=concurrency)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = to_markdown(result)
    report.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
    typer.echo(markdown)
    console.print(f"report written to {report}")


if __name__ == "__main__":
    app()
