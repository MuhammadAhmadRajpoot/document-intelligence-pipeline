"""FastAPI application: `/extract`, `/export/{fmt}`, `/health` and the single-page UI."""

from __future__ import annotations

import base64
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docintel import __version__
from docintel.config import Settings, get_settings
from docintel.export import result_to_json, to_csv, to_xlsx
from docintel.ingest import UnsupportedFileError
from docintel.llm.client import LLMError
from docintel.models import ExtractionResult, Page
from docintel.pipeline import ConfigurationError, Pipeline
from docintel.schemas import DocumentType

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
EXAMPLE_FILE = WEB_DIR / "example.json"
PREVIEW_MAX_WIDTH = 1100


class PageView(BaseModel):
    index: int
    width: int
    height: int
    source: str
    preprocessing: list[str]
    mean_confidence: float
    text: str
    preview: str | None = None


class ExtractResponse(BaseModel):
    result: dict[str, Any]
    pages: list[PageView]


def _preview(image_rgb: np.ndarray) -> str:
    """Downscaled JPEG data URL. Field boxes are scaled client-side using page width."""
    h, w = image_rgb.shape[:2]
    if w > PREVIEW_MAX_WIDTH:
        scale = PREVIEW_MAX_WIDTH / w
        image_rgb = cv2.resize(
            image_rgb, (PREVIEW_MAX_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82]
    )
    if not ok:
        raise RuntimeError("could not encode preview")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _page_view(page: Page, image: np.ndarray | None) -> PageView:
    return PageView(
        index=page.index,
        width=page.width,
        height=page.height,
        source=page.source.value,
        preprocessing=page.preprocessing,
        mean_confidence=round(page.mean_confidence, 3),
        text=page.text,
        preview=_preview(image) if image is not None else None,
    )


def create_app(pipeline: Pipeline | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass a `pipeline` to inject fakes in tests."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Document Intelligence Pipeline",
        version=__version__,
        description="Invoices, receipts and other business documents to validated JSON.",
    )

    @lru_cache(maxsize=1)
    def get_pipeline() -> Pipeline:
        return pipeline if pipeline is not None else Pipeline(settings)

    @app.exception_handler(ConfigurationError)
    async def _config_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "model": settings.model}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(
        file: Annotated[UploadFile, File(description="PDF, PNG or JPG document")],
        doc_type: Annotated[
            DocumentType | None, Query(description="Force a document type instead of classifying")
        ] = None,
        previews: Annotated[bool, Query(description="Include page preview images")] = True,
    ) -> ExtractResponse:
        data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(400, "empty upload")
        name = file.filename or "upload"
        try:
            result, loaded = get_pipeline().process(name, data, doc_type)
        except UnsupportedFileError as exc:
            raise HTTPException(415, str(exc)) from exc
        except LLMError as exc:
            logger.error("llm failure", extra={"file": name, "error": str(exc)})
            raise HTTPException(502, f"LLM error: {exc}") from exc
        pages = [
            _page_view(page, loaded.images[i] if previews else None)
            for i, page in enumerate(loaded.pages)
        ]
        return ExtractResponse(result=result_to_json(result), pages=pages)

    @app.post("/export/{fmt}")
    def export(
        fmt: Literal["csv", "xlsx"],
        results: Annotated[list[dict[str, Any]], Body(description="Results from /extract")],
    ) -> Response:
        parsed = [ExtractionResult.model_validate(r) for r in results]
        if fmt == "csv":
            return Response(
                to_csv(parsed),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="docintel_results.csv"'},
            )
        return Response(
            to_xlsx(parsed),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="docintel_results.xlsx"'},
        )

    @app.get("/example", include_in_schema=False)
    def example() -> JSONResponse:
        if not EXAMPLE_FILE.exists():
            raise HTTPException(404, "no example available")
        return JSONResponse(json.loads(EXAMPLE_FILE.read_text(encoding="utf-8")))

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
