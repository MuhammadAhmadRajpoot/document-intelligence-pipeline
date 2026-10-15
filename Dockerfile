FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgl/libglib are needed by OpenCV; libgomp by ONNX Runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . \
    # Download the OCR models at build time so the container starts offline.
    && python -c "from docintel.ingest.ocr import RapidOCREngine; import numpy as np; RapidOCREngine().recognize(np.full((64, 256, 3), 255, np.uint8))"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8104
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8104/health')"
CMD ["docintel", "serve", "--host", "0.0.0.0", "--port", "8104"]
