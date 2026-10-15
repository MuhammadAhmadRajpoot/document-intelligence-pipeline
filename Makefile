# Task shortcuts. On Windows without make, run the commands shown after each target.
PY ?= python

.PHONY: install dev lint format test integration dataset eval serve example docker

install:      ## pip install -r requirements.txt && pip install -e . --no-deps
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

dev:          ## pip install -r requirements-dev.txt && pip install -e . --no-deps
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:         ## offline unit tests (LLM and OCR are faked)
	$(PY) -m pytest

integration:  ## real Groq calls, needs GROQ_API_KEY
	$(PY) -m pytest -m integration

dataset:
	docintel generate-dataset --out data/synthetic

eval: dataset
	docintel eval --data data/synthetic --report reports/eval_report.json

serve:
	docintel serve --port 8104

example:      ## refresh the seeded example shown in the web UI
	$(PY) scripts/build_example.py examples/invoice_scanned.jpg

docker:
	docker compose up --build
