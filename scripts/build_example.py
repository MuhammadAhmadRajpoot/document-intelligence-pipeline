"""Run one real document through the API and save the response as the UI's seeded example.

Usage: python scripts/build_example.py examples/invoice_scanned.jpg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from docintel.api.app import EXAMPLE_FILE, create_app
from docintel.logging_setup import configure_logging


def main(path: Path) -> None:
    configure_logging("INFO")
    client = TestClient(create_app())
    with path.open("rb") as fh:
        response = client.post("/extract", files={"file": (path.name, fh)})
    response.raise_for_status()
    EXAMPLE_FILE.write_text(
        json.dumps({"name": path.name, "payload": response.json()}), encoding="utf-8"
    )
    print(f"wrote {EXAMPLE_FILE} ({EXAMPLE_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
