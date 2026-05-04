"""Print persisted regression metrics in terminal-friendly format."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics file not found. Run `python scripts/train_model.py` first.")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print("House Prices Metrics")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
