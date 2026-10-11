#!/usr/bin/env python3
"""Training entry point. Phase 2: only the dataset-freeze guard exists; no training code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel.entrypoints import train_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(train_main())
