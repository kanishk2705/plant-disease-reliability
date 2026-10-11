#!/usr/bin/env python3
"""External evaluation entry point. Phase 2: only the run-freeze guard exists.

The run freeze is verified BEFORE any external manifest, image or logit file is opened.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel.entrypoints import eval_external_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(eval_external_main())
