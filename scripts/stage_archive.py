#!/usr/bin/env python3
"""Copy an archive from Drive to local disk, verify its SHA-256, and extract it there."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel.staging import StagingError, stage_archive  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--local-dir", required=True)
    p.add_argument("--sha256", help="expected SHA-256 (required unless --allow-unverified)")
    p.add_argument("--extract-to")
    p.add_argument("--allow-unverified", action="store_true")
    a = p.parse_args()
    try:
        res = stage_archive(a.src, a.local_dir, expected_sha256=a.sha256, extract_to=a.extract_to,
                            allow_unverified=a.allow_unverified)
    except StagingError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
