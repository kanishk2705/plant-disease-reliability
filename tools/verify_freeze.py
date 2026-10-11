#!/usr/bin/env python3
"""Verify a dataset freeze or a run freeze. Exit 0 only if every check passes.

  verify_freeze.py dataset --freeze F --sha256 H --repo-root R [--source-manifest M] [--external-manifest M]
  verify_freeze.py run --freeze F --sha256 H --repo-root R --checkpoint C --source-manifest M \
        --external-manifest M --dataset-freeze D [--dataset-freeze-sha256 H]

Exit codes: 0 verified, 1 verification failed, 2 usage error or unexpected error.
The same hash check can be done by hand: ``sha256sum -c configs/frozen/<id>.sha256``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantrel.freeze import FreezeVerificationError, verify_dataset_freeze, verify_run_freeze  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--freeze", required=True)
    d.add_argument("--sha256", required=True)
    d.add_argument("--repo-root", default=".")
    d.add_argument("--source-manifest")
    d.add_argument("--external-manifest")
    r = sub.add_parser("run")
    r.add_argument("--freeze", required=True)
    r.add_argument("--sha256", required=True)
    r.add_argument("--repo-root", default=".")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--source-manifest", required=True)
    r.add_argument("--external-manifest", required=True)
    r.add_argument("--dataset-freeze", required=True)
    r.add_argument("--dataset-freeze-sha256")
    try:
        a = p.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    try:
        if a.kind == "dataset":
            _, rep = verify_dataset_freeze(a.freeze, expected_sha256=a.sha256, repo_root=a.repo_root,
                                           source_manifest_path=a.source_manifest,
                                           external_manifest_path=a.external_manifest)
        else:
            rep = verify_run_freeze(a.freeze, expected_sha256=a.sha256, repo_root=a.repo_root,
                                    checkpoint_path=a.checkpoint, source_manifest_path=a.source_manifest,
                                    external_manifest_path=a.external_manifest, dataset_freeze_path=a.dataset_freeze,
                                    dataset_freeze_expected_sha256=a.dataset_freeze_sha256)
    except FreezeVerificationError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # fail closed
        print(f"FAIL unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"OK {rep.kind} freeze verified: {', '.join(rep.passed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
