"""Command-line entry points for training and external evaluation (guards only in Phase 2).

Exit codes: 0 success; 1 verification failed (fail closed); 2 usage error; 3 not implemented.
Neither entry point loads anything unless verification has passed. The model-training and
external-evaluation bodies do not exist yet (Phase 5 and later); they are injected so tests can
prove the ordering without a model or dataset.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

from .freeze import FreezeVerificationError, require_dataset_freeze, verify_run_freeze

EXIT_OK, EXIT_VERIFY, EXIT_USAGE, EXIT_NOT_IMPLEMENTED = 0, 1, 2, 3


def _parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--dataset-freeze", required=True)
    p.add_argument("--dataset-freeze-sha256", required=True)
    p.add_argument("--source-manifest", required=True)
    return p


def train_main(argv: list[str] | None = None, trainer: Callable[[argparse.Namespace], int] | None = None) -> int:
    """Refuse to run unless the dataset freeze verifies."""
    p = _parser("train")
    p.add_argument("--config", required=True)
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    try:
        require_dataset_freeze(args.dataset_freeze, expected_sha256=args.dataset_freeze_sha256,
                               repo_root=args.repo_root, source_manifest_path=args.source_manifest)
    except FreezeVerificationError as exc:
        print(f"REFUSED: dataset freeze not valid: {exc}", file=sys.stderr)
        return EXIT_VERIFY
    except Exception as exc:  # fail closed on anything unexpected
        print(f"REFUSED: unexpected {type(exc).__name__} during dataset-freeze check: {exc}", file=sys.stderr)
        return EXIT_VERIFY
    if trainer is None:
        print("Dataset freeze verified. Training is not implemented in Phase 2.", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    return trainer(args)


def eval_external_main(argv: list[str] | None = None,
                       loader: Callable[[argparse.Namespace], int] | None = None) -> int:
    """Independently verify the run freeze BEFORE any external data or logits are loaded."""
    p = _parser("eval_external")
    p.add_argument("--run-freeze", required=True)
    p.add_argument("--run-freeze-sha256", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--external-manifest", required=True)
    p.add_argument("--external-logits")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    try:
        verify_run_freeze(args.run_freeze, expected_sha256=args.run_freeze_sha256, repo_root=args.repo_root,
                          checkpoint_path=args.checkpoint, source_manifest_path=args.source_manifest,
                          external_manifest_path=args.external_manifest, dataset_freeze_path=args.dataset_freeze,
                          dataset_freeze_expected_sha256=args.dataset_freeze_sha256)
    except FreezeVerificationError as exc:
        print(f"REFUSED: run freeze not valid: {exc}", file=sys.stderr)
        return EXIT_VERIFY
    except Exception as exc:
        print(f"REFUSED: unexpected {type(exc).__name__} during run-freeze check: {exc}", file=sys.stderr)
        return EXIT_VERIFY
    if loader is None:
        print("Run freeze verified. External evaluation is not implemented in Phase 2.", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    return loader(args)
