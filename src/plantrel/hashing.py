"""Hashing and canonical-JSON helpers shared by checkpoints and freeze files."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.match(value) is not None


def canonical_json_bytes(obj: Any) -> bytes:
    """Bytes written for a freeze file: UTF-8, sorted keys, 2-space indent, LF, final newline.

    NaN and infinity are rejected so the file is valid JSON everywhere.
    """
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def config_hash(config: dict) -> str:
    """SHA-256 of a compact canonical JSON encoding of a configuration mapping.

    Non-JSON-serialisable values raise TypeError instead of being silently stringified,
    so two different configs cannot collide through str() conversion.
    """
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return sha256_bytes(text.encode("utf-8"))
