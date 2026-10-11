"""Copy dataset archives from Drive to local storage, verify, and extract safely.

Used by the Colab bootstrap notebook. No dataset is bundled or downloaded by this code. An
expected SHA-256 is required unless the caller explicitly allows an unverified archive, in
which case the computed hash is returned so it can be recorded.
"""
from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

from .hashing import is_sha256_hex, sha256_file


class StagingError(RuntimeError):
    pass


def _safe_target(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    if target != dest.resolve() and dest.resolve() not in target.parents:
        raise StagingError(f"unsafe path in archive: {name!r}")
    return target


def extract_archive(archive: str | Path, dest: str | Path) -> list[str]:
    """Extract a .zip, .tar, .tar.gz or .tgz archive, rejecting path traversal and links."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                _safe_target(dest, info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise StagingError(f"symlink in archive: {info.filename!r}")
            z.extractall(dest)
            names = z.namelist()
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = t.getmembers()
            for m in members:
                _safe_target(dest, m.name)
                if m.issym() or m.islnk() or m.isdev():
                    raise StagingError(f"link or device in archive: {m.name!r}")
            t.extractall(dest, members=members, filter="data") if hasattr(tarfile, "data_filter") \
                else t.extractall(dest, members=members)
            names = [m.name for m in members]
    else:
        raise StagingError(f"unsupported or corrupt archive: {archive}")
    return names


def stage_archive(src: str | Path, local_dir: str | Path, *, expected_sha256: str | None,
                  extract_to: str | Path | None = None, allow_unverified: bool = False) -> dict:
    """Copy ``src`` to ``local_dir``, verify its SHA-256, optionally extract.

    The hash is computed on the LOCAL copy, so a corrupt transfer is detected.
    """
    src, local_dir = Path(src), Path(local_dir)
    if not src.is_file():
        raise StagingError(f"archive not found: {src}")
    if expected_sha256 is None and not allow_unverified:
        raise StagingError("expected_sha256 is required (pass allow_unverified=True to record a new hash)")
    if expected_sha256 is not None and not is_sha256_hex(expected_sha256):
        raise StagingError("expected_sha256 is not a 64-character hex string")
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / src.name
    tmp = local.with_name(local.name + ".part")
    shutil.copyfile(src, tmp)
    actual = sha256_file(tmp)
    if expected_sha256 is not None and actual != expected_sha256.lower():
        tmp.unlink()
        raise StagingError(f"SHA-256 mismatch for {src.name}: expected {expected_sha256.lower()}, got {actual}")
    tmp.replace(local)
    result = {"local_path": str(local), "sha256": actual, "verified": expected_sha256 is not None, "extracted": None}
    if extract_to is not None:
        result["extracted"] = {"dest": str(extract_to), "n_entries": len(extract_archive(local, extract_to))}
    return result
