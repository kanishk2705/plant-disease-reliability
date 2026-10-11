"""Dataset-freeze and run-freeze verification (protocol sections 4.8 and 9.4).

Every function fails closed: any problem raises ``FreezeVerificationError``. Nothing here
loads external data or logits.

Roles (protocol 9.4):
* ``code_commit``       -- Git commit of the implementation snapshot; stored INSIDE the run freeze.
* freeze file content   -- ``configs/frozen/<id>.json`` (no hash of itself).
* freeze SHA-256        -- hash of the file's bytes, stored in the sidecar ``<id>.sha256`` and
                           supplied independently on the command line.
* freeze commit         -- the later Git commit that stores the freeze file and sidecar.

Two independent Git protections protect the code (amendment A2.2):
(a) ``git diff --exit-code <code_commit> HEAD -- <code paths>`` (catches changes committed later);
(b) separate staged, unstaged and untracked checks of the working tree.
A clean working tree alone is never accepted as proof that the code is unchanged.
``configs/frozen/`` is excluded from (a) and (b) because freeze metadata is created after
``code_commit``; its integrity is verified by the hash checks.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import canonical_json_bytes, is_sha256_hex, sha256_file

CODE_PATHS = ("src", "scripts", "tools", "configs", "pyproject.toml", "requirements.txt")
CODE_EXCLUDES = (":(exclude)configs/frozen",)
_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class FreezeVerificationError(Exception):
    def __init__(self, check: str, message: str):
        super().__init__(f"[{check}] {message}")
        self.check = check
        self.message = message


@dataclass
class VerificationReport:
    kind: str
    passed: list[str] = field(default_factory=list)

    def ok(self, name: str) -> None:
        self.passed.append(name)


# ------------------------------------------------------------------ writing


def sidecar_path(freeze_path: str | Path) -> Path:
    return Path(freeze_path).with_suffix(".sha256")


def write_freeze_file(path: str | Path, obj: dict) -> str:
    """Write the canonical JSON and its ``sha256sum``-compatible sidecar. Returns the SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(obj))
    digest = sha256_file(path)
    sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


# ------------------------------------------------------------------ hash checks


def verify_file_hash(path: str | Path, expected_sha256: str | None) -> str:
    """File exists, sidecar exists and is well formed, and the actual hash equals both the
    sidecar and the independently supplied ``expected_sha256``."""
    path = Path(path)
    if not path.is_file():
        raise FreezeVerificationError("missing_file", f"freeze file not found: {path}")
    side = sidecar_path(path)
    if not side.is_file():
        raise FreezeVerificationError("missing_sidecar", f"hash sidecar not found: {side}")
    tokens = side.read_text(encoding="utf-8").split()
    if not tokens or not is_sha256_hex(tokens[0]) or len(tokens) > 2 or (len(tokens) == 2 and tokens[1] != path.name):
        raise FreezeVerificationError("malformed_sidecar", f"sidecar is not '<sha256>  {path.name}': {side}")
    actual = sha256_file(path)
    if actual != tokens[0].lower():
        raise FreezeVerificationError("hash_mismatch", f"file hash {actual} differs from sidecar {tokens[0].lower()}")
    if expected_sha256 is None:
        raise FreezeVerificationError("missing_cli_hash", "an independently supplied SHA-256 is required")
    if not is_sha256_hex(expected_sha256):
        raise FreezeVerificationError("invalid_cli_hash", "supplied SHA-256 is not 64 hex characters")
    if actual != expected_sha256.lower():
        raise FreezeVerificationError("cli_hash_mismatch", f"file hash {actual} differs from supplied {expected_sha256.lower()}")
    return actual


def _load_json(path: Path) -> dict:
    def _reject(const):
        raise ValueError(f"non-finite constant {const}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    except (ValueError, UnicodeDecodeError) as exc:
        raise FreezeVerificationError("malformed_json", f"{path.name}: {exc}") from exc
    if not isinstance(obj, dict):
        raise FreezeVerificationError("malformed_json", f"{path.name}: top level must be an object")
    return obj


# ------------------------------------------------------------------ field validation


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _need(obj: dict, key: str, pred, desc: str, what: str) -> None:
    if key not in obj:
        raise FreezeVerificationError("missing_field", f"{what}: missing {key!r}")
    if not pred(obj[key]):
        raise FreezeVerificationError("malformed_field", f"{what}: {key!r} must be {desc}")


def _is_dict(x): return isinstance(x, dict)
def _is_class_list(x): return isinstance(x, list) and len(x) > 0 and all(isinstance(c, str) and c for c in x) and len(set(x)) == len(x)
def _is_nonempty_str(x): return isinstance(x, str) and x != ""


def _eval_lists_ok(lists, class_list) -> bool:
    return (isinstance(lists, dict) and set(lists) == {"source_test", "external"}
            and all(_is_class_list(v) and set(v) <= set(class_list) for v in lists.values()))


def validate_dataset_freeze_fields(obj: dict) -> None:
    w = "dataset freeze"
    _need(obj, "schema_version", lambda x: x == 1, "1", w)
    _need(obj, "protocol_version", lambda x: x == "1.0", "'1.0'", w)
    _need(obj, "class_list", _is_class_list, "a non-empty list of unique strings", w)
    _need(obj, "eval_class_lists", lambda x: _eval_lists_ok(x, obj["class_list"]),
          "an object with 'source_test' and 'external' class lists drawn from class_list", w)
    _need(obj, "class_alignment", lambda x: isinstance(x, list) and len(x) > 0, "a non-empty list", w)
    for key in ("crop_policy", "min_counts", "dedup_settings", "distributions", "evaluation_transform", "licence_findings"):
        _need(obj, key, _is_dict, "an object", w)
    _need(obj, "split_manifest_sha256", is_sha256_hex, "a SHA-256 hex string", w)
    _need(obj, "external_manifest_sha256", is_sha256_hex, "a SHA-256 hex string", w)


def validate_run_freeze_fields(obj: dict) -> None:
    w = "run freeze"
    _need(obj, "schema_version", lambda x: x == 1, "1", w)
    _need(obj, "run_id", lambda x: isinstance(x, str) and _RUN_ID.match(x) is not None, "[A-Za-z0-9._-]+", w)
    _need(obj, "seed", lambda x: isinstance(x, int) and not isinstance(x, bool) and x >= 0, "a non-negative int", w)
    _need(obj, "code_commit", lambda x: isinstance(x, str) and _COMMIT.match(x) is not None, "a 40-character lowercase commit hash", w)
    _need(obj, "T", lambda x: _num(x) and x > 0, "a finite number > 0", w)
    _need(obj, "tau", _is_dict, "an object", w)
    tau = obj["tau"]
    if "r_cov" not in tau or not (_num(tau["r_cov"]) and 0 <= tau["r_cov"] <= 1):
        raise FreezeVerificationError("malformed_field", f"{w}: tau.r_cov must be a number in [0, 1]")
    if "r_risk" not in tau or not (tau["r_risk"] is None or (_num(tau["r_risk"]) and 0 <= tau["r_risk"] <= 1)):
        raise FreezeVerificationError("malformed_field", f"{w}: tau.r_risk must be null or a number in [0, 1]")
    _need(obj, "checkpoint_sha256", is_sha256_hex, "a SHA-256 hex string", w)
    _need(obj, "class_list", _is_class_list, "a non-empty list of unique strings", w)
    _need(obj, "eval_class_lists", lambda x: _eval_lists_ok(x, obj["class_list"]),
          "an object with 'source_test' and 'external' class lists drawn from class_list", w)
    _need(obj, "config_hash", is_sha256_hex, "a SHA-256 hex string", w)
    _need(obj, "manifest_hashes", lambda x: isinstance(x, dict) and set(x) == {"source", "external"}
          and all(is_sha256_hex(v) for v in x.values()), "an object with SHA-256 'source' and 'external'", w)
    _need(obj, "dataset_freeze_sha256", is_sha256_hex, "a SHA-256 hex string", w)
    _need(obj, "environment", _is_dict, "an object", w)


# ------------------------------------------------------------------ Git checks


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FreezeVerificationError("git_error", f"git could not be run: {exc}") from exc


def _pathspec() -> list[str]:
    return ["--", *CODE_PATHS, *CODE_EXCLUDES]


def check_code_unchanged_since_commit(repo: str | Path, code_commit: str) -> None:
    """Protection (a): compare ``code_commit`` against HEAD for the code paths."""
    repo = Path(repo)
    if not _COMMIT.match(code_commit or ""):
        raise FreezeVerificationError("unknown_code_commit", "code_commit is not a 40-character hash")
    r = _git(repo, "rev-parse", "--verify", "--quiet", f"{code_commit}^{{commit}}")
    if r.returncode != 0:
        raise FreezeVerificationError("unknown_code_commit", f"commit {code_commit} does not exist in this repository")
    r = _git(repo, "merge-base", "--is-ancestor", code_commit, "HEAD")
    if r.returncode != 0:
        raise FreezeVerificationError("code_commit_not_ancestor", "code_commit is not an ancestor of HEAD")
    r = _git(repo, "diff", "--quiet", "--exit-code", code_commit, "HEAD", *_pathspec())
    if r.returncode == 1:
        raise FreezeVerificationError("code_changed_since_code_commit",
                                      "tracked code or run configuration differs between code_commit and HEAD")
    if r.returncode != 0:
        raise FreezeVerificationError("git_error", f"git diff failed: {r.stderr.strip()}")


def check_worktree_clean(repo: str | Path) -> None:
    """Protection (b): no unstaged, staged or untracked changes in the code paths."""
    repo = Path(repo)
    r = _git(repo, "diff", "--quiet", "--exit-code", *_pathspec())
    if r.returncode == 1:
        raise FreezeVerificationError("unstaged_code_changes", "unstaged modifications in code paths")
    if r.returncode != 0:
        raise FreezeVerificationError("git_error", f"git diff failed: {r.stderr.strip()}")
    r = _git(repo, "diff", "--cached", "--quiet", "--exit-code", *_pathspec())
    if r.returncode == 1:
        raise FreezeVerificationError("staged_code_changes", "staged modifications in code paths")
    if r.returncode != 0:
        raise FreezeVerificationError("git_error", f"git diff --cached failed: {r.stderr.strip()}")
    r = _git(repo, "ls-files", "--others", "--exclude-standard", *_pathspec())
    if r.returncode != 0:
        raise FreezeVerificationError("git_error", f"git ls-files failed: {r.stderr.strip()}")
    if r.stdout.strip():
        raise FreezeVerificationError("untracked_code_files", f"untracked files in code paths: {r.stdout.split()[:5]}")


def check_tracked_and_unmodified(repo: str | Path, *paths: str | Path) -> None:
    """Each file is tracked and identical to HEAD (staged and unstaged)."""
    repo = Path(repo).resolve()
    for p in paths:
        try:
            rel = Path(p).resolve().relative_to(repo).as_posix()
        except ValueError:
            raise FreezeVerificationError("freeze_outside_repo", f"{p} is not inside the repository") from None
        if _git(repo, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
            raise FreezeVerificationError("freeze_not_tracked", f"{rel} is not tracked by Git")
        r = _git(repo, "diff", "--quiet", "--exit-code", "HEAD", "--", rel)
        if r.returncode != 0:
            raise FreezeVerificationError("freeze_modified", f"{rel} differs from HEAD")


# ------------------------------------------------------------------ orchestration


def verify_dataset_freeze(path: str | Path, *, expected_sha256: str | None, repo_root: str | Path | None = None,
                          source_manifest_path: str | Path | None = None,
                          external_manifest_path: str | Path | None = None,
                          require_source_manifest: bool = False) -> tuple[dict, VerificationReport]:
    """Verify the dataset freeze. ``require_source_manifest`` is used by the training guard."""
    rep = VerificationReport("dataset")
    path = Path(path)
    verify_file_hash(path, expected_sha256)
    rep.ok("file_sidecar_cli_hash")
    obj = _load_json(path)
    validate_dataset_freeze_fields(obj)
    rep.ok("fields")
    if repo_root is not None:
        check_tracked_and_unmodified(repo_root, path, sidecar_path(path))
        rep.ok("tracked_and_unmodified")
    if require_source_manifest and source_manifest_path is None:
        raise FreezeVerificationError("missing_manifest", "source manifest path is required")
    if source_manifest_path is not None:
        if not Path(source_manifest_path).is_file() or sha256_file(source_manifest_path) != obj["split_manifest_sha256"]:
            raise FreezeVerificationError("source_manifest_mismatch", "source manifest missing or differs from the dataset freeze")
        rep.ok("source_manifest")
    if external_manifest_path is not None:
        if not Path(external_manifest_path).is_file() or sha256_file(external_manifest_path) != obj["external_manifest_sha256"]:
            raise FreezeVerificationError("external_manifest_mismatch", "external manifest missing or differs from the dataset freeze")
        rep.ok("external_manifest")
    return obj, rep


def require_dataset_freeze(path: str | Path, *, expected_sha256: str | None, repo_root: str | Path,
                           source_manifest_path: str | Path | None) -> VerificationReport:
    """Training guard: does NOT touch the external manifest."""
    _, rep = verify_dataset_freeze(path, expected_sha256=expected_sha256, repo_root=repo_root,
                                   source_manifest_path=source_manifest_path, require_source_manifest=True)
    return rep


def verify_run_freeze(path: str | Path, *, expected_sha256: str | None, repo_root: str | Path,
                      checkpoint_path: str | Path | None, source_manifest_path: str | Path | None,
                      external_manifest_path: str | Path | None, dataset_freeze_path: str | Path,
                      dataset_freeze_expected_sha256: str | None = None) -> VerificationReport:
    """Full run-freeze verification, to be called before any external data or logits are loaded."""
    rep = VerificationReport("run")
    path, repo_root = Path(path), Path(repo_root)
    verify_file_hash(path, expected_sha256)
    rep.ok("file_sidecar_cli_hash")
    obj = _load_json(path)
    validate_run_freeze_fields(obj)
    rep.ok("fields")
    check_tracked_and_unmodified(repo_root, path, sidecar_path(path))
    rep.ok("tracked_and_unmodified")
    check_code_unchanged_since_commit(repo_root, obj["code_commit"])
    rep.ok("code_unchanged_since_code_commit")
    check_worktree_clean(repo_root)
    rep.ok("worktree_clean")
    # dataset freeze: its own hash chain, plus the hash recorded in the run freeze
    ds_path = Path(dataset_freeze_path)
    if not ds_path.is_file():
        raise FreezeVerificationError("missing_file", f"dataset freeze not found: {ds_path}")
    ds_hash = sha256_file(ds_path)
    if ds_hash != obj["dataset_freeze_sha256"]:
        raise FreezeVerificationError("dataset_freeze_hash_mismatch", "dataset freeze hash differs from the one recorded in the run freeze")
    ds_obj, _ = verify_dataset_freeze(ds_path, expected_sha256=dataset_freeze_expected_sha256 or ds_hash, repo_root=repo_root)
    rep.ok("dataset_freeze")
    # class lists must agree with the dataset freeze
    if obj["class_list"] != ds_obj["class_list"] or obj["eval_class_lists"] != ds_obj["eval_class_lists"]:
        raise FreezeVerificationError("class_list_mismatch", "class lists differ between run freeze and dataset freeze")
    rep.ok("class_lists")
    # manifests
    mh = obj["manifest_hashes"]
    if mh["source"] != ds_obj["split_manifest_sha256"] or mh["external"] != ds_obj["external_manifest_sha256"]:
        raise FreezeVerificationError("manifest_hash_mismatch", "manifest hashes differ between run freeze and dataset freeze")
    for label, p, want in (("source", source_manifest_path, mh["source"]), ("external", external_manifest_path, mh["external"])):
        if p is None or not Path(p).is_file():
            raise FreezeVerificationError("missing_manifest", f"{label} manifest path missing or not a file")
        if sha256_file(p) != want:
            raise FreezeVerificationError(f"{label}_manifest_mismatch", f"{label} manifest differs from the freeze")
    rep.ok("manifests")
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        raise FreezeVerificationError("missing_checkpoint", "checkpoint path missing or not a file")
    if sha256_file(checkpoint_path) != obj["checkpoint_sha256"]:
        raise FreezeVerificationError("checkpoint_mismatch", "checkpoint differs from the freeze")
    rep.ok("checkpoint")
    return rep
