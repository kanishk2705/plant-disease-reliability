"""Build a throw-away Git repository with synthetic dataset and run freezes. No real data or model."""
from __future__ import annotations

import copy
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from plantrel.freeze import sidecar_path, verify_run_freeze, write_freeze_file
from plantrel.hashing import sha256_file

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=GIT_ENV)
    assert r.returncode == 0, f"git {args} failed: {r.stderr}"
    return r.stdout.strip()


def sha(c: str) -> str:
    return c * 64


DATASET_FREEZE = {
    "schema_version": 1, "protocol_version": "1.0",
    "class_list": ["c0", "c1", "c2"],
    "eval_class_lists": {"source_test": ["c0", "c1", "c2"], "external": ["c0", "c1", "c2"]},
    "class_alignment": [{"canonical": "c0"}], "crop_policy": {"min_px": 0}, "min_counts": {"external": 0},
    "dedup_settings": {"phash": 0}, "distributions": {"synthetic": True}, "evaluation_transform": {"size": 224},
    "licence_findings": {"status": "synthetic"},
}


@dataclass
class World:
    repo: Path
    art: Path
    code_commit: str
    ds_path: Path
    run_path: Path
    ds_obj: dict
    run_obj: dict
    ckpt: Path
    src_manifest: Path
    ext_manifest: Path
    extra: dict = field(default_factory=dict)

    def run_hash(self) -> str:
        return sha256_file(self.run_path)

    def ds_hash(self) -> str:
        return sha256_file(self.ds_path)

    def verify(self, **over):
        kw = dict(repo_root=self.repo, checkpoint_path=self.ckpt, source_manifest_path=self.src_manifest,
                  external_manifest_path=self.ext_manifest, dataset_freeze_path=self.ds_path)
        kw.update(over)
        if "expected_sha256" not in kw:
            kw["expected_sha256"] = self.run_hash()
        return verify_run_freeze(self.run_path, **kw)

    def commit_all(self, msg="update"):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", msg)
        return git(self.repo, "rev-parse", "HEAD")

    def rewrite_run_freeze(self, mutate=None, commit=True):
        obj = copy.deepcopy(self.run_obj)
        if mutate:
            mutate(obj)
        write_freeze_file(self.run_path, obj)
        if commit:
            self.commit_all("rewrite run freeze")
        return obj

    def rewrite_dataset_freeze(self, mutate=None, commit=True):
        obj = copy.deepcopy(self.ds_obj)
        if mutate:
            mutate(obj)
        write_freeze_file(self.ds_path, obj)
        if commit:
            self.commit_all("rewrite dataset freeze")
        return obj


def make_world(tmp_path: Path) -> World:
    repo, art = tmp_path / "repo", tmp_path / "artefacts"
    for d in ("src", "scripts", "tools", "configs/frozen"):
        (repo / d).mkdir(parents=True)
    art.mkdir()
    (repo / "src" / "mod.py").write_text("X = 1\n")
    (repo / "scripts" / "run.py").write_text("print('run')\n")
    (repo / "tools" / "t.py").write_text("print('tool')\n")
    (repo / "configs" / "exp.yaml").write_text("lr: 0.1\n")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "requirements.txt").write_text("numpy\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "implementation snapshot")
    code_commit = git(repo, "rev-parse", "HEAD")

    ckpt, src_m, ext_m = art / "best.pt", art / "source_manifest.csv", art / "external_manifest.csv"
    ckpt.write_bytes(b"synthetic checkpoint bytes")
    src_m.write_text("id,label\n1,c0\n")
    ext_m.write_text("id,label\n9,c1\n")

    ds_obj = {**DATASET_FREEZE, "split_manifest_sha256": sha256_file(src_m), "external_manifest_sha256": sha256_file(ext_m)}
    ds_path = repo / "configs" / "frozen" / "dataset_decisions_v1.json"
    write_freeze_file(ds_path, ds_obj)
    run_obj = {
        "schema_version": 1, "run_id": "synthetic-run-1", "seed": 1, "code_commit": code_commit, "T": 1.7,
        "tau": {"r_cov": 0.81, "r_risk": None}, "checkpoint_sha256": sha256_file(ckpt),
        "class_list": ds_obj["class_list"], "eval_class_lists": ds_obj["eval_class_lists"], "config_hash": sha("a"),
        "manifest_hashes": {"source": sha256_file(src_m), "external": sha256_file(ext_m)},
        "dataset_freeze_sha256": sha256_file(ds_path), "environment": {"python": "3.x"},
    }
    run_path = repo / "configs" / "frozen" / "synthetic-run-1.json"
    write_freeze_file(run_path, run_obj)
    w = World(repo, art, code_commit, ds_path, run_path, ds_obj, run_obj, ckpt, src_m, ext_m)
    w.commit_all("freeze files")
    return w


def flip_one_byte(path: Path, offset: int = 20) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


__all__ = ["World", "make_world", "git", "flip_one_byte", "sha", "sidecar_path", "json"]
