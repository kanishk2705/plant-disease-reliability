"""Training must refuse without a valid dataset freeze; external evaluation must verify the run
freeze before loading anything external."""
import subprocess
import sys
from pathlib import Path

import pytest

from freeze_helpers import flip_one_byte, make_world
from plantrel import entrypoints as ep

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def w(tmp_path):
    return make_world(tmp_path)


def train_args(w, **over):
    a = {"--repo-root": str(w.repo), "--dataset-freeze": str(w.ds_path), "--dataset-freeze-sha256": w.ds_hash(),
         "--source-manifest": str(w.src_manifest), "--config": "x.yaml"}
    a.update(over)
    return [t for kv in a.items() for t in kv]


def eval_args(w, **over):
    a = {"--repo-root": str(w.repo), "--dataset-freeze": str(w.ds_path), "--dataset-freeze-sha256": w.ds_hash(),
         "--source-manifest": str(w.src_manifest), "--run-freeze": str(w.run_path), "--run-freeze-sha256": w.run_hash(),
         "--checkpoint": str(w.ckpt), "--external-manifest": str(w.ext_manifest), "--external-logits": "unused.npy"}
    a.update(over)
    return [t for kv in a.items() for t in kv]


def test_train_runs_trainer_only_after_valid_dataset_freeze(w):
    calls = []
    assert ep.train_main(train_args(w), trainer=lambda a: calls.append(a) or 0) == 0
    assert len(calls) == 1


def test_train_default_is_honest_about_not_being_implemented(w):
    assert ep.train_main(train_args(w)) == ep.EXIT_NOT_IMPLEMENTED


@pytest.mark.parametrize("breakage", ["missing", "byte", "bad_cli", "no_sidecar", "manifest", "untracked"])
def test_train_refuses_without_valid_dataset_freeze(w, breakage):
    calls = []
    args = train_args(w)
    if breakage == "missing":
        w.ds_path.unlink()
    elif breakage == "byte":
        flip_one_byte(w.ds_path)
    elif breakage == "bad_cli":
        args = train_args(w, **{"--dataset-freeze-sha256": "0" * 64})
    elif breakage == "no_sidecar":
        w.ds_path.with_suffix(".sha256").unlink()
    elif breakage == "manifest":
        w.src_manifest.write_text("tampered\n")
    elif breakage == "untracked":
        from freeze_helpers import git
        git(w.repo, "rm", "-q", "--cached", "configs/frozen/dataset_decisions_v1.json")
        git(w.repo, "commit", "-q", "-m", "x")
    assert ep.train_main(args, trainer=lambda a: calls.append(a) or 0) == ep.EXIT_VERIFY
    assert calls == []


def test_train_usage_error_is_nonzero_and_does_not_train():
    calls = []
    assert ep.train_main(["--config", "x"], trainer=lambda a: calls.append(a) or 0) == ep.EXIT_USAGE
    assert calls == []


def test_eval_loads_external_data_only_after_run_freeze_verifies(w):
    loaded = []
    assert ep.eval_external_main(eval_args(w), loader=lambda a: loaded.append(a.external_logits) or 0) == 0
    assert loaded == ["unused.npy"]


@pytest.mark.parametrize("breakage", ["missing", "byte", "bad_cli", "code_changed", "ckpt", "ext_manifest", "dataset_freeze"])
def test_eval_never_calls_the_loader_when_verification_fails(w, breakage):
    loaded = []
    args = eval_args(w)
    if breakage == "missing":
        w.run_path.unlink()
    elif breakage == "byte":
        flip_one_byte(w.run_path)
    elif breakage == "bad_cli":
        args = eval_args(w, **{"--run-freeze-sha256": "f" * 64})
    elif breakage == "code_changed":
        (w.repo / "src" / "mod.py").write_text("X = 5\n")
        w.commit_all("later change")
    elif breakage == "ckpt":
        w.ckpt.write_bytes(b"other")
    elif breakage == "ext_manifest":
        w.ext_manifest.write_text("tampered\n")
    elif breakage == "dataset_freeze":
        flip_one_byte(w.ds_path)
    assert ep.eval_external_main(args, loader=lambda a: loaded.append(1) or 0) == ep.EXIT_VERIFY
    assert loaded == []


def test_eval_does_not_open_the_external_logits_file_when_verification_fails(w, tmp_path, monkeypatch):
    sentinel = tmp_path / "ext_logits.npy"
    sentinel.write_bytes(b"x")
    opened = []
    real_open = open

    def spy(file, *a, **k):
        if str(file) == str(sentinel):
            opened.append(file)
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", spy)
    flip_one_byte(w.run_path)
    assert ep.eval_external_main(eval_args(w, **{"--external-logits": str(sentinel)})) == ep.EXIT_VERIFY
    assert opened == []


def run_cli(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_verify_freeze_cli_exit_codes(w):
    base = ["--repo-root", w.repo]
    ok = run_cli(ROOT / "tools" / "verify_freeze.py", "run", "--freeze", w.run_path, "--sha256", w.run_hash(), *base,
                 "--checkpoint", w.ckpt, "--source-manifest", w.src_manifest, "--external-manifest", w.ext_manifest,
                 "--dataset-freeze", w.ds_path)
    assert ok.returncode == 0 and ok.stdout.startswith("OK run freeze verified")
    ds = run_cli(ROOT / "tools" / "verify_freeze.py", "dataset", "--freeze", w.ds_path, "--sha256", w.ds_hash(), *base)
    assert ds.returncode == 0
    flip_one_byte(w.ckpt, 3)
    bad = run_cli(ROOT / "tools" / "verify_freeze.py", "run", "--freeze", w.run_path, "--sha256", w.run_hash(), *base,
                  "--checkpoint", w.ckpt, "--source-manifest", w.src_manifest, "--external-manifest", w.ext_manifest,
                  "--dataset-freeze", w.ds_path)
    assert bad.returncode == 1 and "checkpoint_mismatch" in bad.stderr
    usage = run_cli(ROOT / "tools" / "verify_freeze.py", "run")
    assert usage.returncode == 2


def test_scripts_exit_nonzero_without_a_freeze():
    for script in ("train.py", "eval_external.py"):
        r = run_cli(ROOT / "scripts" / script)
        assert r.returncode != 0
