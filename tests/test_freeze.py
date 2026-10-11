"""Freeze-verification scenarios on synthetic fixtures and temporary Git repositories."""
import json

import pytest

from freeze_helpers import flip_one_byte, git, make_world, sha, sidecar_path
from plantrel.freeze import (FreezeVerificationError, require_dataset_freeze, verify_dataset_freeze, write_freeze_file)
from plantrel.hashing import sha256_file


@pytest.fixture
def w(tmp_path):
    return make_world(tmp_path)


def fails(check, fn, *a, **k):
    with pytest.raises(FreezeVerificationError) as ei:
        fn(*a, **k)
    assert ei.value.check == check, f"expected {check}, got {ei.value}"


# ------------------------------------------------------------------ the happy path
def test_valid_run_freeze_passes_every_check(w):
    rep = w.verify()
    assert rep.passed == ["file_sidecar_cli_hash", "fields", "tracked_and_unmodified", "code_unchanged_since_code_commit",
                          "worktree_clean", "dataset_freeze", "class_lists", "manifests", "checkpoint"]


def test_valid_dataset_freeze_passes(w):
    obj, rep = verify_dataset_freeze(w.ds_path, expected_sha256=w.ds_hash(), repo_root=w.repo,
                                     source_manifest_path=w.src_manifest, external_manifest_path=w.ext_manifest)
    assert obj["protocol_version"] == "1.0" and "source_manifest" in rep.passed


def test_sidecar_is_sha256sum_compatible(w):
    line = sidecar_path(w.run_path).read_text()
    assert line == f"{w.run_hash()}  {w.run_path.name}\n"


# ------------------------------------------------------------------ hash chain (run freeze)
def test_missing_freeze_file(w):
    w.run_path.unlink()
    fails("missing_file", w.verify, expected_sha256="0" * 64)


def test_missing_sidecar(w):
    sidecar_path(w.run_path).unlink()
    fails("missing_sidecar", w.verify)


def test_incorrect_sidecar_hash(w):
    sidecar_path(w.run_path).write_text(f"{sha('b')}  {w.run_path.name}\n")
    fails("hash_mismatch", w.verify)


def test_malformed_sidecar(w):
    sidecar_path(w.run_path).write_text("garbage\n")
    fails("malformed_sidecar", w.verify)


def test_one_byte_modification_detected(w):
    good = w.run_hash()
    flip_one_byte(w.run_path)
    fails("hash_mismatch", w.verify, expected_sha256=good)


def test_one_byte_modification_with_sidecar_rewritten_is_still_caught_by_independent_hash(w):
    good = w.run_hash()
    flip_one_byte(w.run_path)
    sidecar_path(w.run_path).write_text(f"{sha256_file(w.run_path)}  {w.run_path.name}\n")
    fails("cli_hash_mismatch", w.verify, expected_sha256=good)


@pytest.mark.parametrize("value,check", [("not-a-hash", "invalid_cli_hash"), ("abc", "invalid_cli_hash"),
                                          (None, "missing_cli_hash"), ("0" * 64, "cli_hash_mismatch")])
def test_invalid_command_line_sha256(w, value, check):
    fails(check, w.verify, expected_sha256=value)


def test_uppercase_command_line_hash_is_accepted(w):
    w.verify(expected_sha256=w.run_hash().upper())


# ------------------------------------------------------------------ Git protections
def test_code_changed_after_code_commit_and_committed_is_caught_even_though_tree_is_clean(w):
    (w.repo / "src" / "mod.py").write_text("X = 2\n")
    w.commit_all("sneaky later change")
    assert git(w.repo, "status", "--porcelain") == ""  # a clean tree alone would have passed
    fails("code_changed_since_code_commit", w.verify)


def test_config_changed_after_code_commit_and_committed_is_caught(w):
    (w.repo / "configs" / "exp.yaml").write_text("lr: 0.9\n")
    w.commit_all("change run config")
    fails("code_changed_since_code_commit", w.verify)


def test_new_code_file_committed_later_is_caught(w):
    (w.repo / "scripts" / "extra.py").write_text("print(1)\n")
    w.commit_all("add script")
    fails("code_changed_since_code_commit", w.verify)


def test_unstaged_code_modification_is_caught(w):
    (w.repo / "src" / "mod.py").write_text("X = 3\n")
    fails("unstaged_code_changes", w.verify)


def test_staged_code_modification_is_caught(w):
    (w.repo / "src" / "mod.py").write_text("X = 4\n")
    git(w.repo, "add", "src/mod.py")
    fails("staged_code_changes", w.verify)


def test_untracked_code_file_is_caught(w):
    (w.repo / "src" / "new_untracked.py").write_text("Y = 1\n")
    fails("untracked_code_files", w.verify)


def test_changes_outside_code_paths_do_not_block(w):
    (w.repo / "notes.md").write_text("scratch\n")
    (w.repo / "tests").mkdir()
    (w.repo / "tests" / "t.py").write_text("pass\n")
    w.commit_all("docs and tests")
    w.verify()


def test_new_freeze_metadata_in_frozen_dir_is_excluded_from_the_code_comparison(w):
    (w.repo / "configs" / "frozen" / "another.json").write_text("{}\n")
    w.commit_all("more freeze metadata")
    w.verify()


def test_unknown_code_commit(w):
    w.rewrite_run_freeze(lambda o: o.update(code_commit="1" * 40))
    fails("unknown_code_commit", w.verify, expected_sha256=w.run_hash())


def test_code_commit_not_an_ancestor_of_head(w):
    git(w.repo, "checkout", "-q", "-b", "side")
    (w.repo / "src" / "mod.py").write_text("X = 9\n")
    w.commit_all("side work")
    side = git(w.repo, "rev-parse", "HEAD")
    git(w.repo, "checkout", "-q", "main")
    w.rewrite_run_freeze(lambda o: o.update(code_commit=side))
    fails("code_commit_not_ancestor", w.verify, expected_sha256=w.run_hash())


def test_freeze_file_not_tracked(w):
    git(w.repo, "rm", "-q", "--cached", "configs/frozen/synthetic-run-1.json", "configs/frozen/synthetic-run-1.sha256")
    git(w.repo, "commit", "-q", "-m", "untrack")
    fails("freeze_not_tracked", w.verify)


def test_freeze_file_inside_repo_required(w, tmp_path):
    outside = tmp_path / "outside.json"
    write_freeze_file(outside, w.run_obj)
    from plantrel.freeze import verify_run_freeze
    with pytest.raises(FreezeVerificationError) as ei:
        verify_run_freeze(outside, expected_sha256=sha256_file(outside), repo_root=w.repo, checkpoint_path=w.ckpt,
                          source_manifest_path=w.src_manifest, external_manifest_path=w.ext_manifest,
                          dataset_freeze_path=w.ds_path)
    assert ei.value.check == "freeze_outside_repo"


def test_not_a_git_repository_fails_closed(tmp_path):
    w = make_world(tmp_path)
    import shutil
    shutil.rmtree(w.repo / ".git")
    with pytest.raises(FreezeVerificationError):
        w.verify()


# ------------------------------------------------------------------ artefacts
def test_changed_checkpoint(w):
    w.ckpt.write_bytes(b"different")
    fails("checkpoint_mismatch", w.verify)


def test_missing_checkpoint(w):
    w.ckpt.unlink()
    fails("missing_checkpoint", w.verify)


def test_changed_source_manifest(w):
    w.src_manifest.write_text("id,label\n1,c1\n")
    fails("source_manifest_mismatch", w.verify)


def test_changed_external_manifest(w):
    w.ext_manifest.write_text("id,label\n9,c2\n")
    fails("external_manifest_mismatch", w.verify)


def test_missing_manifest_path(w):
    fails("missing_manifest", w.verify, external_manifest_path=None)


def test_manifest_hashes_must_agree_with_dataset_freeze(w):
    w.rewrite_run_freeze(lambda o: o["manifest_hashes"].update(source=sha("c")))
    fails("manifest_hash_mismatch", w.verify, expected_sha256=w.run_hash())


def test_class_list_must_agree_with_dataset_freeze(w):
    w.rewrite_run_freeze(lambda o: o.update(class_list=["c0", "c1", "c2", "c3"]))
    fails("class_list_mismatch", w.verify, expected_sha256=w.run_hash())


# ------------------------------------------------------------------ dataset freeze hash chain
def test_dataset_freeze_byte_change_breaks_the_recorded_run_freeze_hash(w):
    flip_one_byte(w.ds_path)
    fails("dataset_freeze_hash_mismatch", w.verify)


def test_dataset_freeze_hash_mismatch_with_independent_hash(w):
    fails("cli_hash_mismatch", w.verify, dataset_freeze_expected_sha256=sha("d"))


def test_dataset_freeze_sidecar_wrong(w):
    sidecar_path(w.ds_path).write_text(f"{sha('e')}  {w.ds_path.name}\n")
    fails("hash_mismatch", w.verify)


def test_missing_dataset_freeze(w):
    w.ds_path.unlink()
    fails("missing_file", w.verify)


def test_dataset_freeze_missing_file_direct(w):
    w.ds_path.unlink()
    fails("missing_file", verify_dataset_freeze, w.ds_path, expected_sha256="0" * 64)


def test_dataset_freeze_missing_sidecar_direct(w):
    sidecar_path(w.ds_path).unlink()
    fails("missing_sidecar", verify_dataset_freeze, w.ds_path, expected_sha256=w.ds_hash())


def test_dataset_freeze_one_byte_direct(w):
    good = w.ds_hash()
    flip_one_byte(w.ds_path)
    fails("hash_mismatch", verify_dataset_freeze, w.ds_path, expected_sha256=good)


def test_dataset_freeze_bad_cli_hash_direct(w):
    fails("invalid_cli_hash", verify_dataset_freeze, w.ds_path, expected_sha256="xyz")
    fails("cli_hash_mismatch", verify_dataset_freeze, w.ds_path, expected_sha256="0" * 64)


def test_dataset_freeze_source_manifest_mismatch(w):
    w.src_manifest.write_text("changed\n")
    fails("source_manifest_mismatch", verify_dataset_freeze, w.ds_path, expected_sha256=w.ds_hash(),
          source_manifest_path=w.src_manifest)


def test_dataset_freeze_not_tracked(w):
    git(w.repo, "rm", "-q", "--cached", "configs/frozen/dataset_decisions_v1.json", "configs/frozen/dataset_decisions_v1.sha256")
    git(w.repo, "commit", "-q", "-m", "untrack")
    fails("freeze_not_tracked", verify_dataset_freeze, w.ds_path, expected_sha256=w.ds_hash(), repo_root=w.repo)


# ------------------------------------------------------------------ missing and malformed fields
RUN_KEYS = ["schema_version", "run_id", "seed", "code_commit", "T", "tau", "checkpoint_sha256", "class_list",
            "eval_class_lists", "config_hash", "manifest_hashes", "dataset_freeze_sha256", "environment"]
DS_KEYS = ["schema_version", "protocol_version", "class_list", "eval_class_lists", "class_alignment", "crop_policy",
           "min_counts", "dedup_settings", "distributions", "evaluation_transform", "licence_findings",
           "split_manifest_sha256", "external_manifest_sha256"]


@pytest.mark.parametrize("key", RUN_KEYS)
def test_run_freeze_missing_field(w, key):
    w.rewrite_run_freeze(lambda o: o.pop(key))
    fails("missing_field", w.verify, expected_sha256=w.run_hash())


@pytest.mark.parametrize("key", DS_KEYS)
def test_dataset_freeze_missing_field(w, key):
    w.rewrite_dataset_freeze(lambda o: o.pop(key))
    fails("missing_field", verify_dataset_freeze, w.ds_path, expected_sha256=w.ds_hash())


RUN_BAD = {
    "T_negative": lambda o: o.update(T=-1.0), "T_string": lambda o: o.update(T="1.7"), "T_bool": lambda o: o.update(T=True),
    "commit_short": lambda o: o.update(code_commit="abc123"), "commit_upper": lambda o: o.update(code_commit="A" * 40),
    "seed_negative": lambda o: o.update(seed=-1), "run_id_slash": lambda o: o.update(run_id="a/b"),
    "tau_cov_range": lambda o: o["tau"].update(r_cov=1.5), "tau_missing_rrisk": lambda o: o["tau"].pop("r_risk"),
    "tau_not_object": lambda o: o.update(tau=0.8), "ckpt_hash_short": lambda o: o.update(checkpoint_sha256="abc"),
    "classes_duplicate": lambda o: o.update(class_list=["c0", "c0"]),
    "eval_not_subset": lambda o: o["eval_class_lists"].update(external=["zzz"]),
    "eval_missing_key": lambda o: o["eval_class_lists"].pop("external"),
    "manifest_extra_key": lambda o: o["manifest_hashes"].update(other=sha("1")),
    "manifest_bad_hash": lambda o: o["manifest_hashes"].update(source="nope"),
    "env_not_object": lambda o: o.update(environment="linux"),
}


@pytest.mark.parametrize("name", list(RUN_BAD))
def test_run_freeze_malformed_field(w, name):
    w.rewrite_run_freeze(RUN_BAD[name])
    with pytest.raises(FreezeVerificationError) as ei:
        w.verify(expected_sha256=w.run_hash())
    assert ei.value.check == "malformed_field"


DS_BAD = {
    "protocol_version": lambda o: o.update(protocol_version="0.9"), "schema": lambda o: o.update(schema_version=2),
    "classes_empty": lambda o: o.update(class_list=[]), "alignment_empty": lambda o: o.update(class_alignment=[]),
    "crop_policy_str": lambda o: o.update(crop_policy="none"), "manifest_hash_bad": lambda o: o.update(split_manifest_sha256="1"),
}


@pytest.mark.parametrize("name", list(DS_BAD))
def test_dataset_freeze_malformed_field(w, name):
    w.rewrite_dataset_freeze(DS_BAD[name])
    fails("malformed_field", verify_dataset_freeze, w.ds_path, expected_sha256=w.ds_hash())


@pytest.mark.parametrize("raw", [b"not json at all", b"[1, 2, 3]\n", b'{"T": NaN}\n', b'{"a": Infinity}\n', b"\xff\xfe\x00bad"])
def test_malformed_json_is_rejected_even_with_consistent_hashes(w, raw):
    w.run_path.write_bytes(raw)
    sidecar_path(w.run_path).write_text(f"{sha256_file(w.run_path)}  {w.run_path.name}\n")
    fails("malformed_json", w.verify, expected_sha256=sha256_file(w.run_path))


def test_write_freeze_file_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        write_freeze_file(tmp_path / "x.json", {"T": float("nan")})


# ------------------------------------------------------------------ training guard helper
def test_require_dataset_freeze_needs_a_source_manifest(w):
    fails("missing_manifest", require_dataset_freeze, w.ds_path, expected_sha256=w.ds_hash(), repo_root=w.repo,
          source_manifest_path=None)


def test_require_dataset_freeze_does_not_need_the_external_manifest(w):
    w.ext_manifest.unlink()
    rep = require_dataset_freeze(w.ds_path, expected_sha256=w.ds_hash(), repo_root=w.repo, source_manifest_path=w.src_manifest)
    assert "external_manifest" not in rep.passed
