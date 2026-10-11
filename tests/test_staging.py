import tarfile
import zipfile

import pytest

from plantrel import staging
from plantrel.hashing import sha256_file


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def test_stage_verify_extract_zip(tmp_path):
    (tmp_path / "drive").mkdir()
    arc = make_zip(tmp_path / "drive" / "d.zip", {"a/x.txt": "hello"})
    res = staging.stage_archive(arc, tmp_path / "local", expected_sha256=sha256_file(arc), extract_to=tmp_path / "out")
    assert res["verified"] and (tmp_path / "out" / "a" / "x.txt").read_text() == "hello"
    assert res["extracted"]["n_entries"] == 1


def test_hash_mismatch_removes_partial_copy_and_raises(tmp_path):
    arc = make_zip(tmp_path / "d.zip", {"x": "1"})
    with pytest.raises(staging.StagingError, match="mismatch"):
        staging.stage_archive(arc, tmp_path / "local", expected_sha256="0" * 64)
    assert list((tmp_path / "local").iterdir()) == []


def test_missing_hash_refused_unless_explicitly_allowed(tmp_path):
    arc = make_zip(tmp_path / "d.zip", {"x": "1"})
    with pytest.raises(staging.StagingError):
        staging.stage_archive(arc, tmp_path / "l", expected_sha256=None)
    res = staging.stage_archive(arc, tmp_path / "l", expected_sha256=None, allow_unverified=True)
    assert res["verified"] is False and res["sha256"] == sha256_file(arc)


def test_invalid_expected_hash_and_missing_source(tmp_path):
    arc = make_zip(tmp_path / "d.zip", {"x": "1"})
    with pytest.raises(staging.StagingError):
        staging.stage_archive(arc, tmp_path / "l", expected_sha256="abc")
    with pytest.raises(staging.StagingError):
        staging.stage_archive(tmp_path / "none.zip", tmp_path / "l", expected_sha256="0" * 64)


def test_zip_path_traversal_rejected(tmp_path):
    arc = make_zip(tmp_path / "evil.zip", {"../escape.txt": "x"})
    with pytest.raises(staging.StagingError, match="unsafe"):
        staging.extract_archive(arc, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_tar_extract_and_symlink_rejected(tmp_path):
    src = tmp_path / "f.txt"
    src.write_text("hi")
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as t:
        t.add(src, arcname="d/f.txt")
    assert staging.extract_archive(good, tmp_path / "o1") == ["d/f.txt"]
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as t:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        t.addfile(info)
    with pytest.raises(staging.StagingError, match="link"):
        staging.extract_archive(bad, tmp_path / "o2")


def test_corrupt_archive_rejected(tmp_path):
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not an archive")
    with pytest.raises(staging.StagingError):
        staging.extract_archive(junk, tmp_path / "o")
