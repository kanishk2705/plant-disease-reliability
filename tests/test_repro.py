import json
import random

import numpy as np
import pytest

from plantrel import repro


def test_set_seed_reproduces_python_numpy_streams():
    repro.set_seed(11)
    a = (random.random(), float(np.random.rand()))
    repro.set_seed(11)
    assert a == (random.random(), float(np.random.rand()))


def test_torch_stream_reproducible():
    torch = pytest.importorskip("torch")
    repro.set_seed(3)
    a = torch.rand(3)
    repro.set_seed(3)
    assert torch.equal(a, torch.rand(3))


@pytest.mark.parametrize("bad", [-1, 1.5, "1", True])
def test_invalid_seed_rejected(bad):
    with pytest.raises(ValueError):
        repro.set_seed(bad)


def test_status_says_what_is_not_guaranteed():
    st = repro.set_seed(1)
    assert st.deterministic_requested is False
    assert any("GPU" in s for s in st.not_guaranteed)
    assert "python.random" in st.seeded


def test_deterministic_mode_is_requested_and_reported():
    torch = pytest.importorskip("torch")
    try:
        st = repro.set_seed(1, deterministic=True)
        assert st.deterministic_requested and torch.are_deterministic_algorithms_enabled()
        assert st.deterministic_settings["cudnn.deterministic"] is True
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def test_collect_environment_reports_without_assuming():
    env = repro.collect_environment()
    assert env["python"].count(".") == 2
    assert "torch" in env["packages"] and "numpy" in env["packages"]
    assert "cuda_available" in env["cuda"] or env["cuda"]["torch_installed"] is False


def test_write_environment_log_records_config_hash(tmp_path):
    out = tmp_path / "env.json"
    rec = repro.write_environment_log(out, config={"lr": 0.1}, repro=repro.set_seed(0))
    on_disk = json.loads(out.read_text())
    assert on_disk["config_hash"] == rec["config_hash"] and len(on_disk["config_hash"]) == 64
    assert "reproducibility" in on_disk and "environment" in on_disk
