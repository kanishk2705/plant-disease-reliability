import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from plantrel import checkpoint as ck  # noqa: E402
from plantrel.hashing import config_hash, sha256_file  # noqa: E402

CFG = {"lr": 0.1, "epochs": 4, "model": "synthetic"}
H = config_hash(CFG)


def build(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    return model, opt, sched, scaler


def epoch(model, opt, sched):
    """One synthetic epoch whose data come from the global RNGs, so RNG restoration matters."""
    x = torch.randn(8, 4) + float(np.random.rand()) + random.random()
    y = torch.randint(0, 2, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()


def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_save_load_roundtrip_restores_every_state(tmp_path):
    model, opt, sched, scaler = build()
    epoch(model, opt, sched)
    path = tmp_path / "last.pt"
    digest = ck.save_checkpoint(path, model=model, config_hash=H, epoch=3, best_metric=0.42, optimizer=opt,
                                scheduler=sched, scaler=scaler, extra={"note": "x"})
    assert digest == sha256_file(path)
    m2, o2, s2, sc2 = build(seed=99)
    payload = ck.load_checkpoint(path, expected_config_hash=H, model=m2, optimizer=o2, scheduler=s2, scaler=sc2)
    assert payload["epoch"] == 3 and payload["best_metric"] == 0.42 and payload["extra"] == {"note": "x"}
    assert torch.equal(weights(model), weights(m2))
    assert o2.state_dict()["param_groups"][0]["lr"] == opt.state_dict()["param_groups"][0]["lr"]
    assert s2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]


def test_resumed_run_matches_uninterrupted_run_exactly(tmp_path):
    # uninterrupted: 2 epochs
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    a, oa, sa, _ = build()
    epoch(a, oa, sa); epoch(a, oa, sa)
    # interrupted after epoch 1, perturb the global RNGs, resume in fresh objects
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    b, ob, sb, scb = build()
    epoch(b, ob, sb)
    mgr = ck.CheckpointManager(tmp_path, H)
    mgr.save(model=b, epoch=1, best_metric=None, is_best=False, optimizer=ob, scheduler=sb, scaler=scb)
    torch.manual_seed(12345); np.random.seed(12345); random.seed(12345)  # simulate a new process
    c, oc, sc, scc = build(seed=555)
    payload = mgr.resume(model=c, optimizer=oc, scheduler=sc, scaler=scc)
    assert payload["epoch"] == 1
    epoch(c, oc, sc)
    assert torch.equal(weights(a), weights(c))


def test_config_mismatch_is_rejected_before_any_state_is_touched(tmp_path):
    model, opt, sched, _ = build()
    path = tmp_path / "last.pt"
    ck.save_checkpoint(path, model=model, config_hash=H, epoch=1, best_metric=None, optimizer=opt, scheduler=sched)
    m2, o2, s2, _ = build(seed=42)
    before = weights(m2).clone()
    with pytest.raises(ck.ConfigMismatchError):
        ck.load_checkpoint(path, expected_config_hash=config_hash({**CFG, "lr": 0.2}), model=m2, optimizer=o2)
    assert torch.equal(weights(m2), before)


def test_manager_refuses_resume_under_a_different_config(tmp_path):
    model, opt, sched, _ = build()
    ck.CheckpointManager(tmp_path, H).save(model=model, epoch=1, best_metric=1.0, is_best=True, optimizer=opt, scheduler=sched)
    other = ck.CheckpointManager(tmp_path, config_hash({"different": True}))
    with pytest.raises(ck.ConfigMismatchError):
        other.resume(model=build()[0])


def test_manager_best_and_last_and_empty_resume(tmp_path):
    mgr = ck.CheckpointManager(tmp_path / "run", H)
    model = build()[0]
    assert mgr.resume(model=model) is None
    h = mgr.save(model=model, epoch=1, best_metric=0.5, is_best=True)
    assert set(h) == {"last", "best"} and mgr.last_path.exists() and mgr.best_path.exists()
    h2 = mgr.save(model=model, epoch=2, best_metric=0.5, is_best=False)
    assert set(h2) == {"last"}


def test_interrupted_write_leaves_existing_checkpoint_intact(tmp_path, monkeypatch):
    model, opt, sched, _ = build()
    path = tmp_path / "last.pt"
    ck.save_checkpoint(path, model=model, config_hash=H, epoch=1, best_metric=None)
    good_bytes = path.read_bytes()

    def broken_save(obj, f, *a, **k):
        f.write(b"partial garbage")
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(ck.torch, "save", broken_save)
    with pytest.raises(OSError):
        ck.save_checkpoint(path, model=model, config_hash=H, epoch=2, best_metric=None)
    assert path.read_bytes() == good_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]  # temporary file cleaned up


def test_interrupted_first_write_creates_no_file(tmp_path, monkeypatch):
    def broken_save(obj, f, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(ck.torch, "save", broken_save)
    with pytest.raises(KeyboardInterrupt):
        ck.save_checkpoint(tmp_path / "last.pt", model=build()[0], config_hash=H, epoch=1, best_metric=None)
    assert list(tmp_path.iterdir()) == []


def test_missing_and_corrupt_checkpoints_are_rejected(tmp_path):
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(tmp_path / "nope.pt", expected_config_hash=H)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(bad, expected_config_hash=H)


def test_requesting_optimizer_state_that_was_not_saved_is_an_error(tmp_path):
    model, opt, _, _ = build()
    path = tmp_path / "m.pt"
    ck.save_checkpoint(path, model=model, config_hash=H, epoch=1, best_metric=None)
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(path, expected_config_hash=H, optimizer=opt)


def test_config_hash_is_order_independent_and_rejects_non_json():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    with pytest.raises(TypeError):
        config_hash({"a": object()})
