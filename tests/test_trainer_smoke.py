import numpy as np, pytest
torch = pytest.importorskip("torch")
from plantrel.data import augment
from plantrel.models import build_model, count_params
from plantrel import trainer


def _synthetic(tmp_path, n=96, k=3):
    rng = np.random.default_rng(0)
    y = np.arange(n) % k
    x = rng.integers(0, 40, size=(n, 224, 224, 3), dtype=np.uint8)
    for c in range(k):
        x[y == c, :, :, c] += 150
    split = np.array((["train"] * 12 + ["val_sel"] * 2 + ["val_cal"] * 2 + ["test"] * 4) * (n // 20 + 1))[:n]
    np.save(tmp_path / "c.npy", x); np.save(tmp_path / "l.npy", y); np.save(tmp_path / "s.npy", split)
    return tmp_path


def cfg(tmp_path, model="cnn", **kw):
    d = _synthetic(tmp_path)
    c = dict(model=model, seed=1, lr=1e-3, weight_decay=1e-4, batch_size=8, max_epochs=3, patience=5, warmup_epochs=1,
             cache=str(d / "c.npy"), labels=str(d / "l.npy"), splits=str(d / "s.npy"), run_dir=str(d / "run"),
             num_classes=3, amp=False, pretrained=False)
    c.update(kw); return c


def test_models_shapes_and_params():
    x = torch.zeros(2, 3, 224, 224)
    for n in ("cnn", "resnet18"):
        m = build_model(n, 5, pretrained=False)
        assert m(x).shape == (2, 5)
    assert count_params(build_model("resnet18", 1000, pretrained=False)) == 11_689_512


def test_augment_shape_range_and_seed():
    x = torch.rand(4, 3, 64, 64)
    g1 = torch.Generator(); g1.manual_seed(5); g2 = torch.Generator(); g2.manual_seed(5)
    a, b = augment(x, g1), augment(x, g2)
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1 and torch.equal(a, b)


def test_train_learns_and_resumes(tmp_path):
    c = cfg(tmp_path, max_epochs=4)
    s = trainer.run(c)
    assert s["val_sel_acc"] > 0.5
    z = np.load(tmp_path / "run" / "source_logits.npz")
    assert z["test_logits"].shape[1] == 3 and "external" not in " ".join(z.files)
    # resume: running again with same config continues past all epochs without error
    assert trainer.run(c)["config_hash"] == s["config_hash"]


def test_resume_refuses_changed_config(tmp_path):
    c = cfg(tmp_path)
    trainer.run(c)
    from plantrel.checkpoint import ConfigMismatchError
    with pytest.raises(ConfigMismatchError):
        trainer.run({**c, "lr": 5e-4})
