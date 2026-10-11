"""Training loop for one (model, seed, lr) run. Source data only; external data is never touched here.

Selection: lowest val_sel NLL (checkpoint), patience early stopping. val_cal and test are only
used after training, to save logits from the best checkpoint.
"""
from __future__ import annotations

import json
import math
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import CheckpointManager
from .data import augment, normalise, to_float
from .hashing import config_hash
from .models import build_model, count_params
from .repro import set_seed

SPLITS = ("train", "val_sel", "val_cal", "test")


def _batches(arr, labels, idx, bs, shuffle, rng, drop_last):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    n = len(idx) - (len(idx) % bs if drop_last else 0)
    for i in range(0, n, bs):
        b = np.sort(idx[i:i + bs])
        yield torch.from_numpy(np.ascontiguousarray(arr[b])), torch.from_numpy(labels[b]), b


def _prefetch(gen, depth=3):
    q: queue.Queue = queue.Queue(depth)
    END = object()

    def work():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(END)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not END:
        yield item


@torch.no_grad()
def predict_logits(model, arr, labels, idx, device, bs=256, amp=True):
    model.eval()
    out = np.zeros((len(idx), int(labels.max()) + 1 if len(labels) else 0), dtype=np.float32)
    order = {int(j): k for k, j in enumerate(idx)}
    outs, rows = [], []
    for xb, yb, b in _prefetch(_batches(arr, labels, idx, bs, False, None, False)):
        x = normalise(to_float(xb.to(device, non_blocking=True)))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            lg = model(x)
        outs.append(lg.float().cpu().numpy()); rows.extend(int(j) for j in b)
    lg = np.concatenate(outs)
    res = np.zeros_like(lg)
    for r, j in zip(range(len(rows)), rows):
        res[order[j]] = lg[r]
    return res


def nll_acc(logits, y):
    t = torch.from_numpy(logits)
    yt = torch.from_numpy(y).long()
    return float(F.cross_entropy(t, yt)), float((t.argmax(1) == yt).float().mean())


def run(cfg: dict) -> dict:
    """cfg keys: model, seed, lr, weight_decay, batch_size, max_epochs, patience, warmup_epochs, cache,
    labels, splits(.npy of split ids), run_dir, num_classes, amp, pretrained, max_train_batches (smoke)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status = set_seed(cfg["seed"])
    chash = config_hash(cfg)
    run_dir = Path(cfg["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    arr = np.load(cfg["cache"], mmap_mode="r")
    labels = np.load(cfg["labels"]).astype(np.int64)
    split = np.load(cfg["splits"])
    idx = {s: np.where(split == s)[0] for s in SPLITS}
    model = build_model(cfg["model"], cfg["num_classes"], cfg.get("pretrained", True)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps_per_epoch = max(1, len(idx["train"]) // bs)
    if cfg.get("max_train_batches"):
        steps_per_epoch = min(steps_per_epoch, cfg["max_train_batches"])
    total, warm = steps_per_epoch * cfg["max_epochs"], steps_per_epoch * cfg["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = bool(cfg.get("amp", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if dev.type == "cuda" else None
    mgr = CheckpointManager(run_dir, chash)
    state = mgr.resume(model=model, optimizer=opt, scheduler=sched, scaler=scaler if amp else None)
    start, best, bad = 0, float("inf"), 0
    if state is not None:
        start, best = state["epoch"] + 1, state["best_metric"] if state["best_metric"] is not None else float("inf")
        bad = state["extra"].get("bad", 0)
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"] * 1000 + start)
    rng = np.random.default_rng(cfg["seed"] * 1000 + start)
    log = run_dir / "log.jsonl"
    (run_dir / "config.json").write_text(json.dumps({**cfg, "config_hash": chash, "params": count_params(model),
                                                      "device": str(dev), "repro": status.to_dict()}, indent=1))
    for ep in range(start, cfg["max_epochs"]):
        model.train(); t0 = time.time(); tl, tn = 0.0, 0
        for step, (xb, yb, _) in enumerate(_prefetch(_batches(arr, labels, idx["train"], bs, True, rng, True))):
            if step >= steps_per_epoch:
                break
            x = to_float(xb.to(dev, non_blocking=True)); y = yb.to(dev, non_blocking=True)
            x = normalise(augment(x, gen))
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step(); tl += float(loss.detach()) * len(y); tn += len(y)
        lg = predict_logits(model, arr, labels, idx["val_sel"], dev, amp=amp)
        vn, va = nll_acc(lg, labels[idx["val_sel"]])
        is_best = vn < best
        if is_best:
            best, bad = vn, 0
        else:
            bad += 1
        mgr.save(model=model, epoch=ep, best_metric=best, is_best=is_best, optimizer=opt, scheduler=sched,
                 scaler=scaler if amp else None, extra={"bad": bad})
        with open(log, "a") as f:
            f.write(json.dumps({"epoch": ep, "train_loss": tl / max(tn, 1), "val_sel_nll": vn, "val_sel_acc": va,
                                "best": best, "is_best": is_best, "secs": round(time.time() - t0, 1)}) + "\n")
        if bad >= cfg["patience"]:
            break
    from .checkpoint import load_checkpoint
    load_checkpoint(mgr.best_path, expected_config_hash=chash, model=model, restore_rng=False)
    out = {}
    for s in ("val_sel", "val_cal", "test"):
        out[s + "_logits"] = predict_logits(model, arr, labels, idx[s], dev, amp=amp)
        out[s + "_labels"] = labels[idx[s]]
        out[s + "_index"] = idx[s]
    np.savez_compressed(run_dir / "source_logits.npz", **out)
    sel_nll, sel_acc = nll_acc(out["val_sel_logits"], out["val_sel_labels"])
    summ = {"best_val_sel_nll": sel_nll, "val_sel_acc": sel_acc, "config_hash": chash}
    (run_dir / "summary.json").write_text(json.dumps(summ))
    return summ
