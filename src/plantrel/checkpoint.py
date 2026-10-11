"""Atomic, resumable checkpoints.

A checkpoint stores model, optimiser, scheduler and AMP-scaler states, the epoch, the best
metric, the configuration hash and the RNG states. Writes go to a temporary file in the same
directory and are renamed over the target, so an interrupted write never damages an existing
checkpoint. Resuming with a different configuration hash is refused.

RNG states are converted to plain tuples, lists and tensors so that checkpoints load with
``torch.load(weights_only=True)``. Only load checkpoints you created.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hashing import sha256_file

FORMAT_VERSION = 1


class ConfigMismatchError(RuntimeError):
    """The checkpoint was written under a different configuration hash."""


class CheckpointError(RuntimeError):
    """The checkpoint file is missing or structurally invalid."""


def capture_rng_state() -> dict:
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (np_state[0], torch.from_numpy(np.array(np_state[1], dtype=np.int64)), int(np_state[2]),
                  int(np_state[3]), float(np_state[4])),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_checkpoint(path: str | Path, *, model, config_hash: str, epoch: int, best_metric: float | None,
                    optimizer=None, scheduler=None, scaler=None, extra: dict | None = None) -> str:
    """Atomically write a checkpoint and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "config_hash": config_hash,
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def load_checkpoint(path: str | Path, *, expected_config_hash: str, model=None, optimizer=None,
                    scheduler=None, scaler=None, restore_rng: bool = True, map_location="cpu") -> dict:
    """Load a checkpoint, refusing a configuration mismatch BEFORE touching any object."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"checkpoint unreadable: {path} ({type(exc).__name__})") from exc
    for key in ("format_version", "config_hash", "epoch", "best_metric", "model", "rng"):
        if key not in payload:
            raise CheckpointError(f"checkpoint missing field {key!r}")
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint format {payload['format_version']}")
    if payload["config_hash"] != expected_config_hash:
        raise ConfigMismatchError(
            f"config hash mismatch: checkpoint {payload['config_hash'][:12]}..., requested {expected_config_hash[:12]}...")
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise CheckpointError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise CheckpointError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise CheckpointError("checkpoint has no scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload


class CheckpointManager:
    """``last.pt`` every epoch, ``best.pt`` on improvement, in one run directory."""

    def __init__(self, run_dir: str | Path, config_hash: str):
        self.run_dir = Path(run_dir)
        self.config_hash = config_hash

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    def save(self, *, model, epoch: int, best_metric: float | None, is_best: bool, optimizer=None,
             scheduler=None, scaler=None, extra: dict | None = None) -> dict[str, str]:
        hashes = {"last": save_checkpoint(self.last_path, model=model, config_hash=self.config_hash, epoch=epoch,
                                          best_metric=best_metric, optimizer=optimizer, scheduler=scheduler,
                                          scaler=scaler, extra=extra)}
        if is_best:
            hashes["best"] = save_checkpoint(self.best_path, model=model, config_hash=self.config_hash,
                                             epoch=epoch, best_metric=best_metric, optimizer=optimizer,
                                             scheduler=scheduler, scaler=scaler, extra=extra)
        return hashes

    def resume(self, *, model, optimizer=None, scheduler=None, scaler=None) -> dict | None:
        """Return the restored payload, or None when there is nothing to resume.

        A config mismatch raises ConfigMismatchError; it is never ignored.
        """
        if not self.last_path.exists():
            return None
        return load_checkpoint(self.last_path, expected_config_hash=self.config_hash, model=model,
                               optimizer=optimizer, scheduler=scheduler, scaler=scaler)
