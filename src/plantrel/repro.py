"""Seeding and environment recording.

Exact reproducibility is NOT guaranteed across different GPU models, driver or library
versions, or when non-deterministic kernels are enabled. ``set_seed`` returns a
``ReproStatus`` that says plainly what was and was not controlled.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .hashing import config_hash

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "pillow", "pyyaml", "matplotlib", "torch", "torchvision")


@dataclass
class ReproStatus:
    seed: int
    seeded: list[str]
    deterministic_requested: bool
    deterministic_settings: dict
    guarantees: str
    not_guaranteed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int, deterministic: bool = False) -> ReproStatus:
    """Seed Python, NumPy and (if installed) PyTorch.

    ``deterministic=True`` additionally requests deterministic PyTorch algorithms
    (``warn_only=True`` so unsupported ops warn rather than crash) and disables cuDNN
    autotuning. This is slower and still gives same-device, same-version reproducibility
    only.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    seeded = ["python.random", "numpy.random(global)"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # PYTHONHASHSEED only affects interpreters started afterwards (e.g. DataLoader workers
    # are seeded separately by the training code in a later phase).
    os.environ["PYTHONHASHSEED"] = str(seed)
    settings: dict = {}
    not_guaranteed = [
        "results across different GPU models",
        "results across different CUDA, cuDNN, driver or PyTorch versions",
        "DataLoader worker randomness unless worker seeds are set by the training code",
    ]
    try:
        import torch
    except Exception:  # torch not installed: report it, do not fail
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        seeded.append("torch.manual_seed")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded.append("torch.cuda.manual_seed_all")
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            settings = {"use_deterministic_algorithms": "True (warn_only)", "cudnn.deterministic": True,
                        "cudnn.benchmark": False, "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
        else:
            if hasattr(torch.backends, "cudnn"):
                settings = {"cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
                            "cudnn.deterministic": bool(torch.backends.cudnn.deterministic)}
            not_guaranteed.append("bitwise-identical GPU kernels (deterministic mode was not requested)")
    else:
        not_guaranteed.append("torch is not installed, so torch seeding was not performed")
    guarantees = ("Same seed gives the same random streams on the same device and library versions."
                  if deterministic and torch is not None else
                  "Random streams are seeded; bitwise-identical results are not promised.")
    return ReproStatus(seed, seeded, deterministic, settings, guarantees, not_guaranteed)


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info(repo_dir: str | Path | None = None) -> dict:
    cwd = str(repo_dir) if repo_dir else None
    commit = _run(["git", "rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "dirty": None, "note": "not a git repository or git unavailable"}
    status = _run(["git", "status", "--porcelain"], cwd)
    return {"commit": commit, "dirty": bool(status), "note": None}


def collect_environment(repo_dir: str | Path | None = None) -> dict:
    """Facts about the current machine and installed libraries. Nothing is assumed."""
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "packages": {p: _version(p) for p in PACKAGES},
        "git": git_info(repo_dir),
    }
    cuda: dict = {"torch_installed": False}
    try:
        torch = importlib.import_module("torch")
        cuda = {
            "torch_installed": True,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
            "cudnn": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        }
    except Exception as exc:
        cuda["error"] = type(exc).__name__
    env["cuda"] = cuda
    env["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return env


def write_environment_log(path: str | Path, config: dict | None = None, repo_dir: str | Path | None = None,
                          repro: ReproStatus | None = None) -> dict:
    """Write one JSON record: environment, git revision, config and its hash, seeding status."""
    record = {"environment": collect_environment(repo_dir)}
    if config is not None:
        record["config"] = config
        record["config_hash"] = config_hash(config)
    if repro is not None:
        record["reproducibility"] = repro.to_dict()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
