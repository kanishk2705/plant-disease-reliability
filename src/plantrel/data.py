"""Image cache and GPU-side augmentation.

Every image (source and external) is resized once, whole-image, to SIZE x SIZE (bilinear, antialiased,
no crop) and stored as uint8 in a .npy cache. That IS the deterministic evaluation transform (protocol
section 4.3 / dataset freeze). Training augmentation runs on the GPU on batches (random resized crop,
rotation, flips, mild colour jitter) and never imitates the external domain.
"""
from __future__ import annotations

import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _load(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def build_cache(paths: list[str], out_path: str | Path, workers: int = 2, chunk: int = 64) -> Path:
    """Decode ``paths`` in order into a uint8 array [N, SIZE, SIZE, 3] saved at ``out_path`` (.npy)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(paths), SIZE, SIZE, 3))
    if workers > 1:
        with Pool(workers) as pool:
            for i, a in enumerate(pool.imap(_load, paths, chunksize=chunk)):
                arr[i] = a
    else:
        for i, p in enumerate(paths):
            arr[i] = _load(p)
    arr.flush(); del arr
    out_path.with_suffix(".tmp.npy").replace(out_path)
    return out_path


def normalise(x01, device_mean=None):
    import torch
    m = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s


def to_float(batch_u8):
    """uint8 [B,H,W,3] tensor -> float [B,3,H,W] in [0,1]."""
    return batch_u8.permute(0, 3, 1, 2).float().div_(255.0)


def augment(x01, gen, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3), max_rot_deg=15.0, jitter=0.2):
    """Per-sample random resized crop + rotation + h/v flips + brightness/contrast/saturation jitter."""
    import torch
    import torch.nn.functional as F
    B, dev = x01.shape[0], x01.device
    u = lambda lo, hi: torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo
    area = u(*scale)
    logr = u(math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(logr)
    sx = torch.sqrt(area * r).clamp(max=1.0)
    sy = torch.sqrt(area / r).clamp(max=1.0)
    tx = u(-1, 1) * (1 - sx)
    ty = u(-1, 1) * (1 - sy)
    ang = u(-max_rot_deg, max_rot_deg) * math.pi / 180
    fx = (torch.rand(B, device=dev, generator=gen) < 0.5).float() * 2 - 1
    fy = (torch.rand(B, device=dev, generator=gen) < 0.5).float() * 2 - 1
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = c * sx * fx; theta[:, 0, 1] = -s * sy * fy; theta[:, 0, 2] = tx
    theta[:, 1, 0] = s * sx * fx; theta[:, 1, 1] = c * sy * fy; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x01.shape), align_corners=False)
    x = F.grid_sample(x01, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    b = u(1 - jitter, 1 + jitter).view(B, 1, 1, 1)
    ct = u(1 - jitter, 1 + jitter).view(B, 1, 1, 1)
    st = u(1 - jitter, 1 + jitter).view(B, 1, 1, 1)
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * ct + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * st + gray
    return x.clamp_(0.0, 1.0)
