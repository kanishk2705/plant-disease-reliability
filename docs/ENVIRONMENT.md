# Environment and test evidence (Phase 2)

Recorded 2026-10-11 from the development sandbox. This is **not** the Colab environment.

## Where the tests were run

| Item | Value |
|---|---|
| Machine | Linux sandbox, 2 CPUs, no GPU (`cuda_available False`) |
| Python | 3.13.16 |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 (CPU execution only) |
| numpy / scipy / pandas / Pillow / PyYAML | 2.5.3 / 1.18.1 / 3.0.5 / 12.3.0 / 6.0.3 |
| scikit-learn / pytest / matplotlib | 1.9.1 / 9.1.1 / 3.11.2 |
| Git | 2.43.0 |

The verified Colab session from Phase 0 (Python 3.13.15, PyTorch 2.11.0+cu130, one Tesla T4) is **not**
what these tests ran on. PyTorch 2.14.1 was installed here from PyPI because the sandbox had none;
Colab's pre-installed build must be used there and was not touched by the package metadata
(torch is not a core dependency). Compatibility with Colab's torch 2.11.0 is expected but **unverified**.

## Commands and results

| Command | Exit code | Result |
|---|---|---|
| `python -m pytest -q` | 0 | 218 passed, 0 failed, 0 skipped, about 14 s |
| `pip install --no-deps -e .` in a fresh venv, then `import plantrel` | 0 | import ok, version 0.1.0.dev0 |
| `python scripts/log_environment.py --out ...` | 0 | environment record written; warns that no GPU is visible |
| `python scripts/train.py` (no arguments) | 2 | usage error, refuses to run |
| `python tools/verify_freeze.py dataset ... --sha256 <zeros>` (no freeze file exists) | 1 | `FAIL [missing_file]` |

Tests per file: metrics 41, bootstrap 13, checkpoint 10, repro 10, staging 7, freeze 101, entrypoints 20, imports 16.

Mutation check (code deliberately broken, then restored): skipping the commit-to-HEAD comparison,
averaging nothing for AUROC ties (ordinal ranks), dropping absent classes from macro-F1, and resampling
rows instead of clusters each made the relevant tests fail. This shows those tests can fail; it does not
show the code is free of other defects.

## Blocked or not run

- **Colab notebook** (`notebooks/00_colab_bootstrap.ipynb`): not executed. It needs Colab, Drive and a GitHub remote. Only its JSON structure was parsed and each code cell was syntax-checked (`nbformat` was not installed).
- **GPU code paths**: CUDA RNG-state capture and restore, GradScaler on CUDA, and deterministic algorithms on a GPU are not exercised. Checkpoint tests used a CPU GradScaler.
- **Colab versions**: Python 3.13.15 and torch 2.11.0+cu130 not tested.
- **GitHub**: no remote is configured and nothing was pushed.
- **Real data, real model, real freeze**: none exist. Every freeze test uses synthetic fixtures.

## Scenarios still requiring implementation

- Dataset audit tooling, manifests, split generation, leakage and duplicate checks (Phases 3 and 4).
- A command that assembles a run-freeze file from real outputs (the writer and verifier exist; the assembling script does not).
- Dataset-freeze assembly from audit outputs; integration of the freeze check into the real training loop.
- Saving and loading logits and manifests; per-class bootstrap wrappers; reliability-diagram plotting.
- Models, data loaders, training loop, the development run, temperature-scaling pipeline wiring (the scalar fit function exists and is tested on synthetic data).
- Verification that a bootstrap of 10,000 replicates is fast enough on real sizes (only small synthetic sizes were timed).

## Known limitations of the freeze verifier

It guards against accidental or undeclared change. Someone who edits the freeze file, sidecar, run log and
verifier together is not stopped; the Git history and tags are the evidence trail. Files that are listed in
`.gitignore` are not seen by the untracked-file check. `tests/` is deliberately outside the code-snapshot paths.
