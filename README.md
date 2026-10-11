# Plant Disease Classifier Reliability Study

An independent, replication-style comparison of a from-scratch CNN and a fine-tuned ResNet-18,
trained on PlantVillage and evaluated on PlantDoc, with attention to calibration and
confidence-based abstention.

## What this is, and is not

- **Not a claim of novelty.** Cross-dataset plant-disease classification, calibration and selective
  prediction have been studied together in recent work (see `docs/research/literature_review.md`,
  section 6). This project is a transparent, reproducible replication-style comparison.
- **Not an agricultural diagnosis service.** Nothing here should be used to decide treatment, spraying
  or any field action.
- **Licensing and class alignment are unresolved.** The PlantVillage licence conflict, the dataset
  distributions, the PlantVillage-to-PlantDoc class mapping, the crop policy, the minimum counts and the
  deduplication settings are Phase 3 audit items. Nothing is assumed about them.
- **Status at the start of Phase 2:** no dataset has been audited, no model has been trained, and no
  result has been measured. Numbers in the research documents come from *other* studies and are not
  results of this project.

## Research questions (protocol section 1)

1. Generalisation: how large is the accuracy and macro-F1 gap between a PlantVillage test split and PlantDoc?
2. Model comparison: does ResNet-18 beat the scratch CNN on PlantDoc, paired?
3. Confidence reliability: how do ECE and Brier change from source to external data after
   source-validation-only temperature scaling?
4. Abstention: does a threshold chosen on source validation data still control risk on PlantDoc?
5. Error patterns (exploratory): per-class recall, confusions, high-confidence errors.

## Planned workflow

| Phase | Content | State |
|---|---|---|
| 0 | Environment decision (Colab GPU, laptop for development) | closed |
| 1 | Literature review and protocol | protocol v1.0 approved for implementation |
| 2 | Repository, utilities, tests (this repository) | utilities and tests exist; see "Current status" |
| 3 | Dataset acquisition and audit (provisional decisions) | not started |
| 4 | Preprocessing, splits, manifests, leakage checks, **dataset freeze** | not started |
| 5+ | Development run, training, calibration, run freeze, external evaluation, error analysis, demo | not started |

No training may begin before the dataset freeze passes verification (protocol 4.8, amendment A2.1).

## Evaluation methodology (protocol sections 4 to 9)

- **Source split:** by PlantVillage `leaf_id`; train 70%, `val_sel` 7.5%, `val_cal` 7.5%, test 15%; split seed 42; training seeds 1, 2, 3 are the target.
- **Selection:** checkpoint with the lowest `val_sel` NLL. `val_cal` is used only for temperature scaling and the abstention threshold, and its metrics are labelled development diagnostics.
- **Calibration:** one scalar temperature fitted by NLL on `val_cal` only. No PlantDoc labels are used for calibration.
- **Metrics:** accuracy, macro-F1 over a frozen class list (zero-division = 0), per-class metrics, confusion matrices, ECE (15 equal-width and 15 equal-mass bins), multiclass Brier (sum over classes, range [0, 2]), NLL, coverage, selective risk, risk-coverage curve, AURC, AUROC for error detection (positive = incorrect, score = 1 - max probability).
- **Uncertainty:** cluster bootstrap by `leaf_id` (source) and source-image or duplicate-group component (external); paired bootstrap for model differences; seed variability reported separately.
- **Isolation:** external labels may be inspected in the pre-model audit under fixed rules; no model output on external data may influence any decision. External evaluation runs only after the run freeze verifies.

## Repository layout

```
src/plantrel/       hashing, repro, checkpoint, metrics, bootstrap, freeze, staging, entrypoints
scripts/            command-line entry points (train.py and eval_external.py are guards only)
tools/              verify_freeze.py
configs/            placeholder.yaml; frozen/ holds freeze files (none yet)
tests/              pytest suite using synthetic data and temporary Git repositories only
notebooks/          00_colab_bootstrap.ipynb (thin Colab entry point)
docs/research/      literature_review.md, experimental_protocol.md, references.md
data/, outputs/     local artefacts, excluded from Git
```

## Setup

Laptop or any machine (CPU is enough for the current code and tests):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install torch torchvision        # only if torch is not already installed
pip install -e . --no-deps
python -m pytest
```

Colab: open `notebooks/00_colab_bootstrap.ipynb`, edit the placeholders in its first cell, and run it.
It installs the package with `--no-deps` so Colab's pre-installed PyTorch is not replaced. A Colab session
verified in Phase 0 had Python 3.13.15, PyTorch 2.11.0+cu130 and one Tesla T4, but every session must
record its own hardware and versions (`scripts/log_environment.py`); a later session may differ.
Do not run a Streamlit demo on a free Colab runtime; the demo is planned for AWS later.

Dependencies are specified as lower bounds in `requirements.txt`. The versions the tests were actually
run against are in `docs/ENVIRONMENT.md`.

## Freeze verification

```bash
python tools/verify_freeze.py dataset --freeze configs/frozen/dataset_decisions_v1.json --sha256 <hash from the run log> --repo-root .
python tools/verify_freeze.py run --freeze configs/frozen/<run_id>.json --sha256 <hash> --repo-root . \
    --checkpoint <ckpt> --source-manifest <m> --external-manifest <m> --dataset-freeze configs/frozen/dataset_decisions_v1.json
```

Exit status 0 means every check passed; anything else means do not proceed. The verifier compares the
file hash with its sidecar and with an independently supplied hash, runs
`git diff --exit-code <code_commit> HEAD` over the code paths, and separately checks the staged, unstaged
and untracked state. A clean working tree alone is never treated as proof that the code is unchanged.

## Current status (end of Phase 2 work in this repository)

See "Test evidence" in `docs/ENVIRONMENT.md`. Utilities and tests exist; nothing has been run on Colab,
on a GPU, or on real data. No dataset audit, training, calibration result or freeze for real data exists.

## Licence and citation

Placeholders only: `LICENSE-PLACEHOLDER.md`, `CITATION.cff`. No licence has been chosen. Dataset
licences are unresolved; datasets and weights are never committed.
