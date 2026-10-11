# Experimental Protocol, version 1.0 (approved for implementation; AUDIT items unresolved)

Plant Disease Classifier Reliability Study. Written 2026-10-11 (Phase 1). Amended to v1.0 on 2026-10-11 after the user's Phase 1 review (amendments A1.1 to A1.7, section 14).

**Status of this document.** Nothing in it has been executed. No dataset has been downloaded, no model trained, no result measured. Evidence for dataset facts is in `references.md` and `literature_review.md`.

**Status of v1.0.** The user approved the research direction and v1.0 incorporates the required amendments. v1.0 is **approved for implementation**. It is not a statement that the dataset-dependent decisions are made: every `AUDIT` item (PlantVillage licence, dataset distributions, class mapping, crop policy, minimum counts, deduplication settings) remains **unresolved** until Phase 3.

**Two freezes, not one.** (1) The **dataset freeze** (section 4.8), completed at the end of Phase 4 after the audit, preprocessing, partition generation, manifests and leakage checks have passed (amendment A2.1), fixes the taxonomy and every AUDIT value before any model is trained. (2) The **run freeze** (section 9.4) fixes T, tau and checkpoints before any external inference. Any change to this protocol after v1.0 needs a dated amendment in section 14 with a reason, and must never be motivated by seeing a model output on test or external data. The tag `protocol-v1.0` is created in Git when the repository exists (Phase 2).

## Status tags used throughout

| Tag | Meaning |
|---|---|
| **FIXED** | Decided now. Cannot change after v1.0 without a logged amendment. |
| **AUDIT** | Depends on facts from the Phase 3 dataset audit. The *rule* is fixed now; the *value* is filled in before any model is trained. Only data properties (counts, image sizes, labels) may inform it, never model outputs. |
| **DEV** | Settled during the limited development run using source validation data only. |
| **OPEN** | Unresolved. The user or later evidence must decide. Listed in section 13. |

---

## 1. Research questions and hypotheses

The five research questions are those in the project specification. Each gets a testable prediction. **The predictions are hypotheses, not expectations of fact.** Any of them may fail, and failure is reported.

| RQ | Primary endpoint (**FIXED**) | Hypothesis | What would contradict it |
|---|---|---|---|
| RQ1 Generalization | Accuracy and macro-F1 gaps, source test minus external test, in percentage points, per model | H1: both models score lower on the external set than on the source test. | Gap whose 95% interval includes zero or is negative |
| RQ2 Model comparison | Difference in external macro-F1, ResNet-18 minus CNN, paired | H2: ResNet-18 does better on the external set. **No direction is assumed for calibration.** | Interval includes zero, or CNN is better |
| RQ3 Confidence reliability | External ECE and Brier, after source-validation temperature scaling | H3: confidence on the external set exceeds accuracy (overconfidence), and external ECE exceeds source ECE. | External ECE not larger, or underconfidence |
| RQ4 Abstention | External selective risk and coverage at the frozen threshold, and AURC | H4: confidence separates errors from correct predictions better than chance on the external set, but less well than on the source test, and the frozen threshold does not deliver the validation-time risk. | Selective risk on the external set at the frozen threshold is no lower than the full-coverage risk |
| RQ5 Error patterns | Per-class recall and top confusions; high-confidence error counts | Exploratory. No hypothesis. | n/a |

Secondary analyses are **exploratory** and labelled as such in every table. They include per-class metrics, the equal-mass ECE variant, and Grad-CAM.

---

## 2. Datasets and provenance requirements

**FIXED.** Source dataset: PlantVillage. External dataset: PlantDoc. Neither may be used in any other role.

Provenance record, to be created in Phase 3 and stored in `docs/dataset_provenance.md` (proposed path): for each dataset, the exact distribution used, the repository commit hash or dataset revision, download date, the licence text as found in that distribution, citation, file counts, and a SHA-256 of the archive or manifest.

| Item | Status | Decision |
|---|---|---|
| PlantVillage distribution | **OPEN** | Candidate: Hugging Face `mohanty/PlantVillage`, colour configuration. It exposes `leaf_id`. Must confirm its images match the GitHub originals (hash comparison on a sample), and settle the licence conflict (HF card: `cc-by-sa-3.0`; GitHub README: none seen). |
| PlantVillage variant | **FIXED** | Colour images only. No grayscale or segmented variants. |
| PlantDoc distribution | **OPEN** | Candidates: cropped classification release `pratikkayal/PlantDoc-Dataset`, or object-detection release plus our own cropping. The deciding question is whether crops retain a source-photograph identifier (see 4.3). The derivative repository (2,585 images, 29 classes) is **not** the primary source. |
| PlantDoc portions used | **FIXED** | All available images, both original train and test folders, serve as the external **test** set. PlantDoc is never trained on, so its original split is irrelevant to us. Using it all maximises sample size. |
| Dataset counts | **AUDIT** | Counted from files. The specification's numbers are not used. |
| Minimum data | **AUDIT** | A class enters the primary class set only if it meets the minimum counts in 4.4. |

---

## 3. Licensing and redistribution rules

**FIXED.**
- Datasets and model weights are never committed to Git.
- Manifests may be committed only if they contain filenames, labels and hashes, not images. This is confirmed in Phase 3 against each licence.
- The PlantVillage licence question blocks any public release of derived artefacts (weights, hosted demo) until resolved. Weights trained on the data are not published until then.
- Citations to include in every public document: [PV-PAPER], [MOHANTY16], [PD-PAPER].
- PlantDoc is CC BY 4.0 in the repositories read. Attribution is required.

---

## 4. Class alignment and leakage prevention

### 4.1 Alignment rules (**FIXED**)

A PlantDoc label maps to a PlantVillage label only if **all** hold:

1. **Same crop species.**
2. **Same disease**, or both healthy, established by cited plant-pathology evidence recorded in the alignment table, **not by matching names**.
3. The PlantDoc label does not describe a symptom type or a group that is broader or narrower than the PlantVillage class (for example "leaf spot" versus a named pathogen). Doubt means exclusion.
4. The user has reviewed and signed off the table. I propose a mapping, but I am not a plant pathologist, and the user's review is the minimum bar.

The table has the columns in the specification (source dataset, original label, canonical label, crop, disease or healthy state, inclusion decision, reason) plus an `evidence` column and a `reviewer` column.

**Illustration only, not a mapping.** The derivative repository lists PlantDoc names such as "Bell Pepper Leaf Spot", "Corn Leaf Blight" and "Tomato Mold Leaf" [PD-STD]. Whether each corresponds to a specific PlantVillage class is exactly what rule 2 and 3 must decide. **No equivalence is assumed here.**

### 4.2 Primary class set (**AUDIT**)

The set K of canonical classes is the aligned classes passing the minimum counts, applied under the pre-model audit rules of 4.7 and frozen at the dataset freeze (4.8). Eligibility uses only labels, category compatibility, identifiers, duplicate groups and sample counts. It never uses any model output. The model output space is exactly K classes, for both models. Source images from PlantVillage classes outside K are excluded from training and from source evaluation. The number K is whatever the audit yields.

**Secondary setting (exploratory, declared now):** the strict setting of [TINY26], training on all 38 PlantVillage classes and scoring PlantDoc against its mapped labels. It is run only if time allows, is clearly labelled secondary, and never replaces the primary.

### 4.3 PlantDoc crop policy (**AUDIT**, rule **FIXED**)

- Every external example is a crop of one annotated leaf region or an unambiguous whole image, with an immutable record: `external_id`, `source_image_id`, original label, box coordinates, canonical label, inclusion flag and reason.
- A crop whose box carries a label with no valid mapping is excluded and logged. It is never relabelled.
- If one source photograph contains boxes of several classes, each crop keeps its own label. The `source_image_id` is shared.
- If the cropped release cannot be traced back to source photographs, I will use the detection release and crop it ourselves.
- Minimum crop size (**AUDIT**): crops below a pixel threshold are excluded. The threshold is chosen from the size distribution before any model runs.
- Aspect ratio (**AUDIT**): the preprocessing for non-square crops (squash, pad, or centre crop) is chosen from the aspect-ratio distribution and applied identically to every dataset and both models.

### 4.4 Minimum counts (**AUDIT**, values chosen before any model exists)

Proposed starting values, to be adjusted from the audit: a class needs at least 15 external crops (after deduplication) and at least 200 PlantVillage source images spread over at least 20 `leaf_id` groups to enter the primary set. Classes with fewer external crops are listed but excluded from primary aggregate metrics.

### 4.5 Source partitions (**FIXED**)

- Unit of splitting: `leaf_id` (PlantVillage). All images sharing a `leaf_id` go to one partition.
- Four partitions: **train 70%, val_sel 7.5%, val_cal 7.5%, test 15%**, stratified by class at the group level.
  - `val_sel`: early stopping, checkpoint choice, hyperparameter choice.
  - `val_cal`: temperature fitting and abstention threshold, and nothing else.
  - Splitting validation into two disjoint parts differs from the 70/15/15 outline in the specification. It adds a safeguard: a model chosen on `val_sel` has seen those images' loss, so fitting calibration on the same images would be mildly optimistic.
- Split seed: **42**. The split is generated once and shared by both models and all training seeds. Training seeds never change the split.
- Method: a group-aware stratified split (for example `StratifiedGroupKFold` or an equivalent group allocation). The exact method and the achieved class proportions are recorded.
- The test partition is touched once per final model, after all selection is finished.

### 4.6 Duplicate and overlap checks (**FIXED**, thresholds **AUDIT**)

Run in Phase 3 and 4, and asserted by automated tests:
1. Exact duplicates: SHA-256 across all images, within and across datasets.
2. Near duplicates: perceptual hash with a distance threshold (**AUDIT**) within PlantVillage, within PlantDoc, and **between** PlantVillage and PlantDoc.
3. Zero `leaf_id` overlap between partitions.
4. Any duplicate group is treated as one unit for splitting. Any PlantDoc image matching a PlantVillage image is flagged, and removed from the external set if it is a true copy.
5. Corrupt or unreadable files are listed and excluded.

### 4.7 External isolation and the pre-model audit (**FIXED**, amended A1.3)

The study has two stages with different rules.

**Stage 1: pre-model dataset audit (before any model exists).** The audit may inspect, for both datasets including PlantDoc: dataset labels, category compatibility with the alignment rules, duplicate groups (exact and perceptual hashes), source-image identifiers, box coordinates, image sizes and sample counts. It applies the fixed rules of 4.1 to 4.6 to decide the class alignment, which classes are included, the crop policy, the eligibility thresholds and the deduplication settings. All these decisions are recorded and frozen at the dataset freeze (4.8) **before any model is trained**. Using external labels and counts at this stage is permitted because no model has produced an output that could steer the choice.

**Stage 2: experiment (from the first training step).** External labels and external model outputs are isolated from model selection, hyperparameter tuning, calibration fitting, threshold selection and epoch selection. External data are used only for the final evaluation after the run freeze (9.4). The external evaluation script is written and tested on `val_cal` and `test` first.

**Prohibited at all times, in both stages:** no model prediction, external accuracy, external macro-F1, confidence score, calibration result or external risk-coverage result may influence the class mapping, inclusion decisions or eligibility thresholds. Architecture, hyperparameters, augmentation, temperature, abstention threshold, number of epochs and checkpoint are likewise never chosen using external data.

**If an audit error is found after training has begun** (for example a wrongly mapped label found by inspection), the fix is a dated amendment that states how it was found. The affected runs are re-run or clearly marked. The amendment must not be motivated by, or timed after looking at, any model output on the external set. If that cannot be shown, the affected claims are withdrawn.

### 4.8 Dataset freeze (**FIXED** procedure, values **AUDIT**; new, amendment A1.3)

Phase 3 may produce provisional audit results and proposed dataset decisions only. **The final dataset freeze is completed at the end of Phase 4** (amendment A2.1), because it must contain the source split manifest hash, which is generated in Phase 4. Only after the dataset audit, preprocessing, partition generation, manifest creation and leakage checks have passed are the following recorded in `configs/frozen/dataset_decisions_v1.json` (proposed path), with its SHA-256 in a sidecar file and a Git tag `dataset-freeze-v1`:
- The final class-alignment table, the class set K and its fixed ordering, and the evaluation class list for each evaluated set (see 7.1).
- The crop policy, minimum crop size and aspect-ratio policy.
- The minimum-count thresholds and the resulting inclusion and exclusion log.
- The exact-duplicate and perceptual-hash settings and the duplicate-group assignments.
- The PlantVillage and PlantDoc distributions used, with revisions and manifest hashes, and the licence findings.
- The source split manifest (seed 42) hash.
- The evaluation transform.

Training and evaluation scripts verify this file's hash with the same procedure as 9.4 and refuse to run if it is missing or invalid. No training may begin before the dataset freeze passes verification; Phase 5 starts only after it. The user's pathology sign-off on the mapping (rule 4 in 4.1) is a precondition of the dataset freeze.

---

## 5. Models

### 5.1 Common pipeline (**FIXED**)

Identical for both models: partitions, class set and ordering, input size 224×224, loss (cross-entropy without label smoothing, because label smoothing changes calibration), optimiser family AdamW, batch size, augmentation pipeline, early-stopping rule, selection criterion, and seeds.

- **Input normalisation:** ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] for both models. [TV-R18] confirms these for the pretrained weights. Using them for the scratch CNN keeps the input pipeline equal.
- **Training augmentation (applied to `train` only):** random resized crop (scale range **DEV**, starting at 0.5 to 1.0), horizontal and vertical flip, small rotation, mild colour jitter. No augmentation that imitates PlantDoc (for example background replacement) is allowed, because that would tune to the external domain. Such work is future work.
- **Evaluation transform:** deterministic resize to the chosen size using the crop policy in 4.3. The torchvision default for ResNet-18 is resize 256 then centre crop 224 [TV-R18]. Whether to follow it or use a full-image resize is decided once, in **AUDIT**, and applied to both models and all datasets.
- **Mixed precision** is allowed and recorded.

### 5.2 Model A: scratch CNN (**DEV**, constraints **FIXED**)

A convolutional network with batch normalisation, ReLU, pooling, global average pooling and a dropout-regularised linear head. The exact depth and widths are fixed at the end of Phase 5's smoke test and **before** the first full run. Constraint, stated to prevent a strawman: its capacity is chosen so that it can fit the training set well, and the parameter count is recorded and reported next to ResNet-18's 11,689,512 [TV-R18]. The CNN is never weakened to flatter the transfer model. If a deeper CNN gives a better `val_sel` result, it is used.

### 5.3 Model B: ResNet-18 (**FIXED** weights, **DEV** schedule)

- Weights: `ResNet18_Weights.IMAGENET1K_V1` (ImageNet-1K acc@1 69.758) [TV-R18]. The torchvision version is recorded.
- Head: final linear layer replaced with K outputs.
- Strategy: full fine-tuning of all layers is the primary strategy. Learning rate and any warm-up of the head are chosen on `val_sel` (**DEV**). A frozen-backbone variant is allowed only as a declared secondary experiment.
- BatchNorm layers: train mode during fine-tuning; recorded.

### 5.4 Hyperparameter search (**DEV**)

Equal budget for both models: the same small number of configurations each (proposed: 3 learning rates, one seed), selected by `val_sel` loss. Search results, including losers, are logged. No search uses `val_cal`, `test` or external data.

### 5.5 Training and model selection (**FIXED** rules)

- Checkpoint criterion: lowest `val_sel` negative log-likelihood. Patience-based early stopping with a documented patience (proposed 5 epochs) and a maximum epoch count (**DEV**).
- The test partition is never used to choose a checkpoint, hyperparameter or model.
- Which model "wins" is not decided at all; both are always reported.

### 5.6 Seeds (**FIXED**)

Training seeds 1, 2, 3 are the target. The split seed is 42 and separate. If resources allow fewer than three completed runs per model, the actual number is reported and no result is described as repeated beyond what ran. Seeds control initialisation, data order and augmentation. Bit-exact reproducibility across different GPUs is not promised. Software versions and GPU model are logged per run.

---

## 6. Metrics

All metrics are computed on the post-temperature probabilities unless stated, and on pre-temperature probabilities as a labelled comparison.

### 6.1 Classification (**FIXED**)

- Accuracy.
- Macro-F1 over the **frozen evaluation class list** for the evaluated set (7.1). The list is fixed at the dataset freeze and does not change between the point estimate and any bootstrap replicate. The number of classes entering the average is printed in every table. The zero-division rule is in 7.1.
- Per-class precision, recall and support. Per-class values for classes below the minimum support are shown greyed or omitted from interpretation.
- Confusion matrices (counts, and row-normalised).
- Gaps, in percentage points: `metric(source test) − metric(external)`, always shown with both values.

### 6.2 Calibration (**FIXED**)

- **ECE** with confidence = maximum predicted probability, L1 norm. **Primary: 15 equal-width bins**, the common convention [GUO17][ROELOFS]. **Secondary: 15 equal-mass bins** [NIXON19]. Bin counts are shown in the reliability diagrams. Because binned ECE depends on bin count [ROELOFS], a sensitivity table with fewer and more bins is produced.
- **Multiclass Brier score**, sum of squared differences between the predicted vector and the one-hot label, averaged over examples. **This convention ranges over [0, 2].** The convention is stated beside every Brier number, because the other convention (mean over classes) is smaller, and published numbers cannot be compared without checking it.
- **Negative log-likelihood**, a binning-free measure.
- **Reliability diagrams** with a confidence histogram.
- Mean confidence minus accuracy (signed overconfidence).

### 6.3 Temperature scaling (**FIXED**)

- A single scalar T > 0 per model and per seed, applied to logits before softmax.
- Fitted by minimising NLL on `val_cal` only (optimisation in log T, bounded). T, the optimiser and the `val_cal` NLL before and after are recorded.
- No temperature is ever fitted on `test`, external data, or anything containing external labels. Source-validation temperature is applied unchanged to the external set. This corresponds to "Setting A" in [REL26]. Target-aware calibration (its Setting B, which fits temperature on a PlantDoc subset) is **declined** for this study (OPEN-8, resolved by amendment A1.1) and is not part of the approved scope.

### 6.4 Abstention (**FIXED** definitions)

- Confidence score: maximum temperature-scaled softmax probability.
- Accept an example if score ≥ τ; otherwise abstain.
- **Coverage** = accepted / total. **Selective risk** = errors / accepted. Both are always reported together with counts (accepted n, rejected n).
- If no example is accepted, selective risk is undefined and is reported as such, never as zero.
- **Risk–coverage curve:** computed by sweeping τ over all scores, on `val_cal` (labelled as a development diagnostic, see 6.6), `test` and external.
- **AURC** (area under the risk–coverage curve), the **random-rejection reference** (risk equals the full-coverage risk at every coverage level), and the **oracle reference**.
- **AUROC for error detection (A2.3).** Positive label: the prediction is incorrect. Detection score: `1 - max_class_probability` (computed from the temperature-scaled probabilities), so a higher score means greater uncertainty. The result is the area under the ROC curve of that score against that label, reported as "AUROC for error detection". Tied scores receive the average rank (the Mann-Whitney convention, equivalent to counting ties as one half). It is undefined, and reported as such, when all predictions are correct or all are incorrect.

### 6.5 Threshold selection (**FIXED** rules, value computed in Phase 8)

τ is computed from `val_cal` only, before external inference, separately for each model and seed.

- **Primary rule R-cov:** τ is the confidence quantile at which `val_cal` coverage is 90%.
- **Secondary rule R-risk:** the smallest τ whose `val_cal` selective risk is at most 1%. If `val_cal` accuracy is so high that R-risk accepts nearly everything, that is reported as a finding, not tuned away.

The numbers 90% and 1% are fixed now, so they cannot be picked after seeing external results. They are arbitrary operating points, and the full curve is always shown beside them.

### 6.6 Reporting of `val_cal` metrics (**FIXED**, amendment A1.6)

`val_cal` is used for temperature fitting and abstention-threshold selection. Every metric measured on `val_cal` (ECE, Brier, NLL, coverage, selective risk, risk-coverage curve, AURC) is therefore labelled **"development / threshold-selection diagnostic"** in every table and figure. It is shown separately from the test and external results and is **not** interpreted as an independent test estimate. For example, `val_cal` coverage at the R-cov threshold is 90% by construction, and `val_cal` ECE after temperature scaling is optimistic by construction. The untouched source test and the external set carry the final evaluation. No additional split is added: `val_cal` is already disjoint from `train`, `val_sel` and `test`, and a further split has no methodological gain.

---

## 7. Statistical uncertainty

### 7.1 Cluster bootstrap (**FIXED**)

- Resampling unit (preserved, amendment A1.5): `leaf_id` for the source test; for the external set, `source_image_id`, or the duplicate-group identifier where a near-duplicate group links several source images. Operationally the external cluster is the connected component of the union of `source_image_id` and duplicate-group links, computed at the dataset freeze and stored in the external manifest. Individual crops or images are never resampled independently, because they are correlated.
- **Frozen evaluation class list (A1.5).** For each evaluated set (source test, external) the list of classes entering macro-F1 is fixed at the dataset freeze and stored in `dataset_decisions_v1.json`. Under the minimum-count rule of 4.4 every class in K has at least 15 external crops and enough source images, so the list is expected to be K for both sets (AUDIT: confirmed at the freeze). The same list is used for the point estimate and for every bootstrap replicate. A replicate never adds or drops a class.
- **Zero-division rule (A1.5).** For each class in the list, precision is 0 if the class has no predictions, recall is 0 if it has no true examples, and F1 is 0 if precision plus recall is 0 (equivalent to `zero_division=0`). Consequently a class that is absent from a replicate's true labels and also never predicted contributes F1 = 0 and still counts in the denominator, which stays fixed at the list length. This biases bootstrap macro-F1 slightly downward when classes are rare. The code records, per replicate, the number of list classes with no true examples, and the report states the mean and maximum.
- **Sensitivity variant (exploratory, labelled).** Macro-F1 over only the classes present in each replicate may be shown beside the primary, clearly labelled, to bound the effect of the rule. It never replaces the primary.
- B = 2,000 resamples for development and 10,000 for final tables if cheap. Percentile 95% intervals.
- Gap intervals: resample the source and external sets independently and difference the metrics.
- Model comparison intervals: paired bootstrap on the same resampled clusters.
- Applies to accuracy, macro-F1, ECE, Brier, selective risk, coverage and AURC.

### 7.2 Seed variability (**FIXED**)

Reported separately from sampling uncertainty. With three seeds, show every individual seed's value and the range, not a standard deviation of three numbers. Bootstrap intervals are computed per seed. A conclusion is drawn only when per-seed results agree. Softmax outputs are **not** averaged across seeds, since that would create an ensemble and change the models under study.

### 7.3 Multiplicity and interpretation (**FIXED**)

The primary endpoints are those in section 1. Everything else is exploratory and labelled so. No p-values are claimed from three seeds. Differences are described as supported only when the paired interval excludes zero and every seed agrees in sign.

---

## 8. Compute plan (Google Colab, **FIXED** unless noted)

Facts from the official FAQ [COLAB-FAQ]: resources are not guaranteed; GPU type, idle timeout and VM lifetime vary and are unpublished; free notebooks run at most 12 hours depending on availability; data should be copied to the VM as an archive rather than read as many small Drive files; free runtimes may not be used to serve a web UI. The earlier T4 diagnostic describes one session only.

Rules:
1. **Session start check (every session):** record `nvidia-smi` output, Python, PyTorch, torchvision, CUDA, and the Git commit hash into the run log. If no GPU is assigned, do not start training. Evaluation and calibration run on CPU.
2. **Data staging:** keep a single archive per dataset on Drive. Copy it to the VM's local disk and extract there. Training reads local files only. Manifests and checkpoints live on Drive.
3. **Checkpointing:** save `last` every epoch and `best` on improvement, atomically (write to a temporary file, then rename). Store model, optimiser, scheduler, scaler, epoch, the best metric and RNG states. Each run has an experiment id and its own folder.
4. **Resume:** a run restarts from `last` with identical config. A config hash is stored, and resume is refused on mismatch.
5. **Smoke test before any full run:** a few batches or a tiny subset for each model, covering the data loader, mixed precision, checkpoint and resume, and metric code. It also measures throughput. **No time or epoch budget is assumed until then.**
6. **No secrets in notebooks.** No tokens in Git. Notebooks are thin: they clone the repository, install dependencies, and call scripts in `src`. The pre-installed PyTorch is not reinstalled unless required, and versions are logged.
7. **Evaluation and plotting** run from saved logits and manifests, not from live models, so they need no GPU.
8. **The Streamlit demo does not run on Colab** (see above).

---

## 9. Artefacts and freezing

### 9.1 Per-run record (**FIXED**)

`config.yaml`, config hash, seed, split-manifest hash, class-mapping hash, software and hardware versions, git commit, per-epoch metrics, `best` and `last` checkpoints.

### 9.2 Saved outputs (**FIXED**)

For each model, seed and partition (`val_sel`, `val_cal`, `test`, external): the logits, labels and example ids as an array file plus a manifest. Every metric in the report is computed from these files by code in the repository.

### 9.3 Manifests (**FIXED**)

Source manifest, external manifest with crop relations, class-alignment table, and the exclusion log. All are hashed.

### 9.4 Run freeze file (**FIXED**, amended A1.4)

**Roles, kept distinct.**
- **Code commit hash (`code_commit`).** The Git commit of the implementation snapshot used to train the model and to compute T and tau. It is recorded *inside* the freeze file. It identifies the code, and is never the commit that contains the freeze file, which cannot be known in advance and would be circular.
- **Freeze file content.** `configs/frozen/<run_id>.json` holds: `code_commit`, run id, seed, T, tau for both rules, the checkpoint SHA-256, the class list and evaluation class lists, the config hash, the manifest hashes, `dataset_decisions_v1.json`'s SHA-256, and library and hardware versions. It contains no hash of itself.
- **Freeze file SHA-256 (`freeze_sha256`).** The SHA-256 of the file's exact bytes (UTF-8, sorted keys, LF line endings, written once and never reformatted). It is stored in the sidecar `configs/frozen/<run_id>.sha256` and also copied into the run log on Drive.
- **Freeze commit.** The Git commit that stores the freeze file and its sidecar. It is a later commit than `code_commit`. It is tagged `freeze-<run_id>`.

**Procedure.**
1. Training and selection finish. T and tau are computed from `val_cal` only, using code at `code_commit`.
2. The freeze file is written and its SHA-256 computed. The sidecar is written. Both are committed (the freeze commit) and tagged.
3. Before any external inference, an independent verifier (`tools/verify_freeze.py`, with the same check available as plain `sha256sum -c`) recomputes the SHA-256 of the file and compares it against **both** the sidecar and a value passed on the command line from the run log.
4. It also runs two **independent** Git protections (amendment A2.2):
   - **(a) Commit-to-HEAD comparison.** `git diff --exit-code <code_commit> HEAD -- <code paths>` over the tracked source and run-configuration paths (`src/`, `scripts/`, `tools/`, `configs/` excluding `configs/frozen/`, and the dependency specification). Any difference, including a change committed after `code_commit`, fails. The freeze file and its sidecar live in `configs/frozen/` and are excluded from this comparison because they are created after `code_commit`; their integrity is verified separately by the hash checks in step 3. `code_commit` must also be an ancestor of `HEAD`.
   - **(b) Working-tree check.** `git diff --exit-code` (unstaged) and `git diff --cached --exit-code` (staged) over the same code paths must both be clean.
   - **A clean working tree alone is never accepted as proof that the code has not changed since `code_commit`.** Check (a) is mandatory and independent of (b).
   - The verifier also checks that the freeze file is tracked, that the checkpoint hash and the source and external manifest hashes match, and that the dataset-freeze hash matches.
5. **Fail closed.** If the file or sidecar is missing, a hash differs, a field is absent or malformed, or any check fails, the verifier exits with a non-zero status and the external evaluation script does not load any external data or logits. The script calls the verifier itself rather than trusting a flag.

**Limits, stated honestly.** This guards against accidental or undeclared change and makes a deliberate change visible. It cannot stop someone who edits the file, the sidecar, the log and the verifier together. The Git history and tags are the evidence trail.

---

## 10. Tests (checked in Phase 4 and 11)

Automated tests must cover: no `leaf_id` overlap across partitions; reproducibility of the split from seed 42; class mapping totality and the absence of unmapped labels in the model's output space; exact duplicates not crossing partitions; ECE, Brier and risk–coverage computed on small hand-calculated cases; temperature scaling leaves argmax unchanged; threshold selection reads only `val_cal`; refusal to run external evaluation without a valid freeze file, tested with: a missing file, a missing sidecar, a one-byte change, a wrong command-line hash, code changed since `code_commit` (committed later, staged, or unstaged), a changed checkpoint, a changed source or external manifest, malformed or missing freeze fields, and a dataset-freeze mismatch; refusal to train without a valid dataset freeze; bootstrap macro-F1 using the same class list in every replicate, with hand-calculated cases for absent classes; external cluster components merging `source_image_id` and duplicate links; val_cal metrics carrying the development-diagnostic label; and AUROC for error detection on synthetic cases with known rankings and tied scores.

---

## 11. Prohibited claims

Not permitted anywhere in documents, README, portfolio or demo text unless evidence supports them:
- Novelty at the level of the problem, datasets, temperature scaling or selective prediction (see literature review, section 6).
- That the models can diagnose plant disease in the field, or any production, agronomic or business readiness.
- That confidence guarantees correctness, or that abstention removes dangerous errors.
- That Grad-CAM shows biological validity or causal evidence.
- That a PlantVillage background shortcut explains the gap, unless a dedicated test supports it.
- Any number not produced by repository code from saved outputs.
- "Three seeds" if fewer completed.
- Treatment, pesticide or dosage advice.
- Presenting `val_cal` metrics as independent test estimates (6.6).
- Comparing this study's numbers to [REL26] without naming its setting. Only its Setting A is comparable in kind, and Setting B (target-calibrated) must never be presented as a like-for-like baseline.
- Describing this study as having investigated target-aware calibration.

---

## 12. Reporting rules

Every results table shows: n (examples and clusters), number of classes averaged, both source and external values for gaps, coverage with selective risk, interval type, and number of seeds. Poor, null or contradictory results are reported with the same prominence as favourable ones.

---

## 13. Open questions (**OPEN**)

| ID | Question | Needed by |
|---|---|---|
| OPEN-1 | PlantVillage licence: reconcile the HF `cc-by-sa-3.0` metadata with the original distribution | Phase 3 (before any publication of derived artefacts) |
| OPEN-2 | PlantVillage distribution to use, and whether `leaf_id` matches the GitHub originals | Phase 3 |
| OPEN-3 | PlantDoc distribution (cropped versus detection plus own crops) and whether crops keep source-photograph ids | Phase 3 |
| OPEN-4 | Final class mapping and the user's pathology review | Phase 3 |
| OPEN-5 | Minimum counts, minimum crop size, near-duplicate hash threshold, aspect-ratio policy | Phase 3 (data properties only) |
| OPEN-6 | **Resolved (A1.2), with a residual:** full texts accessed. Before public use, re-check every quoted number against the articles' tables, [REL26]'s Brier convention, and the [TINY26] identifier/date inconsistency (recorded, not resolved) | Before any public mention |
| OPEN-7 | Target completion date: unspecified | User |
| OPEN-8 | **Resolved: DECLINED (A1.1).** No target-aware calibration in the current study. Reason: the published [REL26] already investigates target-aware calibration on a parent-image-disjoint PlantDoc calibration subset. This study remains a source-only-calibration replication-style comparison of a scratch CNN and ResNet-18. A future target-aware extension would be a separate study and is not in the approved scope. | Closed |
| OPEN-9 | Hyperparameter budget and epoch limits | Phase 5 smoke test |
| OPEN-10 | Number of completed seeds | Compute-dependent |

---

## 14. Amendment log

| Date | Version | Change | Reason | Informed by any test or external result? |
|---|---|---|---|---|
| 2026-10-11 | 0.9 | Initial draft | n/a | No (nothing executed) |
| 2026-10-11 | 1.0 | **A1.1** OPEN-8 resolved: target-aware calibration declined; section 6.3, section 11 and OPEN-8 updated | [REL26] already studies target-aware calibration; keeps the study source-only and the isolation rule intact | No |
| 2026-10-11 | 1.0 | **A1.2** [REL26] and [TINY26] references upgraded to verified; OPEN-6 closed with a residual re-check; Setting A and Setting B distinguished in all citations | Primary sources accessed (user, and a summarised fetch by Claude) | No |
| 2026-10-11 | 1.0 | **A1.3** Sections 4.2 and 4.7 rewritten to separate the pre-model audit from the experiment; new section 4.8 dataset freeze | Contradiction: eligibility used external counts while 4.7 forbade external influence | No |
| 2026-10-11 | 1.0 | **A1.4** Section 9.4 rewritten: roles of `code_commit`, freeze content, freeze SHA-256 and freeze commit; verification procedure; fail-closed behaviour | Avoid circular commit hash; make verification reproducible | No |
| 2026-10-11 | 1.0 | **A1.5** Bootstrap macro-F1: frozen evaluation class list, zero-division rule and replicate handling; cluster units preserved (7.1, 6.1) | Class set must not vary between replicates | No |
| 2026-10-11 | 1.0 | **A1.6** `val_cal` metrics labelled development / threshold-selection diagnostics (new 6.6); no new split | `val_cal` is used for fitting, so it is not an independent estimate | No |
| 2026-10-11 | 1.0 | **A1.7** Status changed to approved for implementation; the earlier rule that v1.0 required filled AUDIT items is replaced by the two-freeze structure | User decision; AUDIT items stay unresolved until Phase 3 | No |
| 2026-10-11 | 1.0 | **A2.1** Dataset-freeze sequencing: Phase 3 produces provisional audit results and proposed decisions; the final dataset freeze (`configs/frozen/dataset_decisions_v1.json`) is completed and tagged at the end of Phase 4, after preprocessing, partition generation, manifests and leakage checks. No training before it passes verification. Sections header, 4.8 | The freeze must contain the source split manifest hash, which only exists after Phase 4 | No |
| 2026-10-11 | 1.0 | **A2.2** Freeze-verifier Git checks (9.4 step 4): (a) `git diff --exit-code <code_commit> HEAD -- <paths>` and (b) a separate staged and unstaged working-tree check; `configs/frozen/` excluded from (a) and verified by hash; clean tree never sufficient | A clean tree cannot detect changes committed after `code_commit` | No |
| 2026-10-11 | 1.0 | **A2.3** AUROC for error detection defined (6.4): positive = incorrect prediction, score = 1 - max class probability, ties averaged | Remove ambiguity before implementation | No |

---

## 15. Phase 1 gate checklist (updated after the user's review)

- [x] Literature review written with synthesis (`literature_review.md`)
- [x] Reference list with per-source verification status (`references.md`)
- [x] Protocol states which decisions are fixed, audit-dependent, development-time and open
- [x] External isolation rules are explicit and consistent (4.7, 4.8, 6.3, 6.5, 9.4)
- [x] OPEN-8 resolved: target-aware calibration declined (A1.1)
- [x] [REL26] and [TINY26] references verified and updated, Setting A and Setting B distinguished (A1.2)
- [x] Pre-model audit versus experiment distinction documented; dataset freeze procedure defined (A1.3)
- [x] Freeze-file roles and fail-closed verification defined (A1.4)
- [x] Bootstrap macro-F1 semantics fixed (A1.5)
- [x] `val_cal` reporting labelled (A1.6)
- [x] Protocol v1.0 approved for implementation (A1.7)
- [ ] Not done, by design: PlantVillage licence, dataset distributions, class mapping, crop policy, minimum counts and deduplication settings (Phase 3 AUDIT items)
- [ ] Not done, by design: any implementation, dataset download or training
- [ ] Residual: re-check quoted [REL26]/[TINY26] numbers against the articles before public use
