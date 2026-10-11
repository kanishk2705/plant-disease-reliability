# Live project record

Last updated 2026-10-11 (end of Phase 2 work). Everything not listed as "executed" is unverified.

## Phase status

| Phase | Status |
|---|---|
| 0 Environment | Closed. Colab verified once: Python 3.13.15, PyTorch 2.11.0+cu130, CUDA available, one Tesla T4. Colab is the primary GPU environment; laptop for development, Git, docs, light tests; Drive for persistent artefacts; GitHub for code; AWS deferred; target date unspecified. |
| 1 Literature and protocol | Approved after amendments. Protocol v1.0 approved for implementation; AUDIT items unresolved. OPEN-8 declined. REL26 full article and TINY26 verified (TINY26 identifier/date inconsistency recorded). |
| 2 Repository and environment | Utilities, tests and documentation written and executed in the sandbox. Colab notebook written, **not executed**. Awaiting user review. |
| 3 Dataset audit | Executed in the sandbox from GitHub clones (PlantVillage 54,305 images, PlantDoc cropped 2,578, PlantDoc detection 2,603). Provisional decisions proposed; **nothing frozen**. See `docs/dataset_provenance.md` and `docs/audit/phase3_audit_report.md`. Pending: PlantVillage licence, pathology sign-off with source verification. |
| 4 to 12 | Not started. |

## Repository evidence

| Item | Value |
|---|---|
| Protocol commit | `bbd045ef50da4bfe8b8f7ae31bc70ab8e85f7847` |
| Tag `protocol-v1.0` | annotated tag object `e6b3f56e581c058124f9f7ecf400bc88078c7e71`, points to the protocol commit |
| Implementation commit | `3b31067` (full hash in `git log`) |
| Remote | none configured; nothing pushed |
| Tests | 218 passed, 0 failed, 0 skipped, in the working copy and again in a fresh clone (sandbox: Python 3.13.16, torch 2.14.1 CPU) |

Details, blocked tests and remaining implementation work: `docs/ENVIRONMENT.md`.

## Amendments since v1.0 was approved

A2.1 dataset freeze at end of Phase 4; A2.2 two independent Git protections in the verifier; A2.3 AUROC definition. All in `docs/research/experimental_protocol.md`, section 14.

## Unresolved

- PlantVillage licence (HF `cc-by-sa-3.0` versus none seen in the GitHub README).
- Dataset distributions and revisions; PlantDoc source-image identifiers and crop traceability.
- Class mapping and the user's pathology review; crop and aspect-ratio policy; minimum counts; deduplication settings.
- Hyperparameter budget and epoch limits (Phase 5); number of completed seeds; target date.
- Re-check quoted REL26/TINY26 numbers before public use; BARBEDO18, NAEINI15, BRIER50, RESNET not read.
- GitHub remote and Colab notebook execution; compatibility with Colab's torch 2.11.0 unverified.

## Next action

Phase 4: preprocessing, leaf-group-aware splits, manifests, leakage checks and the dataset freeze. Training needs a GPU, which the sandbox does not have (2 CPUs, no GPU, no access to Colab).

## Amendment A3.1 (post-freeze code fix, 2026-10-11)
The first external-evaluation attempt (`final_cnn_s1`) verified its run freeze and computed external logits, then crashed while writing
its report JSON because numpy integers/floats were not serialisable. No external metric had been printed or inspected. The bug was a
serialisation defect only. `report.to_jsonable` and a test were added; no metric, threshold, calibration or model code changed.
Because the fix touched frozen code paths, the six run freezes were re-issued with the new `code_commit`. T, tau, checkpoint hashes,
config hashes, class lists and manifest hashes are byte-identical to the first freezes (verified field by field). The earlier
freeze commit is `0344fa4` and remains in history.
