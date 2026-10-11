# Literature Review: Cross-Dataset Plant-Disease Classification, Calibration and Selective Prediction

Phase 1 deliverable. Compiled 2026-10-11. Amended 2026-10-11 (protocol amendment A1: [REL26] and [TINY26] primary-source access). Citation keys such as [GUO17] refer to `references.md`, which also records how much of each source was actually read.

## 0. Scope and method (read this first)

This is a targeted review, not a systematic one. It rests on roughly two dozen web searches and page fetches in one session. Three limits follow:

- Most papers were seen at abstract or excerpt level only. `references.md` marks each one.
- The most relevant 2026 paper, [REL26], was first seen only as a search excerpt. After amendment A1 the full published article is verified (the user accessed it; a summarised fetch of the publisher page agreed). Numbers quoted from it should be re-checked against its tables before public use. [TINY26] was verified at the arXiv abstract page by me and in full HTML by the user.
- I did not search exhaustively, so failing to find a prior study is not evidence that none exists.

Statements below are tagged **[V]** (read in a primary source), **[E]** (excerpt or abstract only), or **[I]** (my inference, not a source claim).

---

## 1. The lab-to-field gap is documented from the start

PlantVillage supplied the data for the most-cited early deep-learning result in this area. Mohanty et al. report 99.35% accuracy on a held-out test set from images collected under controlled conditions [V, abstract] [MOHANTY16]. In the same abstract they report only 31.4% accuracy on images from trusted online sources taken under different conditions, against 2.6% for random selection, and they state that more diverse training data is needed [V, abstract].

So the central premise of this project, that high in-dataset accuracy does not establish field reliability, was stated by the dataset's own benchmark paper in 2016. It is not a new observation.

**Implication [I]:** the project cannot present "accuracy drops on field images" as a finding of interest by itself. What it can add is a careful measurement of how the confidence behaves across that drop.

## 2. Dataset properties that matter for this study

### 2.1 PlantVillage

- **Size and structure.** The GitHub README states 54,306 images, 14 crop species and 26 diseases [V] [PV-REPO]. Other sources give 54,303 (TensorFlow Datasets, 38 categories) [PV-TFDS], 54,305 (the Hugging Face split totals, and the TinyCNN paper's merged count) [V][E], and 54,309 (a third-party summary) [PV-PAPER]. **The count depends on the distribution.** The exact count must be recorded from the files actually used.
- **Controlled capture.** The original paper describes images collected under controlled conditions [MOHANTY16].
- **Multiple images per leaf.** The Hugging Face card exposes a `leaf_id` and warns that several images often show the same physical leaf, so splitting across `leaf_id` biases evaluation [V] [PV-HF]. This is the key leakage risk. Random per-image splits, which many papers use, can place views of one leaf on both sides.
- **Background shortcut.** A random forest trained on 8 background pixels reached 49.0% test accuracy against 2.6% random [V, abstract] [NOYAN22]. The model can exploit acquisition artefacts that correlate with labels. This is a plausible mechanism for poor transfer, though it does not prove that mechanism operates in our models [I].
- **Licence unresolved.** The Hugging Face card lists `cc-by-sa-3.0` [V]. The GitHub README text I read states none [V]. This must be settled in Phase 3 (see the protocol, section 3.1).

### 2.2 PlantDoc

- **Size.** The paper's abstract says 2,598 data points, 13 plant species and "up to 17 classes of diseases" [V]. An arXiv HTML excerpt says "27 classes (17-10, disease-healthy)" [E]. The standardized derivative repository says 2,585 images and 29 classes [V] [PD-STD]. Roboflow says 2,569 images and 30 classes [PD-RF].
- **The project specification's "2,585 images, 29 classes" comes from the derivative, not the paper.** Its own split table sums to 2,481 images, and it notes that some images are missing from the archive [V]. The correct numbers for this study are whatever the audit counts in the files used.
- **Two distributions exist.** The original authors host a cropped classification version [PD-REPO-CROP] and an object-detection version with CSV box labels [PD-REPO-DET], both under CC BY 4.0 [V].
- **Collection.** Images were scraped from the internet and annotated by hand [V, abstract]. Web-scraped images have unknown cameras, compression histories and possibly duplicated sources [I].
- **A detection-style dataset is not a classification benchmark.** This is why the specification demands a documented crop policy. It is the right call.

### 2.3 Related evidence on partitioning

Other work raises the same concerns. DDD argues that performance is overestimated when train and test sets come from the same domain [E] [ARIMA25]. A 2026 grape-leaf benchmark reports exact image overlap among several controlled datasets and near-perfect within-dataset accuracy, and concludes that within-dataset evaluation discriminates poorly between modern architectures [E] [GRAPE26]. Both support the protocol's duplicate and cross-dataset overlap checks, but I have only read excerpts.

## 3. Calibration and temperature scaling

**Metric and method.** Guo et al. define ECE, find that modern networks are poorly calibrated, and report that temperature scaling, a single-parameter variant of Platt scaling, is effective on most datasets they tested [V, abstract] [GUO17]. Temperature scaling does not change the predicted class, only the confidence, so it cannot change accuracy [I, follows from the method].

**Limits under shift.** Ovadia et al. conclude that traditional post-hoc calibration falls short under dataset shift, and that post-hoc calibration gives good results on i.i.d. data but can fail under even mild input shift [V, abstract and excerpt] [OVADIA19]. This is the direct argument for the project's design. A temperature fitted on source-domain validation data encodes source-domain confidence behaviour [I].

**Architecture effects.** Minderer et al. find that architecture is a major determinant of calibration and that the decay of calibration under shift is less pronounced in recent architectures [V, abstract] [MINDERER21]. Their models are large modern image classifiers, so extrapolating to a small CNN and ResNet-18 is speculative [I]. It does mean model comparisons in this project should not assume a direction.

**The metric itself is fragile.** Nixon et al. argue that ECE has numerous flaws and that its conclusions depend on measurement choices; they recommend adaptive binning for stable rankings [V, abstract] [NIXON19]. An excerpt of [ROELOFS] says equal-width ECE with 15 bins is the most common convention, that binned ECE rises with bin count, and that equal-mass binning is less biased [E]. **Implication [I]:** report ECE under at least two binning schemes, publish bin counts, and add binning-free measures (NLL, Brier). The external set will be small, so ECE there will be noisy and biased.

## 4. Selective classification

Geifman and El-Yaniv define coverage, selective risk and the risk–coverage curve for deep networks, and show that a simple softmax-response confidence can support effective selective classification on CIFAR and ImageNet [V, abstract and excerpts] [GEIFMAN17]. Hendrycks and Gimpel proposed maximum softmax probability as a baseline for detecting misclassified and out-of-distribution examples, and noted that the baseline can be surpassed [V, abstract] [HENDRYCKS17]. SelectiveNet integrates the reject option into the architecture [V, abstract] [GEIFMAN19]. The project uses none of the latter; it is context.

**Implication [I].** Both the ordinary-accuracy and the selective-risk results in those papers come from in-distribution test sets. Whether a source-validation-chosen threshold still separates errors from correct predictions on a shifted domain is exactly the open question the project examines. No result in the cited abstracts answers it for plant images.

## 5. Statistical uncertainty

I did not find, and did not read, a plant-disease paper that reports bootstrap intervals for cross-dataset gaps. [I] The reliable statement is narrower: the PlantDoc test portions can be small (the TinyCNN paper uses a 252-image PlantDoc test subset across 27 mapped classes [V, per user] [TINY26]; [REL26] uses 2,081 matched known-class crops across 21 classes in Setting A [V] and only 199 calibration crops in Setting B), so per-class and calibration estimates will be unstable. Intervals must be reported, and they must resample at the level of the source photograph or physical leaf, because crops from one photograph are not independent [I].

## 6. Closest prior work and the novelty question

Three 2026 items overlap with this project.

| Item | What it appears to do | Overlap | Evidence level |
|---|---|---|---|
| [REL26] | Xiang, Shi and Zhu, *Frontiers in Plant Science*, 22 May 2026. PlantVillage-trained ResNet-50 (21 known classes) evaluated on PlantDoc; temperature scaling, selective prediction, Grad-CAM, frozen-encoder comparison. **Setting A (source-only):** T = 1.18 fitted on the PlantVillage validation split; 2,081 matched PlantDoc crops; accuracy 32.05%, macro-F1 0.2853, ECE 0.4771 before and 0.4400 after, Brier 1.1104 before and 1.0728 after; only 5.1% of samples answered at a nominal 80% coverage target. **Setting B (target-calibrated):** 10% of PlantDoc known crops (199) used for calibration, grouped by reconstructed parent image, 1,882 disjoint test crops, 10 repeats; T = 1.54 for the representative split; ECE 0.4761 before and 0.3651 after; mean selective risk 0.6430 at nominal 80% coverage | **Very high**: same datasets, same questions (calibration, selective prediction, shift) | Full article [V] (user access plus summarised fetch); re-check numbers against its tables |
| [TINY26] | Ho-Lam and Nguyen, arXiv:2609.20290 (2026; identifier/date inconsistency noted in `references.md`). 193K-parameter CNN; 98.88% accuracy and 98.03% macro-F1 on 38-class PlantVillage; PlantDoc test subset of 252 images across 27 mapped classes; stratified 70:15:15 PlantVillage split; Grad-CAM attributes the gap to background attention | High: PlantVillage to PlantDoc with a small CNN | Abstract page [V] by me; full HTML [V] by user |
| [FRAME26] | Focused review proposing PlantVillage and PlantDoc as controlled and field-like datasets with semantically matched classes and leakage-resistant partitioning | High on design, no experiments reported in the excerpt | Excerpt only; preprint |

[MRC26] (conformal prediction for plant disease) is related to reliability but not to cross-dataset evaluation [E].

**Assessment.**

1. The problem, both datasets, temperature scaling and selective prediction have all been used together in recent work. The specification's own caution that "cross-dataset plant-disease classification, confidence calibration and selective prediction have already been studied" is **confirmed**.
2. **I do not claim novelty at the level of problem or method.** The project is a transparent, independently reproducible study.
3. Differences from [REL26], now stated from the full article: this protocol (a) compares a scratch CNN and ResNet-18 under one shared pipeline rather than studying a fine-tuned ResNet-50 (plus frozen encoders), (b) forbids any PlantDoc labels in calibration, which corresponds to its Setting A only, and (c) separates source validation into disjoint selection and calibration parts. Whether these differences matter scientifically is untested, and other work I did not find may already cover them.
   - **Why target-aware calibration is out of scope (OPEN-8, declined, amendment A1).** [REL26] already investigates target-aware calibration (its Setting B) on a parent-image-disjoint PlantDoc calibration subset. Repeating it would add little and would put external labels into the pipeline. The current study stays a source-only-calibration replication-style comparison. A target-aware extension may be considered separately later and is not part of the approved scope.
   - **Comparability.** Only [REL26] Setting A values are comparable in kind to this study's primary results, and even then the class sets, models and Brier convention (not stated in the text read) may differ. Any comparison must name the setting and the metric convention.
4. [I] A useful reading is that this project is closest to an independent replication with a different architecture pair. That is a legitimate contribution, but it must be described as that.

**Dates.** Publication dates and research dates differ. [MOHANTY16] was posted to arXiv in April 2016 and published in September 2016. [PD-PAPER] was posted in November 2019 and published in 2020. [REL26] was published on 22 May 2026 (the publisher page may also show an earlier online date, unresolved), but the dates when its experiments were run were not read. [TINY26] carries identifier 2609.20290 while its abstract page shows v1 dated 2026-07-29. That inconsistency is recorded and not resolved. [FRAME26] has only rough dates from search metadata. See `references.md`.

## 7. What the literature implies for the protocol

| Finding | Design consequence |
|---|---|
| Lab-to-field drop is long established [MOHANTY16] | Do not frame the drop as a discovery; frame the study around confidence and abstention |
| `leaf_id` exists and image-level splits leak [PV-HF] | Split by `leaf_id`; verify zero group overlap with a test |
| Background shortcuts in PlantVillage [NOYAN22] | Interpret gaps cautiously; do not claim the shortcut explains the gap without a test |
| Post-hoc calibration can fail under shift [OVADIA19] | Fit temperature on source validation only; report the failure if it fails |
| ECE is binning-sensitive [NIXON19][ROELOFS] | Two ECE variants plus NLL and Brier; show bin counts |
| Risk–coverage is the standard selective-prediction view [GEIFMAN17] | Report coverage and selective risk together at a frozen threshold, plus the full curve |
| Small external test sets [TINY26] | Cluster bootstrap intervals; minimum per-class counts; flag low-support classes |
| PlantDoc numbers differ across distributions [PD-PAPER][PD-STD] | Count from the downloaded files; pin dataset revisions |

## 8. Compute environment evidence (Colab)

From the official FAQ, read on 2026-10-11 [V] [COLAB-FAQ]:

- Resources are "not guaranteed and not unlimited", and usage limits sometimes fluctuate. Colab does not publish them.
- GPU types vary over time. Idle timeouts, maximum VM lifetime and GPU type are all stated to vary.
- Free-tier notebooks run "at most 12 hours", depending on availability and usage patterns. Runtimes time out if idle. A virtual machine is deleted when idle for a while and has a maximum lifetime.
- Drive I/O can fail on folders with many items. The FAQ recommends copying data to the VM as an archive and unarchiving locally, instead of making many small reads.
- Free-of-charge runtimes disallow, and may terminate without warning, SSH shells, remote desktops, and use that bypasses the notebook UI to interact mainly through a web UI. **A Streamlit demo therefore should not be run from a free Colab runtime.** That is consistent with the plan to host it on AWS later.
- Drive mounting is not supported for local or Marketplace runtimes.

**Feasibility judgement [I].** Everything in this plan (a small CNN, ResNet-18 fine-tuning, inference, calibration) is modest compute. Feasibility depends on the assigned GPU and session lifetime, neither of which is guaranteed. The user's earlier diagnostic showed a Tesla T4 in one session; the FAQ says that may differ next time. The design therefore assumes sessions can end at any moment: resumable checkpoints, local staging of data, and no step that cannot be re-run. Throughput and total training time are not known and will be measured in the smoke test.

## 9. Open questions the literature did not settle

1. ~~Whether [REL26] Setting B uses test labels~~ Resolved by amendment A1: Setting B calibrates on a parent-image-disjoint 10% PlantDoc subset and tests on the remainder. Still open: its Brier convention, and the dates when its experiments were run.
2. Whether PlantVillage's `leaf_id` fully captures near-duplicate views. This is a property of the data to be audited.
3. Whether crops in the PlantDoc cropped release retain a link to their source photographs.
4. The PlantVillage licence (section 2.1).
5. Whether any real PlantVillage–PlantDoc class-name matches are semantically exact. This needs the audit and plant-pathology evidence, not name matching.
