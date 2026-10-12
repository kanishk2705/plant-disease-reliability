# References and Verification Status

Phase 1 reference list for the Plant Disease Classifier Reliability Study.
Compiled 2026-10-11. Amended 2026-10-11 (A1: REL26 and TINY26 upgraded after primary-source access). Every entry below was retrieved during Phase 1 through web search or fetch. The **Status** column says how much of each source was actually seen. Nothing here is cited from memory unless it is explicitly marked as such.

Status codes:
- **Verified (primary)**: the official repository, publisher page or arXiv abstract page was opened and the cited facts were read there.
- **Verified (abstract/snippet)**: only the abstract or a search-result excerpt was seen. The full text was not read.
- **Secondary**: the fact comes from a page that cites the source (a reference list, a dataset catalogue, a third-party summary).
- **Not verified**: the source was found but could not be opened, or its details are incomplete.

---

## A. Datasets

| Key | Reference | Status | Notes |
|---|---|---|---|
| [PV-REPO] | Mohanty, S. P. PlantVillage-Dataset (GitHub). https://github.com/spMohanty/PlantVillage-Dataset | Verified (primary) | README states 54,306 images, 14 crop species, 26 diseases. Provides `raw/color`, `raw/grayscale`, `raw/segmented` and a `leaf_grouping` folder. The README says the code targets Python 2.7. No licence statement appeared in the README text that was read. |
| [PV-HF] | Hugging Face dataset card `mohanty/PlantVillage`. https://huggingface.co/datasets/mohanty/PlantVillage | Verified (primary) | Card metadata lists licence `cc-by-sa-3.0`. Provides a `leaf_id` feature and a default 80/20 split that respects leaf grouping (train 43,596, test 10,709, so 54,305 images for the colour configuration). The card says custom splits must respect `leaf_id`. |
| [PV-PAPER] | Hughes, D. P. and Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060. | Verified (abstract) | The abstract announces "over 50,000" curated images. A third-party summary gives 54,309 images, which is Secondary and unconfirmed. |
| [PV-TFDS] | TensorFlow Datasets catalogue, `plant_village`. https://www.tensorflow.org/datasets/catalog/plant_village | Secondary | Lists 54,303 images in 38 categories, with source URL https://data.mendeley.com/datasets/tywbtsjrjv/1. |
| [PD-REPO-CROP] | Kayal, P. PlantDoc-Dataset (GitHub), described as the cropped dataset used for classification benchmarking. https://github.com/pratikkayal/PlantDoc-Dataset | Verified (primary) | CC BY 4.0. Has `train` and `test` folders. The README repeats the abstract's "2,598 data points". |
| [PD-REPO-DET] | Kayal, P. PlantDoc-Object-Detection-Dataset (GitHub). https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset | Verified (primary) | CC BY 4.0. Has `TRAIN`, `TEST`, `train_labels.csv`, `test_labels.csv`. |
| [PD-STD] | ai-agriculture-circuits-and-systems. plant_doc_detection (GitHub, standardized derivative). https://github.com/ai-agriculture-circuits-and-systems/plant_doc_detection | Verified (primary) | **This is the source of the "2,585 images / 29 classes" figures in the project specification.** It states CC BY 4.0, 8,600 bounding boxes and 29 class names. Its own split table sums to 2,481 images, and the README notes that some images are missing because of URL-encoded filenames. It is a derivative, not the original release. |
| [PD-RF] | Roboflow Universe PlantDoc pages. https://universe.roboflow.com/datasciencepro/plantdoc-7zwva | Secondary | States 2,569 images and 30 classes. Third-party re-hosting. |

## B. Plant-disease literature

| Key | Reference | Status | Notes |
|---|---|---|---|
| [MOHANTY16] | Mohanty, S. P., Hughes, D. P. and Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science* 7:1419. doi:10.3389/fpls.2016.01419. Preprint arXiv:1604.03169. | Verified (abstract) | Reports 99.35% accuracy on a held-out test set. It also reports 31.4% accuracy on images from trusted online sources taken under different conditions, against 2.6% for random selection. |
| [PD-PAPER] | Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S. and Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. *Proc. 7th ACM IKDD CoDS and 25th COMAD*, pp. 249–253. doi:10.1145/3371158.3371196. arXiv:1911.10317 (23 Nov 2019). | Verified (abstract + excerpt) | Abstract: 2,598 data points, 13 plant species, "up to 17 classes of diseases", about 300 human hours of annotation of internet-scraped images. An arXiv HTML excerpt gives "27 classes (17-10, disease-healthy)". |
| [NOYAN22] | Noyan, M. A. (2022). Uncovering bias in the PlantVillage dataset. arXiv:2206.04374. | Verified (abstract) | A random forest on 8 background pixels reached 49.0% test accuracy against 2.6% random. Concludes the dataset contains label-correlated noise that models can exploit. |
| [BARBEDO18] | Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. *Computers and Electronics in Agriculture* 153:46–53. | Secondary | **Could not be verified** (2026-10-12 searches did not find it). Not cited in any claim in this repository; treat as unconfirmed or drop it. |
| [ARIMA25] | Arima, Y., Kagiwada, S. and Iyatomi, H. (2025). DDD: Discriminative Difficulty Distance for plant disease diagnosis. arXiv:2501.00734. | Verified (abstract) | Raises concern about overestimated diagnostic performance when train and test sets come from the same domain. |
| [GRAPE26] | A dataset-centric benchmark of deep learning methods for grape leaf disease classification and detection. arXiv:2608.20608. | Verified (excerpt) | Reports exact image overlap among several controlled datasets. Authors and exact date were not retrieved. |

## C. Closest prior work (2026)

| Key | Reference | Status | Notes |
|---|---|---|---|
| [REL26] | Xiang, K., Shi, D. and Zhu, X. (2026). Quantifying the reliability gap in cross-domain plant disease classification: benchmarking the limited efficacy of standard mitigation techniques under controlled-to-field shift. *Frontiers in Plant Science* 17. Published 22 May 2026. doi:10.3389/fpls.2026.1826962. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1826962/full (PMC13236948 is a PMC copy) | **Verified (primary, full published article)** | Amended 2026-10-11 (protocol amendment A1). The user independently accessed the full article and supplied the bibliographic data; a second fetch of the publisher page the same day agreed with it, except that the fetch summary also mentioned "online 28 April 2026", which I did not resolve. The fetch tool summarises pages and read only the first 100,000 of 101,200 characters, so the quoted values below should still be checked against the article tables before public use. **Design:** ImageNet-pretrained ResNet-50 fine-tuned end-to-end on PlantVillage, head replaced by 2,048 to 21 logits; 21 known classes; an unknown pool of 490 samples in 7 categories excluded from training and calibration. **Setting A (source-only):** temperature scaling fitted on the PlantVillage validation split, T = 1.18, evaluated on all 2,081 matched known-class PlantDoc crops. PlantDoc accuracy 32.05%, macro-F1 0.2853, ECE 0.4771 before and 0.4400 after, Brier 1.1104 before and 1.0728 after; only 5.1% of samples answered at a nominal 80% coverage target. **Setting B (target-calibrated):** 10% of known PlantDoc samples (199 crops) used for calibration, grouped by reconstructed parent-image ID (Hungarian matching on 32x32 thumbnails, 2,571 of 2,572 crops recovered), with 1,882 crops left for testing; 10 repeats with seeds 42 to 51; representative split T = 1.54, ECE 0.4761 before and 0.3651 after, Brier 1.1124 before and 1.0083 after; mean over repeats: accuracy 32.10%, macro-F1 0.2859, calibrated ECE 0.3645, Brier 1.0051, selective risk 0.6430 at nominal 80% coverage. **Brier convention:** not stated in the text I could read; values above 1 imply a sum over classes (my inference). **Citation rule:** always name the setting. Setting A values are the comparable ones for this study. Setting B uses target labels and is not comparable. |
| [TINY26] | Ho-Lam, N.-B. and Nguyen, T.-A. (2026). TinyCNN: A 193K-Parameter Network for On-Device Plant Disease Detection, with a Cross-Dataset Robustness Diagnosis. arXiv:2609.20290. https://arxiv.org/abs/2609.20290 | **Verified (primary: arXiv abstract page; full HTML per user)** | Amended 2026-10-11 (protocol amendment A1). The user accessed the full arXiv HTML including the experimental setup and reports: PlantVillage re-split 70:15:15 stratified; PlantDoc test subset of 252 images across 27 mapped classes. I re-fetched only the abstract page, which gives about 193K parameters, 98.88% PlantVillage test accuracy and 98.03% macro-F1 on 38 classes, and Grad-CAM evidence of background-driven attention. **Bibliographic note (preserved, not resolved):** the identifier prefix 2609 would normally indicate a September 2026 submission, but the abstract page I fetched shows v1 dated 2026-07-29. The user observed the same inconsistency. Cite as "arXiv:2609.20290, 2026" with this note. No corrected date is asserted. |
| [FRAME26] | "AI-Based Crop Disease Detection from Leaf Images: A Focused Review and Robust Evaluation Framework." Preprints.org 202609.1678. | Verified (excerpt only) | Non-peer-reviewed preprint. Proposes PlantVillage as controlled benchmark and PlantDoc as field-like external dataset, with only semantically matched classes. |
| [MRC26] | "MRC-Net: a reliable plant disease classification framework with multi-frequency state-space enhancement and conformal prediction." *Frontiers in Plant Science* 2026, doi:10.3389/fpls.2026.1895287, published 2026-09-03 per the search result. | Verified (excerpt only) | Conformal prediction for plant disease. Not a cross-dataset reliability study. Tangential. |

## D. Distribution shift, calibration and selective classification

| Key | Reference | Status | Notes |
|---|---|---|---|
| [GUO17] | Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, PMLR 70. arXiv:1706.04599. | Verified (abstract + excerpts) | Defines ECE as used in this project. Finds that modern networks are poorly calibrated and that temperature scaling, a single-parameter variant of Platt scaling, is effective on most datasets. |
| [OVADIA19] | Ovadia, Y. et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS* 32. arXiv:1906.02530. | Verified (abstract) | Reports that traditional post-hoc calibration falls short under dataset shift, and that post-hoc calibration can fail even under mild shift. |
| [MINDERER21] | Minderer, M. et al. (2021). Revisiting the calibration of modern neural networks. *NeurIPS*. arXiv:2106.07998. | Verified (abstract) | Finds architecture to be a major determinant of calibration, and that decay of calibration under shift is less pronounced in recent architectures. |
| [HENDRYCKS17] | Hendrycks, D. and Gimpel, K. (2016). A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv:1610.02136. | Verified (abstract) | Maximum softmax probability as a baseline for detecting errors and OOD inputs. The abstract also notes the baseline can be surpassed. |
| [GEIFMAN17] | Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*. arXiv:1705.08500. | Verified (abstract + excerpts) | Defines selective risk, coverage and the risk–coverage curve for deep networks. |
| [GEIFMAN19] | Geifman, Y. and El-Yaniv, R. (2019). SelectiveNet: A deep neural network with an integrated reject option. *ICML*, PMLR 97. | Verified (abstract) | Integrated reject option. Context only. |
| [NIXON19] | Nixon, J., Dusenberry, M., Jerfel, G., Nguyen, T., Liu, J., Zhang, L. and Tran, D. (2019). Measuring calibration in deep learning. arXiv:1904.01685. | Verified (abstract) | ECE has numerous flaws. Adaptive (equal-mass) binning gives more stable metric rankings across bin counts. Class-conditional measures and the L2 norm are also discussed. |
| [ROELOFS] | "Mitigating bias in calibration error estimation." arXiv:2012.08668. | Verified (excerpt only) | Authors were not shown in the excerpt. States that ECE with 15 equal-width bins is the most common choice, that binned ECE grows with the number of bins, and that equal-mass ECE is less biased. |
| [NAEINI15] | Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. | Secondary | Bibliographic details confirmed 2026-10-12 against the AAAI proceedings page (ojs.aaai.org, AAAI 2015). Full text not read; used only as the cited origin of ECE. |
| [BRIER50] | Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1):1–3. | Secondary | Title, Monthly Weather Review 78(1):1–3 and DOI 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 confirmed 2026-10-12 via a secondary page. Original not read; used only as the origin of the Brier score. |

## E. Tooling and compute

| Key | Reference | Status | Notes |
|---|---|---|---|
| [COLAB-FAQ] | Google Colab Frequently Asked Questions. https://research.google.com/colaboratory/faq.html (fetched 2026-10-11) | Verified (primary) | See the literature review, section 8, for the specific claims used. |
| [TV-R18] | torchvision documentation, `resnet18` (docs version 0.29). https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html | Verified (primary) | `ResNet18_Weights.IMAGENET1K_V1` (also `DEFAULT`): ImageNet-1K acc@1 69.758, 11,689,512 parameters, inference transform is resize 256, centre crop 224, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]. |
| [RESNET] | He, K. et al. Deep residual learning for image recognition. arXiv:1512.03385. | Secondary | Title, authors and year confirmed 2026-10-12 on arxiv.org/abs/1512.03385. Full text not re-read here; used only as the architecture reference. |

---

## Items that must be manually verified before any public use

1. [REL26]: full article accessed (user, plus a summarised fetch by me). Before public use, re-check each quoted number against the article tables and confirm the Brier convention from its Equation 17.
2. [TINY26]: full HTML accessed by the user. Before public use, re-check the evaluation protocol wording, and resolve or keep the identifier/date note.
3. PlantVillage licence: the Hugging Face card says `cc-by-sa-3.0`, and the GitHub README states none in the text that was read. Check the original distribution at the Mendeley URL listed in [PV-TFDS]. This is a Phase 3 audit item and remains open.
4. [BARBEDO18], [NAEINI15], [BRIER50], [RESNET]: not read in this phase.
