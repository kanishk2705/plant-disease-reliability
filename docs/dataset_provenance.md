# Dataset provenance (Phase 3, provisional)

Recorded 2026-10-11. Everything below was obtained and counted in this phase. Nothing is copied from the project specification or from other studies. Obtained by `git clone` from GitHub (the sandbox cannot reach Hugging Face, Mendeley or Colab).

| Item | PlantVillage | PlantDoc "cropped" release | PlantDoc detection release |
|---|---|---|---|
| Repository | https://github.com/spMohanty/PlantVillage-Dataset | https://github.com/pratikkayal/PlantDoc-Dataset | https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset |
| Revision used | `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` (commit dated 2026-02-05) | `5467f6012d78d1c446145d5f582da6096f852ae8` (2021-05-02) | `4730a233a555b30ee98e0879c63ad25d82407455` (2022-02-21) |
| Download date | 2026-10-11 | 2026-10-11 | 2026-10-11 |
| Part used | `raw/color` and `leaf_grouping/leaf-map.json` (sparse checkout; no grayscale or segmented) | `train/` and `test/` | `TRAIN/`, `TEST/`, `train_labels.csv`, `test_labels.csv` (used for traceability and diagnostics only) |
| Files counted | 54,305 images in 38 class folders | 2,578 images in 28 class folders (train 2,342, test 236) | 2,603 image files plus XML, 8,921 labelled boxes |
| Unreadable files | 0 | 0 | 0 |
| Licence as found | **No licence file in the repository tree and none in the README text.** The Hugging Face card metadata (read via a page fetch) says `cc-by-sa-3.0` with no licence text. The Mendeley record could not be reached. **Unresolved.** | `LICENSE.txt` in the repository: Creative Commons Attribution 4.0 International. README says the same. | `LICENSE.txt` in the repository (CC BY 4.0 per its README). Images were scraped from the web, so the licence of each underlying photograph is not established by this file. |
| Citation | Mohanty, Hughes and Salathé (2016), doi:10.3389/fpls.2016.01419 (repository `CITATION.cff`); Hughes and Salathé (2015), arXiv:1511.08060 | Singh et al. (2020), doi:10.1145/3371158.3371196 | same |
| Leaf-grouping file | `leaf_grouping/leaf-map.json`, SHA-256 `3b4b253683c6911744959a3870a92509b1faa1ee34e9f8585e21d0fe6337a25b`, 40,328 keys | n/a | n/a |

## Discrepancies recorded, not resolved

- The PlantVillage README says 54,306 images; the colour folder holds **54,305** (one image is RGBA). The Hugging Face card also lists 43,596 + 10,709 = 54,305 for colour while its text says 54,306.
- The PlantDoc paper says 2,598 data points; the cropped release holds 2,578 images; the detection release holds 2,603 image files with 2,587 distinct file names. These do not agree, and none of them equals the 2,585 or 2,569 quoted elsewhere.
- The PlantDoc "cropped" release is not made of crops: the image size equals the detection image size for 2,574 of 2,578 files, so these are whole images carrying one label each.

## Hashes of the audit manifests (kept out of Git under `outputs/audit/`)

| File | SHA-256 |
|---|---|
| `pv_manifest.csv` | `9841de36fb0db93b419da2ae4fd13832afc5a185fab77a6091f8a60760bdfc7e` |
| `pdcrop_manifest.csv` | `5b1995ffb3b856115dfdc949ae17e7f910a7cc319dfb32f2a8b57d9a0a9bd8e0` |
| `pddet_manifest.csv` | `28d67c06c9e7bd9502b51e0761f22b703314ba64dce57193e5a5d3015d228ac3` |
| `pdcrop_annotated.csv` | `af70bc31c8dd09d471cd165710942d36700fdfa57f950249df2acf6845d2a1c2` |
| `external_clean_manifest.csv` | `569add09ee5703957c915436e1383d892f67e413729cb7f2eb8e6bd05ccc6de4` |
| `pv_groups_manifest.csv` | `fff69133e1287203d41129e4aa396f1b20efdb8b4e967b1222b664f198c6f709` |

Per-image hashes are inside the manifests. The manifests are not committed because the PlantVillage licence is unresolved (protocol section 3). Aggregate tables are committed under `docs/audit/`.

## What was inspected versus assumed

Inspected: file counts, image sizes and modes, SHA-256 and a DCT perceptual hash of every image, leaf-map coverage, label consistency against the detection labels, and the closest cross-dataset matches by eye (5 pairs) and within-PlantDoc near-duplicate pairs (6 pairs). Assumed or not checked: that PlantVillage's published `leaf-map.json` is correct; that PlantDoc image-level labels are correct; the licence of the original photographs; the identity between this GitHub PlantVillage copy and the Hugging Face or Mendeley copies (not compared, because those hosts were unreachable).
