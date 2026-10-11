# Phase 3 dataset audit report (provisional)

All numbers are counted from the files listed in `docs/dataset_provenance.md`. Scripts: `scripts/phase3_audit.py`, `scripts/phase3_analysis.py`, `scripts/phase3_counts.py`. No model output was used for any decision. The final dataset freeze happens at the end of Phase 4 (amendment A2.1); this report contains **proposals only**.

## 1. PlantVillage (colour)

- 54,305 images, 38 classes, all 256x256 (one RGBA). 0 unreadable.
- **Leaf grouping is incomplete.** The published `leaf-map.json` resolves 41,111 images to a leaf id; **13,194 images (24.3%) are not in the map** and fall back to one group per image identifier. After the repository loader's rule there are 20,015 groups (7,593 from the map, the rest fallbacks). 11,834 groups are singletons; the largest has 33 images. No group spans two classes.
- Exact duplicates: 21 groups (42 files), all inside one leaf group and one class. Harmless for splitting.
- Near duplicates by perceptual hash (pairs): 22 at distance 0, 119 at <=2, 1,477 at <=4, 10,627 at <=6. Pairs crossing a class: 311 at <=4, 3,513 at <=6. So at distance >=3 the hash links different leaves and different classes (uniform backgrounds); **<=2 within the same class** is the only usable merge rule. It adds 112 same-class links and reduces 20,015 groups to 19,960.

## 2. PlantDoc "cropped" release

- 2,578 whole images, 28 classes (train 2,342, test 236); 0 unreadable. Smallest side 69 px, median 602, 5th percentile 227; 6 images have a smaller side under 128 and none under 64.
- Aspect ratio (width/height): median 1.33, 10th to 90th percentile 0.75 to 1.52, and only 21.8% lie within 0.8 to 1.25.
- Exact duplicates: 12 groups. **9 of them carry conflicting labels** (for example one potato photo labelled both early and late blight, a corn photo labelled both gray leaf spot and leaf blight), and 11 span the original train and test folders. 16 file names occur in both folders.
- Near duplicates by perceptual hash inside PlantDoc: 59 pairs at distance 0, then only 7, 5 and 7 new pairs at <=2, <=4, <=6. The curve is flat, and I inspected the 6 pairs with distance 4 to 6: all were the same photograph with a different watermark or resolution. **<=6 is safe inside PlantDoc.**
- 46 images have boxes of more than one class in the detection labels, so their image-level label is ambiguous.
- **Class content caveat:** the "Apple leaf" and "Peach leaf" classes include fruit photographs and stock images on white backgrounds (seen in the inspected pairs). Treat the healthy classes as "no disease label", not as verified healthy field leaves.

## 3. Cross-dataset overlap (PlantVillage vs PlantDoc)

- **Exact shared images: 0.** Nearest perceptual-hash distance per PlantDoc image: 1 image at 4, 4 images at 6, 26 at 8, 100 at 10. The 5 closest pairs, viewed side by side, are different leaves with similar silhouettes, not copies.
- The hash is global and low resolution, so it cannot find heavily edited copies. I make no claim of zero overlap beyond what this check can see.

## 4. Proposed external unit, cleaning rules and counts

Proposal: use the 2,578 whole images of the cropped release as the external set, one image per source cluster. A cluster is the connected component of exact-hash matches and perceptual-hash distance <=6 inside PlantDoc.

| Step | Images |
|---|---|
| Start | 2,578 |
| Clusters (after linking) | 2,502; 74 clusters hold more than one image |
| Clusters with conflicting labels, all removed | 66 images (32 clusters) |
| Remaining images with several box classes, removed | 28 |
| Redundant members of consistent duplicate clusters, removed | 41 |
| **Clean external set, one image per cluster** | **2,443** |

## 5. Alignment proposal and minimum counts

Proposed table: `configs/proposed/class_alignment_proposed.json`; counts per class: `docs/audit/alignment_counts.csv`. Tiers: A specific and matching, B broader PlantDoc name (doubt rule: excluded from the primary set), C no counterpart.

- Applying the protocol 4.4 starting values (at least 15 external images after cleaning; at least 200 PlantVillage images in at least 20 groups): **primary set K = 23 classes** (all tier A; none from tier B), with **1,993 external images** and 33,417 PlantVillage images. The smallest class has 53 external and 360 PlantVillage images. So the starting thresholds are not binding on the primary set, except that two-spotted spider mite (2 PlantDoc images, 0 after cleaning) is dropped.
- Tier B classes (4): Apple___Cedar_apple_rust, Pepper,_bell___Bacterial_spot, Corn_(maize)___Northern_Leaf_Blight, Corn_(maize)___Common_rust_. Including them would give a **sensitivity set of K = 27 classes and 2,443 external images**. They pass the count rule.
- PlantVillage classes with no PlantDoc counterpart (10): Apple___Black_rot, Cherry___Powdery_mildew, Corn___healthy, Grape___Esca, Grape___Leaf_blight, Orange___Haunglongbing, Peach___Bacterial_spot, Potato___healthy, Strawberry___Leaf_scorch, Tomato___Target_Spot. They are not in the primary model's output space.
- Of the 33,417 PlantVillage images in the primary classes, 8,118 have no published leaf id (fallback groups). Phase 4 must handle this; see the proposal below.
- **Evidence status:** the evidence column is general plant-pathology knowledge written by Claude and **not verified against cited sources**. The user's pathology sign-off is a precondition of the dataset freeze.

## 6. Proposed decisions (for the Phase 4 freeze)

| Decision | Proposal | Basis (data property only) |
|---|---|---|
| External unit | Whole images of the cropped release, one per cluster | Release is whole images with image-level labels; every file traces to a detection image by name |
| `source_image_id` | Cluster id (SHA-256 equality plus perceptual-hash distance <=6, inside PlantDoc) | File names are not unique (16 repeated across train/test) |
| Conflicting-label clusters | Remove all members | Same photograph with two labels |
| Multi-class images | Remove | Image-level label ambiguous |
| Original PlantDoc train/test folders | Ignored; all cleaned images are external test | Protocol 2 |
| Minimum image size | Smallest side >=64 px (removes none) | Smallest side is 69 px |
| Evaluation transform | Resize the whole image to 224x224, no crop, same for both datasets and both models | Only 21.8% of PlantDoc images are near-square, so a centre crop would discard large parts |
| PlantVillage duplicate rule | Merge groups of same-class images at perceptual-hash distance <=2 | Larger distances link different classes |
| Fallback-group risk | Phase 4 reports leakage statistics for the 13,194 fallback images and merges by the rule above; the residual risk is stated, not hidden | Leaf map covers only 75.7% |
| Cross-dataset rule | Remove any PlantDoc image with an exact PlantVillage match (none now); review matches <=6 by eye (5 found, none removed) | Nothing exact found |
| Minimum counts | Keep 15 external and 200 PlantVillage images in 20 groups | Not binding on the primary set |
| Primary set | The 23 classes of tier A passing counts | |
| Sensitivity set | 27 classes (adds tier B), labelled exploratory | |

## 7. Limits and open items

- PlantVillage licence unresolved (no repository licence file; Hugging Face says cc-by-sa-3.0; Mendeley unreachable). No weights or manifests with PlantVillage names are published until resolved.
- Pathology sign-off pending; evidence not source-verified.
- The PlantVillage copy was not compared with the Hugging Face copy.
- PlantDoc image-level labels are taken as given apart from the removals above.
- PlantVillage images are lab photographs on grey backgrounds and PlantDoc images are web photographs; the task measures that shift, and the "healthy" PlantDoc classes include non-leaf photos.
