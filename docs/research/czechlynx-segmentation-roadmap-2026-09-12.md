# CzechLynx recognition and clustering roadmap

## Recommendation

Use fully segmented animal images as the primary development path. Compare frozen MegaDescriptor and MiewID embeddings, replace the current custom reranking rule with a calibrated WildFusion reference, and build a clustering graph that can reject weak links. Then fine-tune the strongest global model using identity labels, with sampling and augmentation designed around camera and illumination changes.

A 50% recognition target is a reasonable research milestone, but it is not established as achievable on this exact protocol. The current segmented result is **29.37% top-1**. Its verified candidate list imposes a **49.17% ceiling**, so tuning geometric verification within the existing top ten cannot reach the target. Clustering is further from useful: it combines 60 real identities into seven groups, with **6.11% pairwise precision**.[^local]

The practical objective should be a review assistant that produces useful identity suggestions and conservative groups, while leaving difficult images unresolved. Reliable population estimates require additional ecological assumptions and validation; photographed identity groups do not establish total park abundance.

This assessment uses the saved development runs and sources available on 12 September 2026. All experimental changes below are proposals. Frozen-model comparisons and supervised fine-tuning are both included.

## 1. Current evidence

### Recognition results

The three completed runs use the same 750 gallery images and 560 queries. Of those queries, 303 have a matching identity in the gallery and 257 are unknown. The accuracy columns below concern the 303 known queries, without applying the rejection threshold. These are development results, not results on the untouched official test split.[^local]

| Pipeline stage | Bbox images | Fully segmented images | Bbox retrieval, masked GV points |
|---|---:|---:|---:|
| Global top-1 | 23.43% | 20.13% | 23.43% |
| Global + Fisher top-1 | 28.05% | 22.11% | 28.05% |
| Full pipeline top-1 | 34.32% | **29.37%** | 33.99% |
| Full pipeline top-5 | 46.53% | **39.60%** | 44.22% |
| Full pipeline mAP@50 | 0.0944 | **0.0760** | 0.0932 |
| Correct top-1 queries | 104/303 | **89/303** | 103/303 |

The bbox advantage is five percentage points over full segmentation. That establishes better performance on this sample, but does not establish better transfer to a new park or camera. Foreground-only development remains justified because the intended application needs identity evidence that survives changes in surroundings.

The segmented pipeline uses supplied masks, crops the animal, replaces the background with gray RGB 128, and saves lossless PNGs. Global embeddings, Fisher features and geometric verification all see those segmented pixels. This is an annotation-assisted benchmark; automatic field segmentation remains a separate source of error.[^implementation]

### The shortlist bottleneck

The current global/Fisher stage forms a union of candidates, sorts them by the mean of clipped cosine scores, and geometrically reranks only the first ten. A correct candidate below that boundary cannot become top-1 through the existing reranking operation.[^implementation]

| Correct identity present within the segmented retrieval list | Known queries covered | Maximum possible top-1 if reranking were perfect within that list |
|---|---:|---:|
| First 10 candidates | 149/303 | **49.17%** |
| First 20 | 175/303 | 57.76% |
| First 50 | 228/303 | 75.25% |
| Entire union of global/Fisher shortlists | 258/303 | 85.15% |

Reaching at least 50% on these 303 queries requires 152 correct predictions, or 63 more than the current 89. Wider candidate verification creates room for improvement; it does not guarantee that the matcher will select the correct animal. Better global retrieval is still needed, especially for the 45 queries with no correct identity anywhere in the union.

### Rejection is also unreliable

At the current segmented rejection threshold, 65 queries are accepted: 45 correct known identities, eight incorrect known identities and 12 unknown animals incorrectly attached to the gallery. Overall accepted-match precision is therefore **45/65 = 69.23%**.

The approximately 4.67% unknown false-accept rate does not mean accepted suggestions are 95% correct. Furthermore, these rejection thresholds were fitted and assessed on the same development unknowns. They need an independent calibration/selection split before supporting any deployment claim.[^local]

### Illumination and acquisition appearance

A foreground colour diagnostic was computed from the existing segmented images. After resizing images and masks to 128 × 128, an image was marked near-monochrome when the 90th percentile of foreground channel spread, `max(R,G,B) − min(R,G,B)`, was at most five. This is an appearance proxy, **not a verified day/night or infrared label**. White-flash night images can be colour, and colourlessness alone does not identify the sensor.[^diagnostic]

| Full pipeline result | Near-monochrome known queries | Colour known queries |
|---|---:|---:|
| Bbox | 37/97 = 38.14% | 67/206 = 32.52% |
| Fully segmented | 26/97 = 26.80% | 63/206 = 30.58% |
| Bbox with masked GV points | 35/97 = 36.08% | 68/206 = 33.01% |

The stronger warning concerns changes between query and reference appearance: **27 known queries have correct gallery examples only in the opposite colour-proxy category. All three pipelines miss all 27.** These cases span five identities, including 19 near-monochrome queries and eight colour queries. Their small, correlated sample does not isolate illumination causally; viewpoint, camera and image quality may also differ.

Clustering shows the same association. One segmented cluster contains **104 images of 22 identities, all near-monochrome**. Another contains 79 images, of which 70 are near-monochrome. Together these groups contain 174 of the 178 near-monochrome queries. Background removal has therefore not removed acquisition-related similarities.

### The clustering graph

The segmented full-pipeline clustering has ARI 0.0649, pairwise precision 6.11%, pairwise recall 37.99%, seven clusters and no unassigned images. All seven clusters mix identities. Although the requested backend is Leiden, the saved run used the NetworkX Louvain fallback.[^local]

Graph construction is a more immediate concern than the choice between those algorithms. Every image nominates 15 neighbours and the graph takes their symmetric union without a meaningful confidence cutoff. With this query distribution, even perfect identity rankings would place at least **2,753 of 8,400 directed neighbour slots, or 32.77%, on another identity**, because many identities have fewer than 16 images.

This is a lower bound on neighbour slots, not an observed count of unique false edges or a proof that every such edge causes a merge. It nevertheless shows why fixed neighbour quotas are inappropriate as sufficient evidence for shared identity. The graph needs permission to have fewer neighbours, isolated nodes and unresolved groups.[^diagnostic]

## 2. External baselines and comparability

### The two Kaggle notebooks

The requested [CzechLynx baseline with WildFusion](https://www.kaggle.com/code/picekl/czechlynx-baseline-with-wildfusion) and [CzechLynx base open ImageNet](https://www.kaggle.com/code/mdnomanhossain/czechlynx-base-open-imagenet) could not be audited at cell level: their executable cells and saved outputs were not accessible. No accuracy, preprocessing choice or split is attributed to those exact notebooks here. An exported notebook with its dataset version and outputs would close that specific evidence gap.

The dataset card explicitly notes an update adding backgrounds and around 100 identities. The current real data contain 39,760 images and 319 identities. Consequently, a notebook using an earlier release may differ substantially from the present benchmark.[^dataset]

A valid reproduction must pin dataset version, image IDs, split assignments, segmentation, checkpoint, calibration identities, gallery construction and metric definition. “Open” in a notebook title is not enough to establish that unknown detection, retrieval and clustering are scored identically.

### WildFusion and the current implementation

The original WildFusion method combines separately calibrated global similarity and local matching scores. Its local branches use counts of sufficiently confident matches; isotonic calibration with interpolation makes heterogeneous scores comparable before fusion. Its published 84% mean identification accuracy across 17 datasets is **not a CzechLynx result**.[^wildfusion]

The official Wildlife Tools implementation exposes separate matcher/extractor/calibration pipelines and a candidate budget. Its examples include MegaDescriptor, DISK, ALIKED and SuperPoint with LightGlue. That budget controls score evaluation work; it is unrelated to Fisher GMM components.[^wildtools]

| Component | Current CzechLynx runner | Proposed reference comparison |
|---|---|---|
| Global model | Frozen MegaDescriptor, 1536-D | Same MegaDescriptor first; MiewID separately |
| Local ensemble | DISK + SuperPoint + ALIKED Fisher vectors | Separate local matching branches |
| GV matcher | DISK + LightGlue | DISK + LightGlue and ALIKED + LightGlue |
| Score scaling | Raw clipped cosine mean | Calibration for each signal |
| Reranking evidence | RANSAC inliers in a custom power score | Calibrated high-confidence match counts |
| Candidate budget | First ten reranked | Compare 20 and 50 candidates |
| Fisher role | Mandatory in full-funnel path | Optional measured contribution |

The current ensemble uses PCA 128 and **256 GMM components per extractor**, followed by an equal-weight sum of separately encoded Fisher vectors and normalization. That sums coordinates from independent descriptor spaces and codebooks, introducing cross-extractor terms when similarities are computed. It deserves an ablation against separate score fusion or normalized concatenation; the existing improvement is not evidence that coordinate summation is optimal.[^implementation]

The repo already contains a WildFusion reference script and calibration classes. Reuse these components, but adapt them to the frozen CzechLynx protocol. The existing reference script is a closed-set evaluator and uses a different model identifier; checkpoint identity must be verified before comparison.[^reusable]

### Evidence for foreground-only recognition

Picek, Neumann and Matas studied 6,743 lynx images, 103 individuals and 81 locations with a temporal train/test split. Their foreground-only model achieved 49.3% overall accuracy versus 49.7% for the whole-image baseline, while improving accuracy at new locations from 13.9% to 16.3%. Foreground training with their PITS objective reached 52.5% overall and 19.1% at new locations.[^foreground]

This supports the concern that headline accuracy can reward location dependence. It also demonstrates a foreground-only result above 50% in a different supervised setting. The small new-location result prevents treating that headline as evidence of reliable park transfer, and its 103-identity classification protocol does not forecast performance on the current open-set benchmark.

### MiewID and recent clustering research

MiewID-msv3 is a 51.11-million-parameter wildlife embedding model with 440 × 440 input. Its card reports Eurasian lynx rank-1 of 69.8% on Whiskerbook-sourced evaluation data. That makes it a worthwhile frozen-model candidate, not a prediction for CzechLynx. Eurasian lynx is represented in its evaluation coverage; unseen-species performance must not be assumed, and exact CzechLynx training overlap remains unverified.[^miewid]

A 2026 AnimalCLEF working paper combines MiewID, DISK/ALIKED LightGlue scores, calibration and graph clustering. It reports private leaderboard ARI of 0.21221 for the competition baseline, 0.66695 for its main frozen-model pipeline, and 0.70393 for its ensemble. Individual fine-tuned variants scored below the frozen pipeline. These are aggregate challenge clustering results, **not CzechLynx top-1 accuracy**.[^animalclef]

The paper's preprocessing description also says lynx inputs were already focused crops. Its aggregate segmentation ablation is therefore not controlled proof that fully masking our lynx images will produce the same improvement. The relevant lesson is to benchmark the entire similarity-to-graph design and retain a frozen baseline throughout fine-tuning. The accompanying repository provides reusable implementation references.[^animalcode]

## 3. Evaluation design

### Separate model fitting, calibration and selection

Preserve the existing development subset as a diagnostic anchor. Establish a new, versioned protocol within official training data with three functions: encoder/Fisher fitting, score and rejection calibration, and development selection. Group by encounter to prevent burst leakage; reserve some identities entirely from encoder training to measure transfer to unseen individuals.

Calibration examples must come from the final checkpoint and the same preprocessing used at inference. Calibrating a frozen model and subsequently changing its weights invalidates that calibration. Development unknown identities must be absent from both the reference gallery and supervised encoder fitting. Separately measure gallery-known animals that were never encoder-training identities.

For the final evaluation, freeze all model choices, transforms, fusion parameters and thresholds, then run the official time-open test. The installed metadata specify 27,587 training and 12,173 test images, with 44 test identities absent from training. Preserve a calibration holdout if refitting on more training data; do not train on the calibration pool and silently reuse its old thresholds.[^dataset][^implementation]

Retain explicit overlap audits. The existing time-closed metadata contain shared mixed-animal encounters, so official split names alone do not establish independence. Report any purged query records rather than silently changing assignments.

### Measure the intended transfer

Report image-, encounter- and identity-averaged metrics, with confidence intervals grouped by encounter or identity rather than treating adjacent photos as independent. Include camera-separated positives, different appearance modes, difficult exposure, blur, visible flank and small animal crops. Show subgroup counts alongside scores.

Evaluate geographic transfer as a separate experiment. A split with disjoint identities cannot be used as an ordinary known-identity retrieval task with the training population as its only gallery. For retrieval, construct a disclosed reference/query division within the held-out population, separated by encounter; for pure discovery, evaluate its clusters without identity access during inference.

The current data have date-only metadata and no directly usable day/night or flank field in the inspected CSV. The initial audit should manually label roughly 150–300 selected development images for visible-light/likely infrared/uncertain appearance, exposure, blur and visible side. Include all 27 cross-proxy failures, their true references and a representative control sample. Keep uncertainty explicit.

### Segmentation transfer

Use supplied masks for controlled recognition development. Before a field demonstration, add a separate end-to-end evaluation using predicted detections and masks on images excluded from any mask-model training or adaptation. Measure missing animals, truncated flanks and background leakage, as well as final retrieval and clustering.

Mask contours and body silhouette can still encode pose, camera angle and size. Compare modest crop padding, preserved aspect ratio and conservative mask-boundary filtering only after a stable reference exists. Keypoint-centre filtering does not guarantee that an entire descriptor's receptive field lies on the animal.

## 4. Recognition improvements

### Frozen reference first

Keep the segmented RGB inputs and current MegaDescriptor checkpoint as a reproducibility control. Add a model adapter for MiewID using its documented preprocessing. Cache each model/view separately, including checkpoint hash, transform, input-image hash and precision settings.

Compare global-only retrieval before adding local matching. If a new embedding improves candidate recall at 20 and 50 but has modest top-1 improvement, it may still improve the final pipeline. Conversely, a model that only improves same-camera matches is a poor pilot candidate.

Use DISK and ALIKED as independent LightGlue branches on segmented pixels. Record match confidence, coordinates, retained foreground points, inliers and spatial coverage. Keep raw-match scoring and geometric-inlier scoring as separate ablations. A single geometric transform may reject correspondences on a bending or differently posed animal; this is a hypothesis to test, not a reason to discard geometry.

Fit simple calibrators on development pairs drawn from the candidate distribution. Include same-identity cross-encounter positives and difficult different-identity negatives with similar camera/appearance conditions. An initial bounded sample might contain 300–500 positives and 1,500–3,000 hard negatives, expanded if learning curves or subgroup coverage demand it.

Balanced or hard-mined pair sampling changes class prevalence. A fitted “probability” is conditional on that sampling distribution until checked on representative candidate pairs. Use separate validation of reliability and accepted-match precision for retrieval and clustering.

### Illumination handling

Test a second grayscale view of the masked animal and fuse its score with the RGB view after calibration. This retains useful colour information while offering a texture-focused comparison. Start with the winning frozen global model; add a grayscale local branch only if diagnostics justify its extra matching cost.

A grayscale image is not a faithful infrared simulation. Sensor spectrum, flash direction, saturation, blur and exposure can alter apparent coat texture. The visible–infrared literature supports channel augmentation and consistency training as a way to reduce colour dependence, but that evidence comes from human recognition and must be validated on lynx.[^illumination]

Use conservative brightness, gamma, contrast, channel/grayscale, noise and blur augmentation during fine-tuning. Compute any exposure normalization from foreground pixels so the artificial background does not dominate its statistics. A deterministic foreground contrast-normalization variant can be compared separately; sharpening or CLAHE should not become defaults without measured benefit.

Do not build independent “day identities” and “night identities.” Such separation would hide the cross-mode problem and fragment the same animal. Exposure and modality should guide diagnosis, sampling and uncertainty, rather than impose hard identity barriers.

### Fine-tuning

Start from a pretrained wildlife embedding model. Compare a small head-only warm-up with gradual unfreezing, using the model's native embedding dimensionality. For MegaDescriptor, preserve the existing architecture-compatible 1536-D representation; a projection-head change would be a separate experiment.[^reusable]

Use identity-balanced batches containing different encounters, and preferentially different cameras and appearance modes for the same animal when available. Include visually similar negative identities. Audit how many identities actually have cross-mode positive pairs; augmentation cannot replace absent real cross-sensor evidence.

The initial objective should be one established identity/metric learning recipe, such as the existing ArcFace-based trainer, with cross-mode sampling. Add an embedding-consistency term only as a subsequent ablation. Begin with a small pretrained-backbone learning rate, approximately 2e-5 as a starting configuration, early stopping on held-out retrieval, and one fixed seed for initial screening.

Ordinary horizontal flips can support orientation robustness, but they do not reveal the animal's unseen opposite flank. If visible-side labels are used, update them under flips. Avoid aggressive crops or erasing that remove nearly all identifying coat pattern.

The available GPU is a 16 GB RTX 4070 Ti SUPER. Use mixed precision and measured microbatch sizing; activation memory, not parameter count alone, determines fit. Gradient accumulation can increase effective batch size, but it does not by itself enlarge the simultaneously available negative set of a batch-based metric loss.

Retain the frozen model if fine-tuning improves familiar identities while harming unseen animals or cameras. Consider score-level ensembling only when the held-out errors are complementary enough to justify extra inference.

## 5. Clustering improvements

### Conservative graph construction

Build a sparse query–query candidate graph from the strongest embeddings. Keep edges only when calibrated evidence exceeds a development-selected threshold, with mutual-neighbour support as an additional option. A top-K list limits search work; it must not force K accepted relationships.

Allow isolated images, singletons and unresolved cases. Compare Chinese Whispers and Leiden/Louvain on the same filtered graph before attributing gains to the clustering algorithm. Also evaluate a conservative agglomerative alternative on the bounded development subset.

A high-confidence edge is not sufficient evidence that every pair in its connected component matches. Audit weak bridge edges and require additional cross-group support before merging established groups. A three-image chain where A matches B and B matches C, while A conflicts with C, is a necessary diagnostic case.

k-reciprocal reranking can incorporate mutual neighbourhood information, but should follow stronger embeddings and graph filtering. Otherwise it may reinforce acquisition-related groups. Its original evidence comes from person re-identification, so its animal benefit remains an experiment.[^rerank]

### Identity attachment and review

Separate discovery among query images from attachment to a known gallery identity. For a query group, require consistent support across independent encounters before accepting an identity; use both best score and margin over competing identities. Conflicting confident votes should trigger review.

Provide a small review set consisting of cluster representatives, the weakest accepted links and proposed merges. Collect “same animal / different animal / uncertain” decisions on those pairs. Save confirmed relationships as persistent constraints, with model version and evidence, and apply human labels only to subsequent evaluation cycles.

Do not automatically join every image sharing an encounter. The installed dataset contains encounters with multiple animals. A verified single-animal tracklet can provide a useful constraint; an encounter ID by itself cannot.

Visible flank is useful context. A published flank classifier reached 88.70% on lynx, which is side-classification accuracy, not identity accuracy. Its remaining errors illustrate why an uncertain side prediction should not hard-block plausible matches.[^flank] Keep left/right reference views where supported, and merge them into an individual only with linking evidence.

### Clustering metrics

Report pairwise precision and recall over all input images, ARI, identity fragmentation, cluster size distributions, and non-singleton coverage. Treat each unassigned image as a separate predicted label for scoring; never score all unresolved images as one identity.

High purity obtained by assigning everything to singletons is not useful clustering. Conversely, high coverage with mixed-identity groups is actively misleading. Calibrate the tradeoff against review workload, including large-cluster audits because a single incorrect bridge can create many incorrect implied pairs.

## 6. Bounded experiment programme

Each stage has a decision gate. Do not launch a full combination of models, GMM sizes, transforms and graph settings.

| Stage | Controlled work | Decision criterion |
|---|---|---|
| 0. Protocol and diagnostics | New fit/calibration/selection manifests; stratified metrics; retained existing results | No overlap or training-label leakage; every metric has a clear denominator |
| 1. Frozen global comparison | Segmented MegaDescriptor vs MiewID | Candidate recall, known top-1 and cross-camera/mode performance |
| 2. WildFusion reference | Independent DISK/ALIKED matching; per-signal calibration; 20/50 candidates | Improvement over current custom fusion at a measured pair budget |
| 3. Appearance ablation | RGB vs calibrated RGB+gray; then one foreground contrast variant | Cross-mode gains without material overall or subgroup regression |
| 4. Graph repair | Thresholded graph; mutual edges; conservative clustering and attachment | Pair precision/recall/coverage together; no giant appearance-only groups |
| 5. Fine-tuning | Best frozen backbone; baseline vs camera/mode-aware training | Gains on unseen identities and cameras, not only familiar animals |
| 6. Final qualification | Locked temporal test; separate geographic and predicted-mask evaluations | Pilot gates and uncertainty reported on untouched data |

Graph experiments can reuse saved embeddings and pair evidence. New local evidence still costs computation: at 560 queries, 50 candidates and two matchers, the upper budget is **56,000 matcher-pair evaluations**, before calibration and query–query work. The previous ten-candidate single-matcher budget was 5,600. Report these units explicitly; library-specific budget parameters may count work differently.

Keep PCA 128/GMM 256 fixed while testing the main hypotheses. If Fisher remains useful after calibrated global/local fusion, compare separate Fisher score fusion and only then a small 64/128/256-component sweep. Remove a branch if it adds cost without held-out benefit.

Store compact pair evidence once and reuse it for scoring/threshold changes. Avoid dense all-pairs matching on the full 39,760 images: there are about 790 million unordered pairs. Use bounded candidate graphs and chunked retrieval. All tests, training and benchmark commands should be handed to the operator to run; runtime estimates should follow measured throughput.

## 7. Proposed implementation and verification

The following is a reviewable change outline, not implemented pipeline functionality. Existing defaults and cached historical reports should remain reproducible.

| Files or proposed module | Planned change and purpose |
|---|---|
| `reid_demo/benchmark.py` | Add explicit scoring strategies, calibrated fusion and wider candidate handling; keep ranking label-free; add subgroup metrics and confidence/coverage reporting |
| `tools/evaluate_czechlynx.py` | Wire model/scoring choices, frozen calibration artifacts and separate fit/selection manifests; remove Fisher as a prerequisite for a WildFusion-only path |
| New `reid_demo/benchmark_embeddings.py` | Shared frozen MegaDescriptor/MiewID adapters and checkpoint/transform-aware caches |
| New `reid_demo/benchmark_matching.py` | Independent local matcher branches, foreground-aligned coordinates, confidence/coverage evidence and versioned pair caches |
| `calibration.py` and benchmark adapter | Reuse calibrators after checking monotonic interpolation, clipping, degenerate inputs and serialization; distinguish similarity calibration from rejection |
| `reid_demo/cluster.py`, `reid_demo/fusion.py` | Optional thresholded/mutual sparse graph, uncertainty handling, conservative merge checks and truthful backend reporting |
| `reid_demo/benchmark_report.py` | Cross-mode/camera/flank metrics, candidate ceilings, accepted precision, cluster contamination and review evidence |
| `utils/train_megadescriptor_scratch.py` plus dataset/sampler helpers | Safe CzechLynx fine-tuning adapter, protected holdouts, encounter/camera/modality sampling; separate MiewID training adapter if selected |
| `tests/test_benchmark*.py`, `tests/test_cluster.py`, focused new tests | Verify data/score/cache/graph contracts with small fixtures and mocked models |
| `docs/czechlynx-benchmark.md`, `AGENTS.md` | Document approved flags, metric definitions, cache versions, split rules and actual capabilities |

Implementation should be approved and delivered in three manageable batches: protocol/metrics and frozen global adapters; calibrated local fusion and graph handling; then fine-tuning and field-mask qualification. Each batch should include its concrete diff and exact runnable commands.

| Verification area | Meaningful checks |
|---|---|
| Splits | Encounters cannot cross fitting/evaluation boundaries; reserved unknown identities never enter encoder training; mixed-animal encounter handling is explicit |
| Label independence | Permuting evaluation identity labels does not change rankings or inferred clusters |
| Model adapters | Correct input size/normalization and output dimension; checkpoint/transform changes invalidate caches |
| Mask alignment | Crops/resizes map keypoints and masks consistently; missing masks and empty foreground are handled explicitly |
| Calibration | Monotonic bounded output, duplicate/constant scores, train-only fitting, serialization, and invalid artifact rejection |
| Candidate handling | Correct item below old rank ten can be promoted; skipped matching differs from attempted zero evidence; comparisons use coherent score scales |
| Graph logic | Weak unrelated pairs remain disconnected; isolates survive; reciprocal edges and conflicting constraints behave correctly; weak bridges do not silently join reviewed groups |
| Metrics | Hand-computable unknown acceptance, subgroup denominators, pair precision/recall, and unique noise labels |
| Fine-tuning | Sampler uses valid positives; holdouts are excluded; flips update side labels; checkpoint reload reproduces embeddings |
| Empirical qualification | Real saved predictions, cluster audits and user-run benchmarks establish accuracy; mocked tests do not |

## 8. Proposed pilot gates

These are engineering targets for discussion, not published standards or demonstrated results. Evaluate them on a frozen test and report confidence intervals. A headline score passing on the small development subset alone is insufficient.

| Capability | Proposed minimum for a limited review pilot |
|---|---|
| Known-identity suggestions | At least **50% top-1** and a working target of **75% top-5** |
| Accepted known-identity links | At least **95% precision**, counting accepted unknowns as errors; target at least **25% correct-accept coverage** of known queries |
| Unknown handling | At most **5% false attachment** on a representative unknown set; report precision as unknown prevalence changes |
| Useful clustering | At least **95% pair precision**, **50% pair recall**, **60% non-singleton image coverage**, and **ARI 0.5** as a working target |
| Acquisition robustness | Cross-camera and cross-mode results improve on the frozen control with adequate subgroup counts; unresolved cross-mode collapse blocks qualification |
| Field input | Predicted-mask evaluation, failure reporting and representative manual audits |
| Review value | Measurably less work than manual sorting; target a **30% reduction in review time** on a matched, blinded task |
| Data integrity | Persistent reviews and collision-safe multi-camera ingestion |

The existing pilot-readiness assessment identified basename collisions and ingestion updates that could reset downstream information or human reviews. Those remain prerequisites before real multi-camera ingestion, independently of recognition quality.[^pilot]

The immediate milestone is a trustworthy segmented benchmark with stronger candidate retrieval and independently calibrated scores. Fine-tuning follows a useful frozen reference, and conservative clustering follows measured pair evidence. This sequence targets the observed failure mechanisms while making it possible to stop spending compute on changes that do not improve held-out results.

## Sources

Source notes below preserve the distinction between repository evidence, published experiments and proposed engineering choices. Web sources were checked on 12 September 2026. Published numbers from different datasets and metrics are not pooled.

1. **Saved CzechLynx evaluations.** [Bbox report](../../evaluations/czechlynx/dev_bbox_ensamble/index.html), [segmented report](../../evaluations/czechlynx/dev_masked_ensamble/index.html), [GV-mask report](../../evaluations/czechlynx/dev_bbox_gv_mask_ensamble/index.html). Underlying `comparison.csv`, prediction JSONs and clustering JSONs in each directory. File hashes and derived counts: [analysis JSON](czechlynx-analysis-2026-09-12.json).
2. **Repository implementation.** [Ranking and metrics](../../reid_demo/benchmark.py), especially `rank_queries` and `retrieval_metrics`; [runner](../../tools/evaluate_czechlynx.py), especially `combine_fisher_matrices` and `run_clustering`; [graph construction](../../reid_demo/cluster.py), `_leiden_labels`; [fusion](../../reid_demo/fusion.py); [mask handling](../../reid_demo/benchmark_masks.py). Working-tree snapshot, 12 September 2026.
3. **Foreground appearance diagnostic.** [Per-image CSV](czechlynx-lighting-diagnostics-2026-09-12.csv) and [aggregate JSON](czechlynx-analysis-2026-09-12.json). Includes 1,310 existing gallery/query images, colour-proxy definition, candidate coverage, acceptance counts and graph bound. Threshold sensitivity: cutoffs 3, 5 and 8 classify 434, 434 and 436 images respectively as near-monochrome. No sensor labels inferred.
4. **Picek et al.** [CzechLynx: A Dataset for Individual Identification and Pose Estimation of the Eurasian Lynx](https://www.nature.com/articles/s41597-026-06853-9). Scientific Data, 2026; accessible [arXiv v2 text](https://arxiv.org/html/2506.04931v2). [Official Kaggle dataset card](https://www.kaggle.com/datasets/picekl/czechlynx), version-update note, metadata and split tables.
5. **Requested notebook references.** Lukas Picek, [CzechLynx baseline with WildFusion](https://www.kaggle.com/code/picekl/czechlynx-baseline-with-wildfusion); mdnomanhossain, [CzechLynx base open ImageNet](https://www.kaggle.com/code/mdnomanhossain/czechlynx-base-open-imagenet). Cells, outputs and exact version dates unavailable; no notebook-specific experimental claims used.
6. **Čermák et al.** [WildFusion](https://arxiv.org/abs/2408.12934), 2024, method sections 3.2–3.4. Full text also available in [local paper](../papers/wildfusion.pdf).
7. **WildlifeDatasets.** [WildFusion documentation](https://wildlifedatasets.github.io/wildlife-tools/wildfusion/) and [Wildlife Tools repository](https://github.com/WildlifeDatasets/wildlife-tools). Matcher pipelines, calibration and shortlist examples.
8. **Existing reusable project code.** [WildFusion reference script](../../test-scripts/run_wildfusion_paper_baseline.py), [calibration classes](../../calibration.py), [MegaDescriptor trainer](../../utils/train_megadescriptor_scratch.py). These require protocol adaptation, not direct reuse of their evaluation splits.
9. **Picek, Neumann and Matas.** [Animal Identification with Independent Foreground and Background Modeling](https://arxiv.org/html/2408.12930v1), 2024, section 4.1 and Table 1. Foreground/background and new-location comparison.
10. **Conservation X Labs / Wild Me.** [MiewID-msv3 model card](https://huggingface.co/conservationxlabs/miewid-msv3), [readable card source](https://huggingface.co/conservationxlabs/miewid-msv3/raw/main/README.md). Architecture, preprocessing, training provenance and Whiskerbook lynx metrics; exact CzechLynx overlap unresolved.
11. **El Bassat et al.** [Calibrated Similarity and Graph Clustering for Open-Set Animal Re-Identification](https://arxiv.org/html/2608.02469v1), arXiv working paper, August 2026, sections 3.2–3.8 and Tables 3–5. AnimalCLEF aggregate ARI; a recent preprint rather than independent validation on this repository.
12. **MIA AI Team.** [AnimalCLEF26 implementation](https://github.com/MIA-AI-Team/AnimalCLEF26), README and file inventory. Reference for model, calibration and clustering integration.
13. **Ye, Ruan, Du and Shou.** [Channel Augmented Joint Learning for Visible-Infrared Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Ye_Channel_Augmented_Joint_Learning_for_Visible-Infrared_Recognition_ICCV_2021_paper.html). ICCV 2021, pp. 13567–13576. Human visible–infrared evidence used only as an augmentation hypothesis.
14. **Suessle et al.** [CNN Based Flank Predictor for Quadruped Animal Species](https://arxiv.org/abs/2406.13588), 2024. Lynx flank classification and bilateral-asymmetry context.
15. **Zhong et al.** [Re-ranking Person Re-identification with k-reciprocal Encoding](https://arxiv.org/abs/1701.08398), CVPR 2017. Mutual-neighbour reranking; animal application requires validation.
16. **Project pilot assessment.** [Pilot readiness, 10 September 2026](../pilot-readiness-2026-09-10.md). Ingestion defects, provisional identity/counting outputs and pilot limitations.

[^local]: [Saved-run analysis and file hashes](czechlynx-analysis-2026-09-12.json); [segmented benchmark report](../../evaluations/czechlynx/dev_masked_ensamble/index.html). Full inventory: source 1.
[^implementation]: [Benchmark ranking/metrics](../../reid_demo/benchmark.py), [runner](../../tools/evaluate_czechlynx.py), [graph construction](../../reid_demo/cluster.py). Full inventory: source 2.
[^diagnostic]: [Per-image appearance CSV](czechlynx-lighting-diagnostics-2026-09-12.csv) and [aggregate analysis](czechlynx-analysis-2026-09-12.json). Method and limitations: source 3.
[^dataset]: Picek et al., [CzechLynx dataset paper](https://arxiv.org/html/2506.04931v2); [dataset card](https://www.kaggle.com/datasets/picekl/czechlynx). Full details: source 4.
[^wildfusion]: Čermák et al., [WildFusion](https://arxiv.org/abs/2408.12934), 2024. Full details: source 6.
[^wildtools]: WildlifeDatasets, [WildFusion documentation](https://wildlifedatasets.github.io/wildlife-tools/wildfusion/). Source 7.
[^reusable]: [WildFusion reference script](../../test-scripts/run_wildfusion_paper_baseline.py), [calibration](../../calibration.py), [trainer](../../utils/train_megadescriptor_scratch.py). Source 8.
[^foreground]: Picek, Neumann and Matas, [foreground/background study](https://arxiv.org/html/2408.12930v1), 2024, Table 1. Source 9.
[^miewid]: Conservation X Labs / Wild Me, [MiewID-msv3 card](https://huggingface.co/conservationxlabs/miewid-msv3/raw/main/README.md). Source 10.
[^animalclef]: El Bassat et al., [AnimalCLEF working paper](https://arxiv.org/html/2608.02469v1), 2026, preprocessing section and Table 3. Source 11.
[^animalcode]: MIA AI Team, [AnimalCLEF26 implementation](https://github.com/MIA-AI-Team/AnimalCLEF26). Source 12.
[^illumination]: Ye et al., [Channel Augmented Joint Learning](https://openaccess.thecvf.com/content/ICCV2021/html/Ye_Channel_Augmented_Joint_Learning_for_Visible-Infrared_Recognition_ICCV_2021_paper.html), ICCV 2021. Source 13.
[^rerank]: Zhong et al., [k-reciprocal reranking](https://arxiv.org/abs/1701.08398), CVPR 2017. Source 15.
[^flank]: Suessle et al., [CNN Based Flank Predictor](https://arxiv.org/abs/2406.13588), 2024. Source 14.
[^pilot]: [Project pilot-readiness assessment](../pilot-readiness-2026-09-10.md). Source 16.
