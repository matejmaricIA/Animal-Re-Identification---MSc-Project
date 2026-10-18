# Pilot readiness assessment — 10 September 2026

**22 September requirement update:** The user requires reliable clustering and defensible assessment of photographed individual numbers for the lynx demo; assisted matching alone is insufficient. Upstream MiewID exposure, the lack of local-feature gain and nighttime/IR handling remain unresolved. Read the [current decision log](lynx-demo-decision-log.md). The historical exploratory-outreach recommendation below does not establish demo readiness or replace these requirements.

The project is a working research prototype with substantial pilot functionality. It is ready for exploratory outreach and a tightly scoped feasibility collaboration. Data-integrity fixes, field validation, and a refreshed demonstration are needed before a park relies on its results. The first pilot should be operated by the researcher, with expert review.

## Evidence checked

- Inspected reid_demo/, tests, saved evaluations, the demo bundle, status board, and thesis limitations.
- Ran `venv/bin/python -m pytest tests -q`: **302 passed**, 19 warnings, 94 seconds. Warnings included expected missing-image placeholders and a LightGlue deprecation. Model-dependent tests commonly use mocks; passing tests do not establish field accuracy or a fresh installation.
- Regenerated the Medvednica report from existing JSON under `/tmp/reid-pilot-readiness-20260910/medvednica`: 8,208 input photos, 2,031 frames without detections, and 5,801 frames with animal detections. These are model-output counts, not verified accuracy measurements. Detection/classification models were not rerun.
- Reran current `leiden` defaults on 1,388 cached CowDataset embeddings, without changing the source database: **18 clusters versus 12 known individuals**, no unassigned crops, **ARI 0.7009**, V-measure 0.8053. The backend supports a Louvain fallback; this assessment did not distinguish which implementation was available. Output: `/tmp/reid-pilot-readiness-20260910/cow_current_clustering.json`. Embedding extraction was not rerun.
- Reproduced camera filename collisions and review erasure in temporary fixtures. No real records were changed.

## What already exists

SQLite storage, ingestion adapters, animal crops, species filtering, global embeddings, local/Fisher features, geometric verification, several clustering backends, portable HTML catalogues, evaluation reports, a basic browser review interface, and a demo runner are implemented. This is substantial reusable engineering. The M1 board still marks tickets as in review; M2 remains a backlog.

The review module can confirm, reject, split, and merge assignments. Its default queue checks uncertain crops against members of their existing cluster. The broader workflow that searches across clusters for ranked candidate matches remains proposed in `docs/design-notes/two-mode-reid.md`.

## Gaps that matter

1. **Camera-safe identifiers.** Ingestion reduces source paths to basenames; record IDs use only the stem and detection index. Both `cam_a/IMG_0001.JPG` and `cam_b/IMG_0001.JPG` become `IMG_0001__crop1`. Preserve source paths and scope identifiers to project/camera/deployment or equivalent stable provenance. Detect duplicates explicitly.

2. **Safe repeated ingestion.** Store upserts replace downstream fields with incoming defaults. Re-ingesting a reviewed record changed `review_status=confirmed`, `cluster_id=7`, and an embedding reference to `unreviewed`, `None`, and `None`. Clustering's review-preservation guard does not protect ingestion. Acceptance: adding/retrying batches preserves prior decisions and derived state unless explicitly invalidated.

3. **Species report credibility.** The Medvednica report calls 59 predicted labels species found and includes coyote, mule deer, fisher, and other questionable local assignments. Distinguish predictions, uncertainty, and expert-confirmed observations. The live SpeciesNet wrapper already accepts `country="HRV"`; the report uses older artifacts. Validate the actual inference/output path. Review samples of rejected as well as retained images before claiming detection recall.

4. **Clusters are not verified animal counts.** The old cow demo reports 182 groups for 12 identities and 554 possible-new crops. Current clustering improves this to 18 groups but still fragments identities. ARI 0.70 is not 70% identification accuracy. The percentage called photos correctly grouped is cluster purity, which can improve by fragmenting identities. Report merge/split errors, unresolved cases, and verified identities alongside clustering metrics.

5. **Flanks require care.** Left/right flanks are clustered separately; raw-image ingestion sets orientation to unknown. A flank catalogue needs reliable side annotation and evidence linking opposite sides before it becomes a unique-individual catalogue. Do not sum independent left/right groups as unique animals.

6. **Field validation and usefulness.** The thesis says the intended Medvednica individual-ID study was not completed because of technical issues and insufficient good-quality images. No verified Croatian lynx performance or measured reviewer time saving was found in the inspected artifacts. Test different visits, cameras, day/night conditions, and unknown animals; avoid evaluating mainly near-identical burst frames. Hold out examples from threshold selection.

7. **Counting and scale.** Legacy `--count` uses identity labels as a simulated human oracle, not a real annotator workflow. Photographed-individual counts do not establish total park abundance/density, as the thesis explicitly acknowledges. Current clustering constructs dense pairwise matrices, so large archives need batching or different neighbor search. Neither abundance modeling nor archive-scale deployment is a prerequisite for a small matching pilot.

## Recommended first offer

“I can process a small sample of your existing camera-trap photographs, organize animal detections, and test whether suggested individual matches reduce expert review time. I will return a reviewable catalogue, a spreadsheet of observations, and an evaluation of accuracy and limitations.”

Choose one partner, one species, and a bounded collection. Lynx is relevant because individual identification from coat patterns is already used in monitoring; this implementation's performance still needs measurement. If matching is weak, determine whether triage and searchable observations alone solve a useful problem. Do not promise quantified time savings before measuring them.

Suggested sample: 500–2,000 photos, or all available target-species encounters if rarer, retaining original timestamps, camera/deployment IDs, and encounter grouping. Include different visits, ordinary poor-quality examples, and a small expert-identified subset or same/different pair labels. Exact coordinates need not be transferred initially. Agree on access, sharing, retention, and deletion.

Return a portable catalogue, CSV observations with source references and review status, unresolved matches, and a short evaluation. Confirm an export format that fits the partner's current workflow.

Measure review time against their process, missed target-species detections, precision of suggested/accepted matches, correct matches in the top five suggestions, and manual workload. Agree on acceptable error/workload and set thresholds before held-out evaluation. A useful outcome can be a narrower workflow or evidence that the imagery cannot support reliable individual identification.

## Order of work

1. Begin exploratory contact with the limited offer and concrete data request. Learn the institution's current workflow and bottleneck.
2. Before multi-camera ingestion, fix identifier collisions and review preservation. Verify repeated filenames, repeated ingestion, and reviewed records.
3. Refresh the demonstration using reviewed Medvednica examples and a relevant lynx benchmark. Replace unverified species/count headlines with measured outputs and uncertainty.
4. Complete ranked-candidate review and observe an expert using it. Keep automatic merges conservative and uncertain cases unresolved.
5. Run the bounded partner pilot; expand based on measured usefulness. Further model training, multi-user hosting, and density estimation can follow demonstrated need.

## Relevant lynx data is already available

The roadmap describes leopards as the closest available proxy. The published **CzechLynx** dataset now provides Eurasian lynx images and identities with geographic and temporal evaluation protocols. The 2026 paper reports 39,760 camera-trap images covering 319 individuals and links public data and baseline code. Use its real-image protocols, document model-training overlap, and reserve park data for external validation. No dataset was downloaded or evaluated here. [Dataset paper](https://www.nature.com/articles/s41597-026-06853-9).

## Outreach targets

- Start with the people responsible for the existing Medvednica collection if that connection is available.
- Plitvice describes camera traps used by its professional service for large-carnivore monitoring, with stored photographs for analysis. [Park description](https://np-plitvicka-jezera.hr/kamere-za-pracenje-zivotinja-u-np-plitvicka-jezera/).
- Risnjak remains a sensible candidate from the existing project plan. Include researchers as well as park administrations: the University of Zagreb Veterinary Faculty describes Croatian lynx monitoring using cameras, telemetry, and DNA. Contact the monitoring team without assuming the older LIFE Lynx project is still accepting collaborations. [Faculty account](https://www.vef.unizg.hr/novosti/ispustanjem-risa-u-nacionalnom-parku-plitvicka-jezera/).

Contact conservation/research staff with a small concrete offer. Existing monitoring programs indicate a relevant workflow, not an agreement to share data or a demonstrated need for this tool.
