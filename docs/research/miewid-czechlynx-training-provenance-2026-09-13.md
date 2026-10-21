# MiewID training provenance and CzechLynx

## Conclusion

**The public evidence does not establish whether the exact MiewID-msv3 weights used in the CzechLynx benchmark were trained on any of its photographs or individuals. It also does not establish that those data were excluded.** There is direct evidence that an earlier MiewID release trained on Eurasian lynx, and strong reason to avoid presenting this model family as having no prior lynx exposure. The remaining uncertainty concerns which animals and source collections contributed to the specific v3 checkpoint.

The 51.49% development top-1 result remains a measured result on the saved benchmark. Its interpretation should be “frozen pretrained MiewID on CzechLynx, upstream overlap unverified.” It is not yet evidence of 51.49% recognition on animals independently verified as absent from all model training.

Two findings materially improve the evaluation plan. First, the locally cached model file matches the publisher’s original October 2024 weight artifact. Second, the full prepared dataset contains 1,614 photographs whose recorded dates are later than that upload. Verified capture dates would make those photographs useful for testing recognition on images that could not have trained these fixed weights. They would not automatically establish that the photographed individuals were unseen.

## Evidence and confidence

| Question | Finding | Confidence and boundary |
| --- | --- | --- |
| Did the benchmark use the public frozen v3 release? | The saved revision identifies v3; the cached weight-file checksum matches the publisher’s artifact. | Confirmed for the saved run identity and inspected cached file. |
| Has MiewID trained on Eurasian lynx? | The v2 publisher card explicitly lists lynx training annotations and names. | Confirmed for v2; the counts must not be assigned to v3. |
| Did v3 train on CzechLynx photographs? | No reviewed source provides a CzechLynx image-level inclusion or exclusion statement. | Unresolved. |
| Did v3 train on the same animals through another collection? | There is no published identity crosswalk to CzechLynx. | Unresolved, even if exact photographs were excluded. |
| Does CzechLynx’s later publication prove independence? | A related private Czech collection existed before the checkpoint upload. | No; publication chronology is insufficient. |
| Does the 51.49% development sample contain demonstrably later photographs? | No dated gallery or query image is later than December 2021. Six query dates are missing. | No temporal exclusion argument for this sample. |
| Could later CzechLynx photographs help? | Metadata identifies 1,614 images after 15 October 2024, across 13 identities. | Promising, conditional on capture-date validation; limited source and identity coverage. |
| Does Croatian provenance guarantee independent pilot data? | Croatian lynx photographs were already uploaded to Whiskerbook before v3. | No; platform presence is not evidence of training inclusion either. |

The distinctions matter. Training on the same **species** is expected for a specialist wildlife representation. Training on the same **individuals** affects claims about transfer to new animals. Training on the same **photographs**, including alternate crops or masks, threatens image-level independence. These are different questions and require different evidence.

## Exact checkpoint identity

The saved benchmark manifest identifies **conservationxlabs/miewid-msv3**, pinned to revision **4f1d7f2b521149e5fe34bb85f377248ce9971a7d**. The publisher’s history traces its weight file to upload commit **5ff899ef579bebdabc9482aafabe96cfac027060**, dated **15 October 2024**. Subsequent changes through the pinned revision concern the README. The release contains weights, configuration and model code, but no image-level training inventory.[^1]

The publisher’s model.safetensors SHA-256 is:

```text
adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348
```

The cached file produces exactly that checksum.[^2] This provides a specific artifact against which capture dates and any future provenance statement can be checked. A generic answer about “MiewID” would be insufficient because the project has multiple releases and downstream fine-tuned variants.

The benchmark also records a model-state hash:

```text
8bb56ea54bc45fa7ed0c33a60623a49f2944fb5444d373955a4919d6a2666168
```

That identifies the loaded model state under the benchmark’s hashing convention; it is not the serialized-file checksum. The differing values are not a mismatch. The [audit receipt](miewid-czechlynx-training-provenance-2026-09-13.json) retains both values, the revision and the input-file hashes.

An upload date is a conservative upper bound on when these weight bytes existed. It is not a published training cutoff. Photographs verifiably captured after that date could not have been training inputs to those bytes.

## Training evidence across versions

The **msv2** model card has explicit columns for training and test counts. Its Eurasian lynx row lists **2,251 training annotations and 285 training names**, with **726 test annotations and 199 test names**, sourced from Wildbook and covering left/right views. This is direct evidence of lynx training in that release. “Names” should not be silently converted into a count of distinct biological animals.[^3]

The original research paper describes a mixture of community-contributed and public datasets. Its Table 1 includes **4,716 Eurasian lynx annotations and 1,642 identity entries** in the collected dataset. These are not a statement that every listed annotation was used for training; the paper describes subsequent splits and treating opposite viewpoints separately. The acknowledgements identify carnivore-data coordination and multiple contributing organisations, but do not provide a CzechLynx image inventory or map each lynx photograph to its contributor.[^4]

The publisher later described the paper’s model as **v2**, distinguishing it from v3 and v4. That prevents using the paper’s dataset totals as an exact v3 training manifest.[^5]

The **msv3** card describes training across 64 wildlife species, including community data from Wildbook platforms. Its evaluation table contains an Eurasian lynx row sourced to Whiskerbook, reporting 69.8% rank-1. Unlike the v2 card, this row does not expose per-species training-image and identity counts. An evaluation result alone cannot prove exactly what was used for training, and 69.8% on that evaluation is not a comparable CzechLynx target.[^6]

Taken together, these sources strongly support treating v3 as a model with likely prior lynx exposure, rather than claiming unseen-species transfer. They do not identify whether that exposure includes FoE Czech Republic, Šumava National Park, or the animals in the current benchmark.

The public implementation does not close this gap. Its default training configuration references an external annotation file under a private filesystem path, while the repository excludes several split and dataset-file formats. That default is not established as the v3 training recipe. Deployment configuration routes lynx to an older specialist model in some entries; routing configuration is not a list of the v3 training examples.[^7]

A Hugging Face discussion comparing msv2 and msv3 contains a third-party interpretation of their cards, rather than a publisher-supplied provenance statement.[^8] Later projects that fine-tune a model on CzechLynx are likewise not evidence about the origin of the fixed October 2024 artifact.

## Czech data history

The CzechLynx preprint was first submitted on **5 June 2025**; the Scientific Data article appeared on **24 February 2026**. Its current dataset combines FoE Carpathians, FoE Bohemia and Šumava National Park collections, covering photographs from years before and after the checkpoint release.[^9]

However, a related paper, **Animal Identification with Independent Foreground and Background Modeling**, appeared on **23 August 2024**. It describes a West Carpathian lynx collection of **6,743 images and 103 individuals**, with access requiring an agreement with Friends of the Earth Czech Republic. The CzechLynx paper identifies that earlier study as using part of its FoE data.[^10][^9]

| Date | Event | Relevance |
| --- | --- | --- |
| August 2024 | Related private Czech lynx collection documented | Source imagery existed before the checkpoint release. |
| 15 October 2024 | Exact v3 weight artifact uploaded | A usable upper bound for excluding genuinely later photographs. |
| June 2025 | CzechLynx preprint submitted | Later public documentation does not rule out prior private sharing. |
| February 2026 | CzechLynx journal article published | Publication date is not the image collection or first research-use date. |

There is no evidence here that the earlier private collection was shared with MiewID’s developers. Equally, restricted access does not prove that no authorised sharing occurred. The defensible conclusion is that release chronology alone cannot answer the overlap question.

A source-level answer would require confirmation about all three contributing projects, including older exports, alternative collection names, re-labelled identities and derivatives of original images. Searching only for the later “CzechLynx” dataset name could miss a contribution supplied under an organisation’s name.

## Local capture-date audit

The following counts come from existing CSV metadata and the completed run manifest. No model was run to obtain them. Full input paths, checksums and audit counts are in the accompanying [JSON receipt](miewid-czechlynx-training-provenance-2026-09-13.json).

| Collection | Images | Missing dates | Latest recorded date | Dated after 15 October 2024 |
| --- | ---: | ---: | --- | ---: |
| Saved development gallery | 750 | 0 | 10 December 2021 | 0 |
| Saved development queries | 560 | 6 | 28 December 2021 | 0 |
| Full prepared masked collection | 39,760 | 413 | 12 October 2025 | 1,614 |

All 1,614 later-dated images belong to **foe_carpaths** and the official **time_open test** split. They span **13 identities and 833 encounter identifiers**. There are **394 images from 2024 and 1,220 from 2025**. Those counts describe available records; they are not the number of statistically independent recognition trials.

The raw day-month-year dates were independently parsed and compared with the prepared ISO dates for all 1,614 candidate images. There were **zero conversion mismatches**. This confirms consistency between the two metadata representations, not the correctness of the original camera clocks.

The dataset also contains a recorded **2 January 2006** date despite its published 2009–2025 collection span. That discrepancy should remain visible and be checked against original records, rather than silently repaired. The proposed temporal exclusion claim needs validated acquisition dates, ideally corroborated by original files, camera overlays, encounter records or curator confirmation.

Of the 1,614 images, **1,194 depict identities also present in the official training partition**. The remaining 420 are not thereby proven absent from MiewID training. “Not in CzechLynx’s official train split” and “never seen by MiewID” are separate properties.

These records should stay reserved for a later, locked evaluation. Their metadata was inspected for provenance; recognition outcomes were not evaluated. Choosing model settings from repeated experiments on them would consume their value as a final test.

## Relevance to a Croatian pilot

A LIFE Lynx C5 report hosted in April 2024 states that photographs of **68 Slovenian and 106 Croatian lynx individuals** had been uploaded to Whiskerbook, covering Slovenian monitoring from 2018–2023 and Croatian monitoring from 2021–2023. This is evidence of platform participation before v3, not proof that the material was selected for model training.[^11]

The practical implication is that “obtained from a Croatian institution” cannot serve as the entire independence check. A partner could provide images already present on a model supplier’s platform. The provenance question should cover prior uploads and model-development permissions, while a prospective collection of genuinely new photographs offers a clearer image-level boundary.

Croatian researchers have also published **Spotted correctly? Human expertise remains essential in AI-assisted identification of Eurasian lynx**, with authors affiliated with the University of Zagreb. The publisher’s abstract and associated public image resource establish a relevant local research connection. They do not establish what trained the October 2024 weights.[^12]

These connections are useful for finding knowledgeable collaborators. They should not be used to imply that any institution supplied CzechLynx data to MiewID without a source stating that relationship.

## Evaluation implications and next decisions

**Keep frozen MiewID as the development baseline.** The saved benchmark shows a substantial improvement on identical segmented inputs, and upstream uncertainty does not make the measured ranking counts disappear. Record that uncertainty alongside every claim about generalisation.

**Keep upstream provenance separate from local split protection.** Independent fitting, calibration and model-selection partitions remain necessary for the next implementation phase. They prevent leakage introduced by subsequent work in this repository. They cannot undo exposure that may already be embedded in a pretrained checkpoint.

**Reserve a temporal test before choosing further settings.** A proposed protocol should freeze the exact model artifact, verify the candidate acquisition dates, define eligible gallery/query encounters and report known/unknown identity composition. It should preserve official assignments and explain any explicit exclusions. A result on these later photographs would test image-level novelty, not automatically new-individual or new-location transfer.

**Evaluate new individuals separately where provenance permits.** A curator-confirmed identity crosswalk, or a prospective partner collection with documented prior sharing, could strengthen that claim. The 13 identities available in the later-date subset are too narrow to support a broad claim about all parks, cameras and lighting conditions.

**Do not merge recognition, acceptance and clustering claims.** The current development run has 156/303 correct top-1 known queries, but its in-sample accepted-match precision is 97/118 and clustering was disabled. Even a favourable provenance answer would not validate autonomous clustering or population estimates. The [existing results review](czechlynx-miewid-results-2026-09-13.md) remains the source for those separate limitations.

No benchmark change is required to preserve these findings. Implementation should begin with the concrete phase 2 proposal already tracked in the [implementation plan](../czechlynx-implementation-plan.md), with provenance checks added to the final evaluation design. Test and benchmark execution remains with the project operator.

## Author clarification needed

An answer that materially resolves the uncertainty should refer to the exact v3 artifact and distinguish training, validation and evaluation data. The most useful evidence would be a contributor inventory with acquisition cutoffs, image hashes or stable annotation identifiers, and an identity crosswalk. If images cannot be shared publicly, a documented private overlap check or source-exclusion statement can still be useful.

The MiewID publisher can establish which source exports entered the checkpoint’s training. CzechLynx’s curators can establish whether source collections were previously supplied to Wild Me, Conservation X Labs, Whiskerbook or other research collaborators. A statement about public dataset downloads alone would not settle the earlier private-data question.

Suggested unsent clarification:

> We are evaluating conservationxlabs/miewid-msv3 at revision 4f1d7f2b521149e5fe34bb85f377248ce9971a7d, whose weight-file SHA-256 is adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348. Did its training, validation or model-selection data include photographs or individuals from the FoE Czech Republic Carpathian/Bohemian collections or Šumava National Park, including the earlier collection used in Animal Identification with Independent Foreground and Background Modeling? If so, can you provide the relevant sources, capture-date limits and an image/identity overlap check against CzechLynx? We need to distinguish unseen photographs from unseen individuals.

No outreach is recorded as completed. The public evidence supports an explicit uncertainty statement and a stronger evaluation design; it does not support a definitive “yes” or “no” about CzechLynx overlap.

## Sources

Sources and local records reviewed through 13 September 2026. Links identify original publishers or author-provided papers. Access limitations are recorded where relevant.

[^1]: Conservation X Labs. [MiewID-msv3 upload commit](https://huggingface.co/conservationxlabs/miewid-msv3/commit/5ff899ef579bebdabc9482aafabe96cfac027060), 15 October 2024; [commit history](https://huggingface.co/conservationxlabs/miewid-msv3/commits/main); [release file inventory](https://huggingface.co/conservationxlabs/miewid-msv3/tree/main). Used for release identity, chronology and available artifacts.
[^2]: Conservation X Labs. [MiewID-msv3 model.safetensors file metadata](https://huggingface.co/conservationxlabs/miewid-msv3/blob/main/model.safetensors). Publisher SHA-256 compared with the local cached file; comparison recorded in the accompanying JSON.
[^3]: Conservation X Labs. [MiewID-msv2 model card](https://huggingface.co/conservationxlabs/miewid-msv2), Eurasian lynx row under Detailed results. Explicit annotation/name training and test counts.
[^4]: Lasha Otarashvili, Tamilselvan Subramanian, Jason Holmberg, J.J. Levenson and Charles V. Stewart. [Multispecies Animal Re-ID Using a Large Community-Curated Dataset](https://arxiv.org/abs/2412.05602), December 2024, Sections 3.2–3.3, Table 1 and acknowledgements. Table text also available in the [author-uploaded paper](https://www.researchgate.net/publication/386577080_Multispecies_Animal_Re-ID_Using_a_Large_Community-Curated_Dataset). The table describes the collected dataset, not v3’s exact training split.
[^5]: Wild Me. [MiewID v4 Announcement](https://community.wildme.org/t/miewid-v4-announcement/5406), 30 January 2026. Identifies the paper’s model as v2 and distinguishes later versions.
[^6]: Conservation X Labs. [MiewID-msv3 model card](https://huggingface.co/conservationxlabs/miewid-msv3), Training Data and Evaluation Results. Public description of training sources and Eurasian lynx evaluation; no image-level source manifest supplied.
[^7]: Wild Me. wbia-plugin-miew-id: [default training configuration](https://github.com/WildMeOrg/wbia-plugin-miew-id/blob/main/wbia_miew_id/configs/default_config.yaml), [dataset-related exclusions](https://github.com/WildMeOrg/wbia-plugin-miew-id/blob/main/wbia_miew_id/.gitignore), and [deployment model configuration](https://github.com/WildMeOrg/wbia-plugin-miew-id/blob/main/wbia_miew_id/configs/model_config.json). Inspected repository state; these files are not a verified recipe for the pinned v3 release.
[^8]: Hugging Face community. [MiewID-msv3 discussion: msv2 versus msv3](https://huggingface.co/conservationxlabs/miewid-msv3/discussions/1), opened May 2025. Third-party discussion, not publisher confirmation of training membership.
[^9]: Lukáš Picek et al. [CzechLynx preprint record](https://arxiv.org/abs/2506.04931), first submitted 5 June 2025; [revised paper](https://arxiv.org/html/2506.04931v2), source-collection table and earlier-study discussion; [Scientific Data article](https://www.nature.com/articles/s41597-026-06853-9), 24 February 2026. These are versions of the same work, not independent corroborating datasets.
[^10]: Lukáš Picek, Lukáš Neumann and Jiří Matas. [Animal Identification with Independent Foreground and Background Modeling](https://arxiv.org/html/2408.12930v1), 23 August 2024, Section 4 and access footnote. Earlier Czech lynx collection and access conditions.
[^11]: LIFE Lynx. [C5 annual report](https://www.lifelynx.eu/wp-content/uploads/2024/04/LIFE-Lynx-C5-annual-report_FINAL.pdf), April 2024 hosting path, p. 10. Whiskerbook upload counts and monitoring years verified from the indexed passage; the complete large PDF was not available for review. No training-membership claim is inferred.
[^12]: Silvia Blašković, Magda Sindičić, Maureen Reilly, Ira Topličanec and Tomislav Gomerčić. [Spotted correctly? Human expertise remains essential in AI-assisted identification of Eurasian lynx](https://zslpublications.onlinelibrary.wiley.com/doi/10.1111/jzo.70095), first published 22 December 2025; Journal of Zoology 328(3), 223–233 (2026). Publisher abstract and [public photo resource](http://lynx.vef.hr/public/); full methods were not available for review.

