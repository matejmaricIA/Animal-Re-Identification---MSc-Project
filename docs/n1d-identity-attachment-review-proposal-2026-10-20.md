# N1d proposal: review questions that attach proposals to confirmed animals

Date: 20 October 2026. Work item N1d in the [decision log](lynx-demo-decision-log.md) ("Active work (live)"). Follows [N1b](n1b-pipeline-integrity-proposal-2026-10-20.md) (commit `99e9a2da`), defect 3 "visible consequence", and the Risnjak engineering row "Review queue is narrow" ([assessment](research/project-risnjak-assessment-2026-10-14.md)).

**Status: approved as written on 20 Oct 2026 and implemented. Focused tests passed (user-run, 21 Oct). The full suite passed on 21 Oct, run by the assistant at the user's request: 787 passed, 1 skipped. Committed at the user's request on 21 Oct (see git log; not pushed).**

Implementation notes:
- **Attachment marker:** `same` also writes `extra_json['identity_attached']`, so `review_status_summary` can count attachments separately from pair merges.
- **Page:** the browser page shows both photos for identity items (`build_pair_image`).
- **Tests:** 8 new tests as listed. No existing test edited.

Scope: the `reid_demo` field-pipeline review tool (`reid_demo/review.py`, T08). The sibling CzechLynx review apps (input audit, N3 timed matching) are not touched.

## Problem

After N1b, a reviewed row's `cluster_id` is a confirmed animal ("identity"), and an unreviewed row's is a temporary proposal that never reuses an identity's number. When clustering reruns (new photos, a new model), unreviewed photos that used to share a cluster with reviewed photos become a separate proposal. The count then shows the same animal twice until someone attaches the proposal to the identity.

The review queue cannot ask that question (`build_review_queue`, `review.py:231`):

- **Pair items** compare a low-confidence member (`cluster_conf < 0.6`) with the best member of **its own** cluster. A confident proposal cluster gets no item at all, so a nine-photo proposal of a known animal is never asked about.
- **Singleton items** ask "is this a new individual?". The only way to attach a singleton to an existing animal is `same` + `target_cluster_id`, and the browser page has no control for that: it sends only `item_id`, `answer` and `note`.
- Nothing compares a proposal with confirmed animals, so the reviewer has no way to see the candidate.

Without this question, every recluster after a review adds proposals that can only be fixed by hand-written decision files. Counts drift upward each season.

## Proposed behaviour

A new review item kind, `identity`: "Is this proposal the same animal as confirmed animal #N?"

### Definitions

- **Identity:** a `cluster_id >= 0` held by at least one row with `review_status` in {`confirmed`, `merged`, `split`}. Rejected rows do not define identities.
- **Proposal:** either a `cluster_id >= 0` all of whose rows are unreviewed, or an unreviewed singleton (`is_candidate_new = 1` or `cluster_id = -1`).

### Candidate choice (no tuned threshold)

1. Load the normalised embeddings once with `embed.get_embedding_matrix(normalize=True)`. This is a lazy import; T08 gains a read-only dependency on T04.
2. For each proposal P and identity I, use single-linkage similarity: the highest cosine between any member of P and any **reviewed** member of I. Pairs with a known `left` on one side and a known `right` on the other are excluded (D4; `other`/`unknown` is compatible with anything).
3. The candidate is the identity with the highest similarity. Identities this proposal was already told "different" are skipped (rule below). The anchor shown is the identity's member closest to P's representative; P's representative is its highest-confidence member.
4. With no embeddings, no identities or no compatible candidate, nothing changes: singletons get today's "new individual?" item, and confident clusters get nothing.

There is no similarity floor by default. The CzechLynx calibration showed that fixed cut-offs move with the model, so a number chosen here would be arbitrary. An optional `identity_min_similarity` parameter, default `None`, is available if the user later wants one.

### Queue order

Identity items come **first**, ordered by proposal size (largest first, because one answer moves the most photos), then by similarity. Pair items and singleton items follow as today, and the whole queue is still capped at `queue_size` (30). A singleton that receives an identity item does not also get a singleton item: the identity item offers "New individual" as an answer.

### Answers (`apply_decisions`)

| Answer | Effect | Status written |
| --- | --- | --- |
| `same` | Every member of P (or the singleton) moves to identity N via the existing `_merge_clusters(keep=N, drop=P)`. The identity's number is kept by construction, not by the smaller-ID rule | `merged` |
| `different` | No cluster change. Each member of P gets `{identity: N, at, note}` appended to `extra_json['identity_rejections']`. The next queue offers the next-best identity, or "new individual?" | unchanged (`unreviewed`) |
| `new` | P is confirmed as a new animal. A cluster proposal keeps its current label, which is already above every identity (N1b). A singleton gets a fresh `max+1`, as today | `confirmed` |
| `skip` | nothing | — |

**Rejection rule:** identity N is skipped for proposal P when at least half of P's members carry a rejection for N. Rejections live on the records, so they survive reclustering, which may regroup the members.

Every mutation goes through `store.update_review` / `update_extra` and is recorded in the session JSON, like existing decisions. Replaying a session is idempotent: a merge already applied is a no-op, and a rejection already present is not appended twice.

### Data shape

`ReviewItem` gains optional fields, so old session JSON still loads through `ReviewItem(**d)`:

```python
    proposal_size: Optional[int] = None     # photos in the proposal
    identity_size: Optional[int] = None     # reviewed photos of the candidate identity
    similarity: Optional[float] = None      # single-linkage cosine (ordering only)
```

For `kind="identity"`: `record_id_a` is P's representative, `record_id_b` the identity anchor, `cluster_id_a` P's label (or -1), and `cluster_id_b` = N. Item IDs are `identity__<rep>__<N>`.

### Browser page

- The same page and buttons (Same animal / Different animal / New individual / Not sure).
- For identity items the heading reads "Is this the same animal as confirmed animal #N?", and the reason line gives "proposal of 9 photos · animal #N has 5 reviewed photos".
- The decision posts only `item_id`, `answer` and `note`, as today; the target is carried by the item.
- `review_status_summary` adds `identity_attachments` and `identity_rejections` counts.

## Planned changes

| File | Change |
| --- | --- |
| `reid_demo/review.py` | identity/proposal discovery; candidate search; `identity` items in `build_review_queue` (opt-out `include_identity_items=True`); `_apply_identity`; rejection log; three `ReviewItem` fields; page heading/reason; two summary counts; docstring (T04 read dependency) |
| `reid_demo/DATA_CONTRACT.md` | `identity_rejections` key; identity versus proposal definitions; T08 attachment rule |
| `tests/test_review.py` | new tests below |

Not touched: `cluster.py`, `store.py`, `run_demo.py`, `catalogue.py`, benchmark modules, the sibling apps.

## Tests

**Edits to existing tests:** none expected. The `seeded` fixture has no embeddings, so no identity items are generated and every existing queue assertion holds. `test_no_fisher_fusion_import` still passes: the new import is `reid_demo.embed`, lazily.

**New tests** (`tests/test_review.py`, with a small fixture that has embeddings and one confirmed identity):

| Test | Asserts |
| --- | --- |
| `test_confident_proposal_gets_identity_item` *repro* | A high-confidence proposal of a reviewed animal gets one `identity` item naming that animal; today the queue is empty for it |
| `test_identity_item_respects_flanks` | A known left proposal is never offered a right-only identity |
| `test_identity_same_merges_whole_proposal` | All members move to N with status `merged`; N unchanged; session JSON records each mutation |
| `test_identity_different_logs_and_offers_next` | A rejection is logged on the members; the next queue offers the second-best identity; with none left, the singleton gets "new?" |
| `test_identity_new_confirms_proposal` | Cluster proposal keeps its label with status `confirmed`; singleton gets `max+1` |
| `test_identity_items_first_and_capped` | Identity items precede pair and singleton items, largest proposal first; `queue_size` still applies |
| `test_no_embeddings_no_identity_items` | Without embeddings the queue equals today's |
| `test_identity_session_replay_idempotent` | Replaying the session JSON changes nothing |

## Not in scope

- **Showing more than two photos.** The item shows P's representative next to the identity anchor. A strip of several members per side would help hard cases; a later UI item.
- **Using the fused or GV affinity** for candidate choice instead of global cosine. The existing `pair_scores` ordering hint is unchanged.
- **Error measurement.** How often reviewers mis-attach is what the N3 timed trial measures for the CzechLynx workflow; nothing here estimates it.

## Validation

After implementation I run `py_compile` and Ruff F. Tests are user-run unless you authorize me:

```bash
venv/bin/python -m pytest tests/test_review.py tests/test_cluster.py -q
```

```bash
venv/bin/python -m pytest tests -q
```

## Risks

- **A wrong `same` merges a whole proposal.** That is the existing semantics of cross-cluster pair merges; the session JSON keeps every old→new mutation, so a merge can be reversed by hand.
- **Candidate quality depends on the embedding.** With weak same-side discrimination (lynx top-1 ≈ 68–71% on calibration), the top candidate will often be wrong, and the reviewer will answer "different" more than "same". The question is still cheaper than finding the match by hand.
- **Size:** about 180 lines in `review.py` and 220 lines of tests.
