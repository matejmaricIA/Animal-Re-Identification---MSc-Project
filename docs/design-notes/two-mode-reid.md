# Design note — Two operating modes for open-set re-ID

**Status:** proposed (backlog) · **Date:** 2026-06-20 · **Owner:** Rino

This note records an empirical finding from validating the `reid_demo` pipeline on two
contrasting datasets, and the product decision it implies. It is the honest framing for the
park pitch and a constraint on what we promise.

## The finding

The pipeline was run end-to-end on two datasets that fail in **opposite** ways. The decisive
variable is **base feature strength** — measured as leave-one-out 1-NN identity accuracy on
the MegaDescriptor embeddings.

| Dataset | base 1-NN | Best clustering (ARI) | GV effect |
|---|---|---|---|
| **CowDataset** (12 ids, 1388 imgs) | **98.7%** (strong) | DBSCAN 0.15 → **leiden 0.70** | negligible (+0.01) |
| **ELPephants** (30-id subset, 504 imgs) | **14–25%** (weak) | **~0.01** (any backend/affinity) | rescues *retrieval* 25→48% rank-1, **not** clustering |

Two independent lessons:

1. **When features are strong, the clustering *algorithm* is the lever.** DBSCAN / threshold
   agglomerative cluster by an absolute distance cut, which fails when per-individual
   embedding clouds are diffuse and overlapping (the normal metric-learning / unseen-class
   regime) even though nearest-neighbour rank is near-perfect. Rank/graph methods
   (**leiden**, **finch**) that cluster the kNN graph ~4.6×'d ARI on cows (0.15 → 0.70),
   with no `eps` and no `k`. *(Implemented in `reid_demo/cluster.py`, T05.)*

2. **When features are weak, no clustering method works — and geometric verification does
   not save it.** On ELPephants, GV (LightGlue + RANSAC) nearly doubled **retrieval** rank-1
   (25 → 48%, matching the thesis's 13.7 → 51% lever) but left **clustering** at ARI ≈ 0.

### Why GV helps retrieval but not clustering

Retrieval rank-1 asks *"is the single best match correct?"* — GV reranks a shortlist and
fixes the top pick. Clustering needs the **entire affinity graph to be coherent** (high
precision across *all* pairs). With ~48% rank-1 (and far worse below rank-1), the graph is
still dominated by wrong edges; GV sharpens the handful of pairs it verifies but cannot
overcome the sea of false high-similarity edges from other individuals. **A method can double
retrieval and not move clustering**, because the two tasks demand different things.

## The decision: ship two modes

| Mode | When | What the biologist sees | How count is produced |
|---|---|---|---|
| **A — Auto-catalogue** | base features strong (good photos, favourable species) | clusters → visual catalogue + a count | clustering (leiden) + light review |
| **B — Assisted match** | base features weak (hard species, poor angles) | per-animal **top-N GV-verified candidate matches** | **human-confirmed merges** (T08 review), not autonomous clustering |

Mode A is the existing `--signals … + leiden` path. Mode B reuses the same embeddings + GV
machinery but as a **retrieval-into-review** workflow, not autonomous clustering: it surfaces
ranked candidates for a human to confirm. The individual **count** in Mode B comes from the
reviewer's confirmed merges.

### Picking the mode automatically

Estimate base feature strength cheaply (e.g. mean top-1 cosine margin, or a small labelled
calibration set's 1-NN accuracy) and route: high → Mode A, low → Mode B. Always expose the
estimate so we never silently present a Mode-A count that the features can't support.

## Pitch implications

- **Don't promise an automatic count for every species.** Promise: "filter your footage,
  and for the species where it's reliable, an automatic individual catalogue; for the rest,
  GV-assisted matching that makes a human reviewer many times faster."
- Lynx is a spotted cat (favourable, like leopard) and **unseen** by MegaDescriptor — so the
  honest expectation is "Mode A if photo quality is good, Mode B as the safety net," with the
  real number coming from a no-cost pilot on the park's footage.

## Follow-ups (backlog)

- Implement **Mode B** as a first-class path in T08 review (GV-reranked candidate queue).
- Add a **feature-strength estimator** + auto-routing to the T10 runner.
- Revisit clustering for weak features: k-reciprocal re-ranking, query-expansion, or
  fine-tuning the backbone on a small labelled set (the only thing that lifts base 1-NN).
- Quantify Mode A vs B on more datasets (ATRW, a real lynx set once labelled).
