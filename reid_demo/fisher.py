"""reid_demo.fisher — local-feature + Fisher-vector service (T11).

The Fisher-vector sibling of T04's global-embedding service. Where T04
(`reid_demo/embed.py`) wraps `global_embedding.py` to produce one cached GLOBAL
vector per crop keyed by ``record_id``, this module wraps the repo's existing,
proven local-feature pipeline — ``feature_extraction.py`` (DISK / ALIKED /
SuperPoint local descriptors) + ``feature_aggregation.py`` (PCA + GMM -> Fisher
vector) — to produce one cached FISHER vector per crop keyed by ``record_id``.

T11 is a THIN wrapper: it does NOT reimplement descriptor extraction, PCA/GMM
fitting, or the Fisher-vector math. It calls the existing functions and adds:
store integration (read crops, write refs via ``update_extra``), ``record_id``-keyed
descriptor + Fisher caching, optional species filtering, robust per-record skip,
and the read-side ``get_fisher_matrix`` helper that T12 consumes.

Pipeline position (produces a SECOND per-crop vector type alongside T04's global one):

    raw image --(T02)--> crop + record   [record_id -> crop_path]
              --(T04)--> global embedding (reid_demo/embed.py)
              --(T11 THIS MODULE)--> Fisher vector (reid_demo/fisher.py)
              --(T12)--> fused affinity (global+Fisher) + GV rerank

Boundary (D8, binding): T11 PRODUCES and CACHES a per-crop Fisher vector plus a
read-side matrix accessor. It does NOT fuse, calibrate, cluster, or run geometric
verification. It imports NONE of ``reid_demo.cluster`` (T05), ``reid_demo.fusion``
(T12), ``reid_demo.embed`` (T04), ``calibration``, or ``geometric_verification``.
Its only hard deps are T01 (store) and the records T02 writes.

Heavy deps (numpy, torch, lightglue, h5py, sklearn) are imported LAZILY inside the
functions that need them, so this module imports cleanly under plain ``python3``
even where those packages are absent. numpy is used in type hints only via strings.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from reid_demo import store
from reid_demo.store import (
    DEFAULT_DB_PATH,
    connect,
    query_records,
    update_extra,
)

# constants.py lives at the repo root; import the binding default PCA dim from it
# (do NOT hard-code 128 in logic — derive everything from the fitted models).
try:
    from constants import N_COMPONENTS_PCA as _N_COMPONENTS_PCA
except Exception:  # pragma: no cover - constants.py is always present in the repo
    _N_COMPONENTS_PCA = 128

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream tickets / T12 import these)
# --------------------------------------------------------------------------- #

DEFAULT_FISHER_DIR: str = "data/reid_demo/fisher"
DEFAULT_METHOD: str = "disk"               # DISK is the binding default local-feature method (D8)
DEFAULT_PCA_DIM: int = _N_COMPONENTS_PCA   # =128 from constants.py; never hard-coded in logic


# --------------------------------------------------------------------------- #
# Result dataclass (mirrors T04's EmbedResult)
# --------------------------------------------------------------------------- #

@dataclass
class FisherResult:
    """Outcome of a ``build_fisher_records`` run (mirror of T04's EmbedResult)."""

    dataset: Optional[str]
    cache_path: str               # the .pkl holding the {record_id -> FV} dict
    method: str                   # local-feature method, e.g. "disk"
    fisher_label: str             # fisher_cache_label(method, pca_dim)
    pca_dim: int                  # PCA components actually used
    gmm_components: Optional[int] # GMM components (from the fitted/loaded GMM); None if nothing computed
    fisher_dim: Optional[int]     # observed vector length = 2*gmm_components*pca_dim (None if nothing computed)
    n_total: int                  # candidate records considered
    n_fishered: int               # records assigned a Fisher vector this run
    n_skipped: int                # already had fisher_ref (only_missing)
    n_failed: int                 # crop missing/unreadable (excluded before extraction)
    n_zero_vectors: int           # crops that produced an all-zero FV (no local descriptors)
    failed_ids: list = field(default_factory=list)   # record_ids excluded for missing/unreadable crops


# --------------------------------------------------------------------------- #
# Cache-path helpers (mirror T04's *_cache_label / *_cache_path)
# --------------------------------------------------------------------------- #

def fisher_cache_label(method: str = DEFAULT_METHOD, pca_dim: int = DEFAULT_PCA_DIM) -> str:
    """Deterministic label distinguishing caches by method + pca_dim, e.g. ``'disk_pca128'``.

    Used in the Fisher pickle filename so different methods/dims get distinct caches.
    Mirrors the role of ``global_embedding.global_embedding_cache_label`` in T04.
    """
    return f"{str(method).lower()}_pca{int(pca_dim)}"


def fisher_cache_path(dataset: Optional[str], method: str = DEFAULT_METHOD,
                      pca_dim: int = DEFAULT_PCA_DIM,
                      cache_dir: str = DEFAULT_FISHER_DIR) -> str:
    """Deterministic path to the ``{record_id -> FV}`` pickle.

    ``f'{cache_dir}/{dataset or "all"}_{fisher_cache_label(method, pca_dim)}.pkl'``.
    Creates no files; just returns the path (parent dir is created by writers).
    Mirrors T04's ``embedding_cache_path``.
    """
    label = fisher_cache_label(method, pca_dim)
    ds = dataset if dataset else "all"
    return os.path.join(cache_dir, f"{ds}_{label}.pkl")


def _descriptor_dir(dataset: Optional[str], method: str,
                    cache_dir: str = DEFAULT_FISHER_DIR) -> str:
    """Per-(dataset, method) local-descriptor HDF5 cache dir, under DEFAULT_FISHER_DIR.

    Holds ``descriptors.h5`` + ``keypoints.h5`` keyed by ``record_id``. Recommended layout
    from the ticket: ``data/reid_demo/fisher/{dataset}_{method}_descriptors/``.
    """
    ds = dataset if dataset else "all"
    return os.path.join(cache_dir, f"{ds}_{str(method).lower()}_descriptors")


def _fitted_model_ds_tag(dataset: Optional[str]) -> str:
    """ds_tag passed to load_or_train_fisher_vectors (PCA/GMM/FV pickle namespace)."""
    return dataset if dataset else "all"


def _fitted_cache_suffix(pca_dim: int) -> str:
    """cache_suffix for load_or_train_fisher_vectors; keeps T11 caches distinct from
    the legacy closed-set caches (PCA_PATH/GMM_PATH/FISHER_VECTORS templates)."""
    return f"reiddemo_pca{int(pca_dim)}"


def _desc_covers(desc_h5: str, wanted: "set") -> bool:
    """True if the descriptor HDF5 at ``desc_h5`` contains every record_id in ``wanted``."""
    if not os.path.exists(desc_h5):
        return False
    try:
        import h5py  # lazy
    except Exception:  # pragma: no cover - h5py is in the venv
        return False
    try:
        with h5py.File(desc_h5, "r") as f:
            keys = set(f.keys())
    except Exception:
        return False
    return wanted.issubset(keys)


def _invalidate_fitted_cache(ds_tag: str, method: str, pca_dim: int) -> None:
    """Delete the fitted PCA/GMM/FV pickle triplet so the next fit recomputes.

    Targets the ``constants.PCA_PATH/GMM_PATH/FISHER_VECTORS`` templates with this
    module's ``(ds_tag, method, cache_suffix)`` so only T11's own caches are touched.
    """
    try:
        from constants import PCA_PATH, GMM_PATH, FISHER_VECTORS  # lazy
    except Exception:  # pragma: no cover
        return
    suffix = _fitted_cache_suffix(pca_dim)
    for tmpl in (PCA_PATH, GMM_PATH, FISHER_VECTORS):
        path = tmpl.format(ds_tag, method, suffix)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Low-level: crop paths -> {record_id -> Fisher vector}  (mirror embed.embed_crops)
# --------------------------------------------------------------------------- #

def build_fisher_vectors(crop_paths: Dict[str, str], cache_path: str, *,
                         dataset: Optional[str] = None,
                         method: str = DEFAULT_METHOD,
                         pca_dim: int = DEFAULT_PCA_DIM,
                         desc_dir: Optional[str] = None,
                         device=None) -> "Dict[str, np.ndarray]":
    """Given ``{record_id -> crop_image_path}`` produce + persist Fisher vectors.

    Steps (each REUSES existing repo code; nothing is reimplemented here):

      1. Ensure local descriptors exist for these crops in an HDF5 cache keyed by
         ``record_id`` via ``feature_aggregation.ensure_local_descriptors``
         (``method='disk'`` -> ``extract_features``; ``'aliked'``/``'superpoint'`` ->
         ``extract_features_lightglue``). We pass ``[(record_id, path), ...]`` tuples so
         the HDF5 dataset keys ARE ``record_id`` (see ``_parse_image_item``).
      2. Fit-or-load PCA + GMM and compute per-crop Fisher vectors via
         ``feature_aggregation.load_or_train_fisher_vectors`` (which manages the fitted
         PCA/GMM/FV pickle cache under ``constants.PCA_PATH/GMM_PATH/FISHER_VECTORS``).
      3. Persist the ``{record_id -> FV}`` dict to ``cache_path`` (the file T12 reads)
         and return it.

    Keys whose ``crop_path`` does not exist on disk are pre-dropped (NOT in the result).
    Crops with zero descriptors yield an all-zero FV (KEPT in result; the caller counts
    them as zero-vectors). This function does NOT touch the store.

    NOTE (descriptor-cache additivity): ``ensure_local_descriptors`` is WHOLE-SET — it
    short-circuits if ``desc_dir`` already exists, else writes the HDF5 in overwrite mode.
    For the demo's batch-per-(dataset, method) runs this is the intended usage: the first
    run extracts + fits + computes; an ``only_missing`` re-run recomputes nothing because
    ``load_or_train_fisher_vectors`` returns the cached models + FV dict.
    """
    import numpy as np  # lazy
    from feature_aggregation import (  # lazy: pulls torch/h5py/sklearn
        ensure_local_descriptors,
        load_descriptors,
        load_or_train_fisher_vectors,
    )

    method = str(method).lower()
    pca_dim = int(pca_dim)

    if desc_dir is None:
        desc_dir = _descriptor_dir(dataset, method)

    # Pre-filter: drop any record whose crop file is missing/unreadable so the DISK /
    # lightglue extractor (cv2.imread / load_image) never raises mid-batch.
    items: List[Tuple[str, str]] = []
    for rid, path in sorted(crop_paths.items()):
        if path and os.path.exists(path):
            items.append((rid, path))

    fv_dict: "Dict[str, np.ndarray]" = {}

    if items:
        # (1) descriptor HDF5 cache keyed by record_id.
        # ensure_local_descriptors is WHOLE-SET: it short-circuits if desc_dir already
        # exists. That is correct for true re-runs of the same record set, but a stale
        # dir that does NOT cover the requested record_ids would otherwise silently yield
        # nothing. Detect that and rebuild (also invalidating the now-stale fitted
        # PCA/GMM/FV pickle so the transform recomputes against the new descriptor pool).
        Path(desc_dir).parent.mkdir(parents=True, exist_ok=True)
        desc_h5 = os.path.join(desc_dir, "descriptors.h5")
        wanted = {rid for rid, _ in items}
        if os.path.isdir(desc_dir) and not _desc_covers(desc_h5, wanted):
            import shutil
            shutil.rmtree(desc_dir, ignore_errors=True)
            _invalidate_fitted_cache(_fitted_model_ds_tag(dataset), method, pca_dim)
        ensure_local_descriptors(items, method, desc_dir)

        # (2) fit-or-load PCA+GMM and compute per-crop FVs. load_or_train_fisher_vectors
        # writes its PCA/GMM/FV pickles under ./data/{ds_tag}/... WITHOUT creating that
        # parent dir, so we create it here defensively.
        ds_tag = _fitted_model_ds_tag(dataset)
        Path(os.path.join("data", ds_tag)).mkdir(parents=True, exist_ok=True)
        _pca, _gmm, fv_dict = load_or_train_fisher_vectors(
            ds_tag=ds_tag,
            method_name=method,
            cache_suffix=_fitted_cache_suffix(pca_dim),
            pca_dim=pca_dim,
            descriptors_loader=lambda: load_descriptors(desc_h5),
        )
        # Keep only the records we were asked about (the FV dict mirrors the descriptor
        # HDF5 keys, which == the records that had crops on disk).
        fv_dict = {k: np.asarray(v, dtype=np.float32) for k, v in fv_dict.items() if k in wanted}

    # (3) persist the {record_id -> FV} dict to the contracted cache path.
    parent = Path(cache_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(fv_dict, fh)

    return fv_dict


# --------------------------------------------------------------------------- #
# Main entry: store -> Fisher vectors + refs  (mirror embed.embed_records)
# --------------------------------------------------------------------------- #

def _parse_extra(extra_json: Optional[str]) -> dict:
    """Best-effort parse of a record's extra_json blob into a dict (empty on failure)."""
    if not extra_json:
        return {}
    try:
        blob = json.loads(extra_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return blob if isinstance(blob, dict) else {}


def build_fisher_records(conn, *,
                         dataset: Optional[str] = None,
                         method: str = DEFAULT_METHOD,
                         pca_dim: int = DEFAULT_PCA_DIM,
                         cache_dir: str = DEFAULT_FISHER_DIR,
                         only_missing: bool = True,
                         only_species: Optional[str] = None,
                         limit: Optional[int] = None,
                         device=None) -> FisherResult:
    """Main entry (mirror of ``embed.embed_records``).

    1. Query candidate records (``query_records(conn, dataset=...)``, optionally
       species-filtered). With ``only_missing=True`` a record is a SKIP iff its
       ``extra_json`` already holds a non-null ``fisher_ref`` AND that ``record_id`` is a
       key in the on-disk FV pickle.
    2. Build ``{record_id -> crop_path}``, dropping missing-crop records into
       ``failed_ids`` / ``n_failed``.
    3. Call ``build_fisher_vectors`` into ``fisher_cache_path(...)``.
    4. For each record now present in the cache, write ``extra_json`` keys
       ``fisher_ref = record_id``, ``fisher_path = <pickle>`` (and ``fisher_label``) via
       ``update_extra``.
    5. Return a ``FisherResult``.

    ``only_species`` matches the store ``species`` column by exact lowercase equality
    (same convention as T04).
    """
    import numpy as np  # lazy

    method = str(method).lower()
    pca_dim = int(pca_dim)
    label = fisher_cache_label(method, pca_dim)
    cache_path = fisher_cache_path(dataset, method, pca_dim, cache_dir)

    # ----- (1) candidate records (optional species filter, exact lowercase equality) ---
    records = query_records(conn, dataset=dataset, order_by="record_id")
    if only_species is not None:
        target = only_species.strip().lower()
        records = [r for r in records if (r.species or "").strip().lower() == target]
    if limit is not None:
        records = records[:limit]

    n_total = len(records)

    # Existing on-disk FV cache (for only_missing skip detection).
    existing_cache: Dict[str, object] = {}
    if os.path.exists(cache_path):
        try:
            existing_cache = load_fisher_vectors(cache_path)
        except Exception:
            existing_cache = {}

    to_process: List = []
    n_skipped = 0
    for r in records:
        if only_missing:
            ex = _parse_extra(r.extra_json)
            if ex.get("fisher_ref") and (r.record_id in existing_cache):
                n_skipped += 1
                continue
        to_process.append(r)

    # ----- (2) {record_id -> crop_path}; drop missing-crop records --------------------
    crop_paths: Dict[str, str] = {}
    failed_ids: List[str] = []
    for r in to_process:
        if r.crop_path and os.path.exists(r.crop_path):
            crop_paths[r.record_id] = r.crop_path
        else:
            failed_ids.append(r.record_id)
    n_failed = len(failed_ids)

    # ----- (3) compute Fisher vectors -------------------------------------------------
    gmm_components: Optional[int] = None
    fisher_dim: Optional[int] = None
    n_fishered = 0
    n_zero_vectors = 0

    if crop_paths:
        fv_dict = build_fisher_vectors(
            crop_paths, cache_path,
            dataset=dataset, method=method, pca_dim=pca_dim, device=device,
        )
        # merge: a previous run may have computed records that aren't in this batch.
        # build_fisher_vectors persists only THIS batch's FVs, so re-merge the prior
        # cache so the contracted pickle stays additive across runs.
        if existing_cache:
            merged = dict(existing_cache)
            merged.update(fv_dict)
            if len(merged) != len(fv_dict):
                with open(cache_path, "wb") as fh:
                    pickle.dump(merged, fh)
            fv_dict = merged
    else:
        # Nothing new to compute; fall back to the existing cache for ref writing.
        fv_dict = existing_cache

    # Derive dim / gmm components from the produced vectors (NEVER hard-coded).
    if fv_dict:
        sample = next(iter(fv_dict.values()))
        fisher_dim = int(np.asarray(sample).shape[0])
        gmm_components = fisher_dim // (2 * pca_dim) if pca_dim else None

    # ----- (4) write store refs into extra_json (no new column) -----------------------
    for r in to_process:
        rid = r.record_id
        if rid in crop_paths and rid in fv_dict:
            vec = np.asarray(fv_dict[rid])
            update_extra(conn, rid, "fisher_ref", rid)
            update_extra(conn, rid, "fisher_path", cache_path)
            update_extra(conn, rid, "fisher_label", label)
            n_fishered += 1
            if float(np.linalg.norm(vec)) < 1e-9:
                n_zero_vectors += 1

    return FisherResult(
        dataset=dataset,
        cache_path=cache_path,
        method=method,
        fisher_label=label,
        pca_dim=pca_dim,
        gmm_components=gmm_components,
        fisher_dim=fisher_dim,
        n_total=n_total,
        n_fishered=n_fishered,
        n_skipped=n_skipped,
        n_failed=n_failed,
        n_zero_vectors=n_zero_vectors,
        failed_ids=failed_ids,
    )


# --------------------------------------------------------------------------- #
# Read side (for T12) — mirror T04's load_embeddings / get_embedding_matrix
# --------------------------------------------------------------------------- #

def load_fisher_vectors(cache_path: str) -> "Dict[str, np.ndarray]":
    """Load a Fisher cache pickle (``{record_id -> np.ndarray}``).

    Thin wrapper over pickle for downstream symmetry (mirror of T04's ``load_embeddings``).
    Raises ``FileNotFoundError`` with a clear message if absent.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"fisher cache not found: {cache_path}")
    with open(cache_path, "rb") as fh:
        return pickle.load(fh)


def get_fisher_matrix(conn, *,
                      dataset: Optional[str] = None,
                      normalize: bool = True,
                      only_clustered: bool = False) -> "Tuple[np.ndarray, list]":
    """READ side for T12 (mirror of T04's ``get_embedding_matrix``).

    Query records whose ``extra_json`` has a non-null ``fisher_ref``, load each vector
    from its ``fisher_path`` (key = ``fisher_ref``), stack into an ``(N, D)`` float32
    matrix in stable sorted-``record_id`` order, and return ``(matrix, record_ids)``.

    Fisher vectors are already L2-normalized on disk; ``normalize=True`` re-normalizes
    defensively with an epsilon guard (zero vectors stay exactly zero, never NaN). Each
    referenced pickle is loaded ONCE (grouped by ``fisher_path``). Raises a clear
    ``RuntimeError`` naming the ``record_id`` + path if ``fisher_ref`` is set but the key
    is missing from the pickle (stale cache), instead of a bare ``KeyError``.

    ``only_clustered`` is a convenience pass-through (cluster_id NOT NULL), default off.
    """
    import numpy as np  # lazy

    records = query_records(conn, dataset=dataset, order_by="record_id")

    # Collect (record_id, fisher_path, fisher_ref) for records that carry a fisher_ref.
    selected: List[Tuple[str, str, str]] = []
    for r in records:
        if only_clustered and r.cluster_id is None:
            continue
        ex = _parse_extra(r.extra_json)
        ref = ex.get("fisher_ref")
        if not ref:
            continue
        path = ex.get("fisher_path")
        if not path:
            raise RuntimeError(
                f"record {r.record_id!r} has fisher_ref but no fisher_path in extra_json"
            )
        selected.append((r.record_id, path, str(ref)))

    selected.sort(key=lambda t: t[0])  # stable sorted-record_id order
    record_ids = [rid for rid, _, _ in selected]

    if not selected:
        return np.empty((0, 0), dtype=np.float32), record_ids

    # Group by fisher_path so each pickle loads exactly once.
    caches: Dict[str, "Dict[str, np.ndarray]"] = {}
    rows: List["np.ndarray"] = []
    for rid, path, ref in selected:
        if path not in caches:
            caches[path] = load_fisher_vectors(path)
        cache = caches[path]
        if ref not in cache:
            raise RuntimeError(
                f"stale fisher cache: record {rid!r} ref'd {ref!r} but missing from {path!r}"
            )
        rows.append(np.asarray(cache[ref], dtype=np.float32).ravel())

    matrix = np.vstack(rows).astype(np.float32, copy=False)

    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # epsilon guard: zero rows stay zero, not NaN
        matrix = matrix / norms

    return matrix, record_ids


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid_demo.fisher",
        description="T11 local-feature + Fisher-vector service for the lynx re-ID demo.",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help=f"store DB path (default {DEFAULT_DB_PATH}).")
    parser.add_argument("--dataset", default=None,
                        help="dataset selector to scope which records to process.")
    parser.add_argument("--method", default=DEFAULT_METHOD,
                        help=f"local-feature method (default {DEFAULT_METHOD}); "
                             "alternates: aliked, superpoint.")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                        help=f"PCA components (default {DEFAULT_PCA_DIM}).")
    parser.add_argument("--cache-dir", default=DEFAULT_FISHER_DIR,
                        help=f"Fisher cache dir (default {DEFAULT_FISHER_DIR}).")
    parser.add_argument("--only-species", default=None,
                        help="only Fisher-vectorize records whose species matches "
                             "(exact, case-insensitive).")
    parser.add_argument("--all", action="store_true",
                        help="recompute store-side refs for ALL matching records "
                             "(only_missing=False). PCA/GMM/FV pickle is still reused "
                             "if present.")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of records processed.")
    parser.add_argument("--device", default=None,
                        help="torch device hint (cuda|cpu); extractors default to "
                             "cuda-if-available. CPU works (slow).")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        conn = connect(args.db, create=False)
    except Exception as exc:  # pragma: no cover - bad DB path
        print(f"[fisher] FATAL: cannot open store {args.db!r}: {exc}", file=sys.stderr)
        return 1

    try:
        res = build_fisher_records(
            conn,
            dataset=args.dataset,
            method=args.method,
            pca_dim=args.pca_dim,
            cache_dir=args.cache_dir,
            only_missing=not args.all,
            only_species=args.only_species,
            limit=args.limit,
            device=args.device,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[fisher] FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(
        f"T11 fisher: dataset={res.dataset} method={res.method} pca={res.pca_dim} "
        f"dim={res.fisher_dim} total={res.n_total} fishered={res.n_fishered} "
        f"skipped={res.n_skipped} failed={res.n_failed} zero={res.n_zero_vectors} "
        f"cache={res.cache_path}"
    )

    # "Nothing to do" is worth surfacing: 0 found to process AND none already ref'd.
    if res.n_fishered == 0 and res.n_skipped == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
