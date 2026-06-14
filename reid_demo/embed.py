"""reid_demo.embed — global embedding service (T04).

Thin, clean wrapper around the existing, proven ``global_embedding`` code. Turns the
crop images that T02 wrote to disk (optionally species-filtered by T03) into cached
**MegaDescriptor-L-384 global embeddings** — one vector per crop, keyed by
``record_id`` — and writes a *reference* to each vector back into the T01 store
(``embedding_ref``, ``embedding_path``) so T05 can stack the vectors and discover
individuals.

THE EMBEDDING CONTRACT (binding — D2; T05/T07 must conform to this, not the reverse):
  * Embeddings are stored at their MODEL-NATIVE dimension: 1536 for the base model
    ``megadescriptor-l-384``, 384 ONLY for a ``linear_l2`` checkpoint. The dimension is
    NEVER hard-coded; read it from ``EmbedResult.embedding_dim`` / ``matrix.shape[1]``.
  * Stored vectors are RAW / NOT L2-normalized — exactly what the model returns.
  * Every consumer obtains its matrix via ``get_embedding_matrix(conn, ...,
    normalize=True)`` which L2-normalizes rows at read time so cosine == dot product
    regardless of which model/checkpoint produced the vectors.

Reuse over reinvention: the single hard dependency is
``global_embedding.extract_global_embeddings`` (loads the model once, preprocesses to
384x384 + ImageNet norm, runs inference, returns ``{key: 1-D float32 ndarray}``). We do
NOT re-open the model or re-preprocess. Our value-add is: store integration, partial /
additive ``record_id``-keyed caching, species filtering, robust per-record skip, and the
read-side matrix helper.

Heavy deps (torch / timm / megadescriptor) are imported LAZILY (inside the functions
that need them), mirroring ``reid_demo.ingest``, so this module imports cleanly under
plain ``python3`` without torch present. ``numpy`` is used only for typing/light array
work at module import; if it is unavailable we still want the import to succeed for
contract introspection, but in practice the venv always has numpy.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reid_demo import store
from reid_demo.store import (
    connect,
    query_records,
    update_embedding,
)

# --------------------------------------------------------------------------- #
# Module-level constants (exact names — downstream tickets import these)
# --------------------------------------------------------------------------- #

DEFAULT_EMB_DIR: str = "data/reid_demo/embeddings"
DEFAULT_MODEL_NAME: str = "megadescriptor-l-384"


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class EmbedResult:
    dataset: Optional[str]
    cache_path: str               # the .pkl that holds the vectors
    model_name: str
    model_label: str              # global_embedding_cache_label(model_name, checkpoint_path)
    embedding_dim: Optional[int]  # observed vector length, e.g. 1536 (None if nothing embedded)
    n_total: int                  # candidate records considered
    n_embedded: int               # newly computed this run
    n_skipped: int                # already had embedding (only_missing)
    n_failed: int                 # crop missing/unreadable
    failed_ids: list = field(default_factory=list)   # record_ids that failed


# --------------------------------------------------------------------------- #
# Cache path
# --------------------------------------------------------------------------- #

def _model_label(model_name: str, checkpoint_path: Optional[str]) -> str:
    """Resolve the cache label via the existing global_embedding helper (lazy import
    keeps torch out of plain `python3` imports — global_embedding pulls in torch)."""
    from global_embedding import global_embedding_cache_label
    return global_embedding_cache_label(model_name, checkpoint_path)


def embedding_cache_path(
    dataset: Optional[str],
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_path: Optional[str] = None,
    cache_dir: str = DEFAULT_EMB_DIR,
) -> str:
    """Deterministic cache path: ``f'{cache_dir}/{dataset or "all"}_{model_label}.pkl'``
    where ``model_label = global_embedding.global_embedding_cache_label(model_name,
    checkpoint_path)``. Creates no files; just returns the path (parent dir is created by
    the writers)."""
    label = _model_label(model_name, checkpoint_path)
    name = f"{dataset or 'all'}_{label}.pkl"
    return os.path.join(cache_dir, name)


# --------------------------------------------------------------------------- #
# Low-level: embed a {key -> crop_path} dict with additive caching
# --------------------------------------------------------------------------- #

def _load_cache(cache_path: str) -> Dict[str, np.ndarray]:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"embedding cache {cache_path!r} is not a Dict[str, np.ndarray]"
            )
        return data
    return {}


def _save_cache(cache_path: str, cache: Dict[str, np.ndarray]) -> None:
    parent = Path(cache_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)


def embed_crops(
    crop_paths: Dict[str, str],
    cache_path: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_path: Optional[str] = None,
    device=None,
) -> Dict[str, np.ndarray]:
    """Low-level: given ``{key -> crop_image_path}``, return ``{key -> np.ndarray}`` and
    persist to ``cache_path`` as a ``Dict[str, np.ndarray]`` pickle.

    If ``cache_path`` exists, load it and compute only MISSING keys, then merge + re-save
    (so re-runs are cheap and additive). Keys whose crop file does not exist on disk are
    omitted from the result (not fatal). Internally calls
    ``global_embedding.extract_global_embeddings`` ONCE on the subset that needs
    computing (load model once, infer many). ``key`` is caller-defined; ``embed_records``
    passes ``record_id``. Does NOT touch the store.

    Returns the merged dict restricted to the requested keys that have a vector (i.e. the
    full set of keys in ``crop_paths`` that succeeded, including ones already cached).
    """
    cache = _load_cache(cache_path)

    # Pre-filter on existence so extract_global_embeddings never trips over a bad path;
    # missing files are silently dropped (the caller — embed_records — accounts for them
    # as failures by checking which requested keys ended up with a vector).
    to_compute: Dict[str, str] = {}
    for key, path in crop_paths.items():
        if key in cache:
            continue
        if path and os.path.exists(path):
            to_compute[key] = path
        # missing file -> skip (omitted from result)

    if to_compute:
        device = _coerce_device(device)
        from global_embedding import extract_global_embeddings
        computed = extract_global_embeddings(
            to_compute,
            model_name=model_name,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        # Store RAW, as float32 1-D arrays (exactly as the model emits — D2).
        for key, vec in computed.items():
            cache[key] = np.asarray(vec, dtype=np.float32).reshape(-1)
        _save_cache(cache_path, cache)

    # Return only the requested keys that now have a vector.
    return {k: cache[k] for k in crop_paths if k in cache}


def _coerce_device(device):
    """Accept None | str | torch.device; convert a string to torch.device. None is passed
    through so extract_global_embeddings picks cuda-if-available."""
    if device is None or not isinstance(device, str):
        return device
    import torch
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Main entry: embed records from the store
# --------------------------------------------------------------------------- #

def embed_records(
    conn,
    *,
    dataset: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_path: Optional[str] = None,
    cache_dir: str = DEFAULT_EMB_DIR,
    only_missing: bool = True,
    only_species: Optional[str] = None,
    limit: Optional[int] = None,
    device=None,
) -> EmbedResult:
    """Main entry. Pulls candidate detection records from the T01 store, embeds their
    crops, caches the vectors keyed by ``record_id``, and writes ``embedding_ref`` /
    ``embedding_path`` back into the store.

    Steps:
      1. ``query_records(conn, dataset=dataset, [species filter], has_embedding=False if
         only_missing)`` to get candidate records.
      2. Build ``{record_id -> crop_path}``.
      3. ``embed_crops`` into ``embedding_cache_path(...)`` (additive cache).
      4. For each record that now has a vector in the cache, call
         ``update_embedding(conn, record_id, embedding_ref=record_id,
         embedding_path=cache_path)``.
      5. Return ``EmbedResult``.

    ``only_species`` matches the store ``species`` column by **exact lowercase equality**
    (whitespace-stripped). Records with no crop on disk (or unreadable) go to
    ``failed_ids`` and are not updated.
    """
    label = _model_label(model_name, checkpoint_path)
    cache_path = embedding_cache_path(dataset, model_name, checkpoint_path, cache_dir)

    # Always query the full matching set (for stable n_total / skip accounting). When
    # only_missing=True, records that already carry an embedding_ref are SKIPPED (not
    # recomputed and not re-written); when False, all are (re)embedded.
    records = query_records(
        conn,
        dataset=dataset,
        order_by="record_id",
    )

    # Optional species filter (exact lowercase equality, whitespace-stripped). Done in
    # Python so behaviour is identical regardless of store-side collation.
    if only_species is not None:
        target = only_species.strip().lower()
        records = [
            r for r in records
            if r.species is not None and r.species.strip().lower() == target
        ]

    if limit is not None:
        records = records[:limit]

    n_total = len(records)

    # Partition into "skip (already embedded)" vs "to embed". Under only_missing, a record
    # is skipped iff it already has embedding_ref set in the store. Under only_missing
    # False, everything is (re)embedded.
    skipped_records = []
    to_embed = []
    for r in records:
        if only_missing and r.embedding_ref is not None:
            skipped_records.append(r)
        else:
            to_embed.append(r)

    crop_paths: Dict[str, str] = {r.record_id: r.crop_path for r in to_embed}

    merged = embed_crops(
        crop_paths,
        cache_path,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    n_skipped = len(skipped_records)
    n_embedded = 0
    n_failed = 0
    failed_ids: List[str] = []
    embedding_dim: Optional[int] = None

    for rid in crop_paths:
        if rid in merged:
            vec = merged[rid]
            if embedding_dim is None:
                embedding_dim = int(np.asarray(vec).reshape(-1).shape[0])
            n_embedded += 1
            # Write the reference back into the store (refreshes updated_at).
            update_embedding(conn, rid, embedding_ref=rid, embedding_path=cache_path)
        else:
            n_failed += 1
            failed_ids.append(rid)

    # If nothing was newly embedded but skipped records exist, surface their dim too.
    if embedding_dim is None and skipped_records:
        try:
            cache = _load_cache(cache_path)
            for r in skipped_records:
                ref = r.embedding_ref
                if ref in cache:
                    embedding_dim = int(np.asarray(cache[ref]).reshape(-1).shape[0])
                    break
        except Exception:
            pass

    return EmbedResult(
        dataset=dataset,
        cache_path=cache_path,
        model_name=model_name,
        model_label=label,
        embedding_dim=embedding_dim,
        n_total=n_total,
        n_embedded=n_embedded,
        n_skipped=n_skipped,
        n_failed=n_failed,
        failed_ids=failed_ids,
    )


# --------------------------------------------------------------------------- #
# Read side (T05 / T07)
# --------------------------------------------------------------------------- #

def load_embeddings(cache_path: str) -> Dict[str, np.ndarray]:
    """Load a cache pickle. Thin wrapper over pickle for downstream symmetry. Raises
    ``FileNotFoundError`` with a clear message if absent."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"embedding cache not found: {cache_path}")
    with open(cache_path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"embedding cache {cache_path!r} is not a Dict[str, np.ndarray]")
    return data


def get_embedding_matrix(
    conn,
    *,
    dataset: Optional[str] = None,
    normalize: bool = True,
    only_clustered: bool = False,
) -> Tuple[np.ndarray, list]:
    """READ side for T05. Query records with ``embedding_ref NOT NULL``
    (``has_embedding=True``), load each vector from its ``embedding_path`` /
    ``embedding_ref``, stack into an ``(N, D)`` float32 matrix in a stable (sorted)
    ``record_id`` order, and return ``(matrix, record_ids)``.

    If ``normalize=True``, L2-normalize each row (so cosine == dot product), with an
    epsilon guard. Groups by ``embedding_path`` so each pickle is loaded once. Raises a
    clear ``RuntimeError`` naming the ``record_id`` + path if ``embedding_ref`` is set but
    the key is missing from the pickle (stale cache). ``only_clustered`` is a convenience
    pass-through filter (``cluster_id NOT NULL``), default off; T05 typically calls with
    defaults BEFORE clustering exists.
    """
    where_sql = None
    if only_clustered:
        where_sql = "cluster_id IS NOT NULL"

    records = query_records(
        conn,
        dataset=dataset,
        has_embedding=True,
        where_sql=where_sql,
        order_by="record_id",
    )
    # Stable, deterministic ordering by record_id.
    records = sorted(records, key=lambda r: r.record_id)

    # Group by embedding_path so each pickle is loaded exactly once.
    caches: Dict[str, Dict[str, np.ndarray]] = {}
    vectors: List[np.ndarray] = []
    record_ids: List[str] = []
    dim: Optional[int] = None

    for r in records:
        path = r.embedding_path
        ref = r.embedding_ref
        if path is None or ref is None:
            # has_embedding=True implies embedding_ref NOT NULL; embedding_path should be
            # set by embed_records. Treat a missing path as a stale/inconsistent ref.
            raise RuntimeError(
                f"record {r.record_id!r} has embedding_ref={ref!r} but embedding_path="
                f"{path!r}; cannot resolve vector (inconsistent store)."
            )
        if path not in caches:
            caches[path] = load_embeddings(path)
        cache = caches[path]
        if ref not in cache:
            raise RuntimeError(
                f"stale embedding cache: record {r.record_id!r} references key {ref!r} "
                f"in {path!r}, but that key is missing from the pickle."
            )
        vec = np.asarray(cache[ref], dtype=np.float32).reshape(-1)
        if dim is None:
            dim = int(vec.shape[0])
        elif vec.shape[0] != dim:
            raise RuntimeError(
                f"embedding dim mismatch for record {r.record_id!r}: expected {dim}, "
                f"got {vec.shape[0]} (mixed models in one matrix?)."
            )
        vectors.append(vec)
        record_ids.append(r.record_id)

    if not vectors:
        matrix = np.empty((0, 0), dtype=np.float32)
    else:
        matrix = np.stack(vectors, axis=0).astype(np.float32)
        if normalize:
            matrix = _l2_normalize_rows(matrix)

    return matrix, record_ids


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows with an epsilon guard (mirrors
    ``nested_importance_sampling._l2_normalize_rows``). Zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    return (matrix / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid_demo.embed",
        description="T04 global embedding service for the lynx re-ID demo.",
    )
    parser.add_argument("--db", default=None,
                        help=f"store DB path (default {store.DEFAULT_DB_PATH}).")
    parser.add_argument("--dataset", default=None,
                        help="dataset selector to scope which records to embed.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"global embedding model (default {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--checkpoint", default=None,
                        help="optional local checkpoint path (passed through unchanged).")
    parser.add_argument("--cache-dir", default=DEFAULT_EMB_DIR,
                        help=f"embeddings cache dir (default {DEFAULT_EMB_DIR}).")
    parser.add_argument("--only-species", default=None,
                        help="embed only records whose species matches (case-insensitive).")
    parser.add_argument("--all", action="store_true",
                        help="recompute even already-embedded records (only_missing=False).")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on number of records processed.")
    parser.add_argument("--device", default=None,
                        help="torch device override (e.g. cuda, cpu).")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    db_path = args.db or store.DEFAULT_DB_PATH
    try:
        conn = connect(db_path)
    except Exception as exc:  # pragma: no cover - connect failure is environmental
        print(f"[embed] FATAL: could not open store {db_path!r}: {exc}", file=sys.stderr)
        return 1

    try:
        res = embed_records(
            conn,
            dataset=args.dataset,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            only_missing=not args.all,
            only_species=args.only_species,
            limit=args.limit,
            device=args.device,
        )
    except Exception as exc:
        print(f"[embed] FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(
        f"T04 embed: dataset={res.dataset} dim={res.embedding_dim} total={res.n_total} "
        f"embedded={res.n_embedded} skipped={res.n_skipped} failed={res.n_failed} "
        f"cache={res.cache_path}"
    )

    # "Nothing to do" is an error worth surfacing: 0 found to embed AND none already done.
    if res.n_embedded == 0 and res.n_skipped == 0:
        print("[embed] nothing to embed and nothing already embedded.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
