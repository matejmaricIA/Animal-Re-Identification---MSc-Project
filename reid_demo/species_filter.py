"""reid_demo.species_filter — SpeciesNet species-classification + target filter (T03).

This stage attaches a species label + confidence + full taxonomy string onto the
per-crop detection records produced by T02, and marks which crops are the target
species (lynx for field data; leopard/tiger for the public eval datasets).

Three entry points:

* ``ingest_speciesnet_json`` — PRIMARY path: re-use a precomputed SpeciesNet
  predictions JSON (e.g. ``data/MedvednicaDS/animals_classified.json``) and write
  its per-detection classifications onto the matching T01 records. Joins each JSON
  detection to a record by ``(source_stem, bbox)`` NEAREST-MATCH (D3), NOT by a
  positional ``det_index`` or any index/index+1 heuristic. No model, no re-cropping.
* ``classify_and_filter`` — end-to-end. Delegates to ``ingest_speciesnet_json`` when
  ``reuse_existing_json`` is given; otherwise runs the SpeciesNet CLI live on the
  dataset's crops (SECONDARY path, needs GPU/model; degrades to a clear RuntimeError).
* ``set_known_species`` — stamp a fixed species (LeopardID2022 -> ``leopard``,
  ATRW -> ``tiger``) on every row of a dataset with NO model (B-track, D7d).

The ``species_kept`` flag T03 writes (into ``extra_json`` via ``store.update_extra``)
is REPORT-ONLY (D7d): T09's Medvednica report and human triage read it, but T05 does
NOT — T05 selects clustering inputs by the ``species`` column. T03 NEVER deletes rows.

All store access goes through ``reid_demo.store`` (T01); ``extra_json`` is written
ONLY through ``store.update_extra`` (never raw SQL). Heavy/optional deps (PIL,
SpeciesNet CLI) are used only on the live path and imported lazily/guarded, so the
primary path and ``set_known_species`` need zero SpeciesNet dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reid_demo import store


# --------------------------------------------------------------------------- #
# Target-species policy (tolerant alias / inclusion sets)
# --------------------------------------------------------------------------- #

#: Per-target lowercase common-name alias sets. Deliberately a bit inclusive for the
#: demo so the biologist's T08 review can prune low-confidence keeps rather than miss
#: a borderline felid. Easy to edit. NOTE: ``bobcat`` is genus ``lynx`` species
#: ``rufus`` — a true lynx-genus felid — so it is INCLUDED under target ``lynx`` both
#: by name here AND (more robustly) by the genus match in ``is_target_species``.
TARGET_SPECIES_ALIASES: Dict[str, set] = {
    "lynx": {
        "lynx", "eurasian lynx", "canada lynx", "iberian lynx",
        "wild cat", "cat family", "bobcat",
    },
    "leopard": {
        "leopard", "african leopard", "amur leopard", "snow leopard",
        "panthera pardus",
    },
    "tiger": {
        "tiger", "amur tiger", "bengal tiger", "panthera tigris",
    },
}

#: Genus (full-taxonomy field index 4) that should match each target by taxonomy,
#: regardless of the common name SpeciesNet emits. e.g. genus ``lynx`` -> target
#: ``lynx`` (catches ``bobcat``/``rufus``); genus ``panthera`` is shared by leopard
#: and tiger so it is gated on the species epithet (index 5) below.
_TARGET_GENERA: Dict[str, set] = {
    "lynx": {"lynx"},
    "leopard": {"panthera"},
    "tiger": {"panthera"},
}

#: Species epithet required when the genus alone is ambiguous (panthera). If the
#: target is in this map, a genus match must ALSO have one of these epithets.
_TARGET_EPITHETS: Dict[str, set] = {
    "leopard": {"pardus"},
    "tiger": {"tigris"},
}


def is_target_species(species_name: str, target_species: str) -> bool:
    """True if the SpeciesNet name matches the target species.

    ``species_name`` may be a bare common name (``'eurasian lynx'``) OR a full
    ``'uuid;class;order;family;genus;species;common_name'`` taxonomy string.
    Matching is case-insensitive and uses ``TARGET_SPECIES_ALIASES[target_species]``;
    it ALSO matches when the genus field of a full taxonomy string equals the target's
    genus (e.g. genus ``lynx`` for target ``lynx``; genus ``panthera`` + epithet
    ``pardus`` for ``leopard``). ``target_species`` must be a key of
    ``TARGET_SPECIES_ALIASES`` — otherwise ``KeyError`` (listing valid targets).
    """
    if target_species not in TARGET_SPECIES_ALIASES:
        raise KeyError(
            f"unknown target_species {target_species!r}; "
            f"valid targets: {sorted(TARGET_SPECIES_ALIASES)}"
        )
    if not species_name:
        return False

    aliases = TARGET_SPECIES_ALIASES[target_species]

    if ";" in species_name:
        parts = [p.strip().lower() for p in species_name.split(";")]
        common = parts[-1] if parts else ""
        genus = parts[4] if len(parts) > 4 else ""
        epithet = parts[5] if len(parts) > 5 else ""
        if common in aliases:
            return True
        # Genus-based taxonomy match (e.g. genus 'lynx' -> bobcat kept under 'lynx').
        if genus and genus in _TARGET_GENERA.get(target_species, set()):
            req_epithets = _TARGET_EPITHETS.get(target_species)
            if req_epithets is None:
                return True
            if epithet in req_epithets:
                return True
        return False

    return species_name.strip().lower() in aliases


# --------------------------------------------------------------------------- #
# Result summary
# --------------------------------------------------------------------------- #

@dataclass
class SpeciesFilterResult:
    dataset: str
    target_species: str
    n_classified: int                  # records with a non-null species after this run
    n_kept: int                        # species_kept == 1
    n_dropped: int                     # classified but species_kept == 0
    n_unclassified: int                # T01 rows in dataset left with species NULL
    skipped_unmatched: int             # JSON detections with no matching T01 record
    species_breakdown: Dict[str, int] = field(default_factory=dict)  # {common_name: count}
    kept_record_ids: List[str] = field(default_factory=list)         # species_kept == 1


# --------------------------------------------------------------------------- #
# bbox matching helpers (D3 — nearest-match, NOT positional)
# --------------------------------------------------------------------------- #

#: Max L2 distance over (x, y, w, h) for a JSON detection to count as matching a
#: stored T01 bbox. Small — the same bbox should be near-identical to the one T02
#: stored. Detections with no near row are genuine ``skipped_unmatched`` (expected,
#: since T02's per-detection conf>=0.5 set may not be 1:1 with the raw JSON).
BBOX_MATCH_TOLERANCE: float = 0.05


def _bbox_l2(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """L2 distance over the four normalized bbox coords."""
    return sum((float(ai) - float(bi)) ** 2 for ai, bi in zip(a, b)) ** 0.5


def _common_name(class_string: str) -> str:
    """Human-readable common name = last ';'-field of a SpeciesNet taxonomy string."""
    return class_string.split(";")[-1].strip().lower()


# --------------------------------------------------------------------------- #
# Core write: apply one (species, conf, class) to one record + keep flag
# --------------------------------------------------------------------------- #

def _apply_classification(conn, record_id: str, class_string: str, score: float,
                          *, target_species: str, keep_threshold: float) -> Tuple[str, bool]:
    """Write species fields + the species_kept flag for one record. Returns
    (common_name, kept)."""
    common = _common_name(class_string)
    store.update_species(conn, record_id, common, float(score), class_string)
    kept = (
        common != "blank"
        and is_target_species(class_string, target_species)
        and float(score) >= keep_threshold
    )
    store.update_extra(conn, record_id, "species_kept", 1 if kept else 0)
    return common, kept


def _summarize(conn, *, dataset: str, target_species: str,
               drop_nontarget: bool, skipped_unmatched: int) -> SpeciesFilterResult:
    """Compute a SpeciesFilterResult from the store after writes (authoritative)."""
    rows = store.query_records(conn, dataset=dataset)
    n_classified = 0
    n_unclassified = 0
    n_kept = 0
    breakdown: Dict[str, int] = {}
    kept_ids: List[str] = []
    for r in rows:
        if r.species is None:
            n_unclassified += 1
            continue
        n_classified += 1
        breakdown[r.species] = breakdown.get(r.species, 0) + 1
        try:
            extra = json.loads(r.extra_json) if r.extra_json else {}
        except (TypeError, json.JSONDecodeError):
            extra = {}
        if extra.get("species_kept") == 1:
            n_kept += 1
            kept_ids.append(r.record_id)
    n_dropped = n_classified - n_kept
    if drop_nontarget:
        # drop_nontarget only affects what goes into kept_record_ids; it NEVER deletes
        # rows and does not change the counts (kept_ids already == species_kept==1).
        kept_ids = list(kept_ids)
    return SpeciesFilterResult(
        dataset=dataset,
        target_species=target_species,
        n_classified=n_classified,
        n_kept=n_kept,
        n_dropped=n_dropped,
        n_unclassified=n_unclassified,
        skipped_unmatched=skipped_unmatched,
        species_breakdown=breakdown,
        kept_record_ids=kept_ids,
    )


# --------------------------------------------------------------------------- #
# PRIMARY path: ingest a precomputed SpeciesNet predictions JSON
# --------------------------------------------------------------------------- #

def ingest_speciesnet_json(
    conn,
    json_path: str,
    *,
    dataset: str,
    target_species: str,
    keep_threshold: float = 0.0,
    drop_nontarget: bool = False,
    species_index: int = 0,
) -> SpeciesFilterResult:
    """Write per-detection classifications from a SpeciesNet JSON onto T01 records.

    For each detection: ``stem = Path(filepath).stem``; among the dataset's rows with
    that ``source_stem`` pick the one whose stored bbox is NEAREST (min L2 over the four
    normalized coords) to the JSON detection's bbox, requiring L2 < BBOX_MATCH_TOLERANCE.
    Matched rows are greedily consumed so two detections in a frame cannot claim the same
    row. Writes ``species=classes[k].split(';')[-1]``, ``species_conf=scores[k]``,
    ``species_class=classes[k]`` (``k=species_index``) via ``store.update_species``, then
    ``extra_json['species_kept']`` via ``store.update_extra``. Detections with no matching
    row -> ``skipped_unmatched``; rows never classified stay NULL (``n_unclassified``).
    Idempotent.
    """
    if target_species not in TARGET_SPECIES_ALIASES:
        raise KeyError(
            f"unknown target_species {target_species!r}; "
            f"valid targets: {sorted(TARGET_SPECIES_ALIASES)}"
        )
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    predictions = payload.get("predictions", [])

    # Cache the dataset's rows by source_stem; track which record_ids are consumed.
    rows_by_stem: Dict[str, List[store.DetectionRecord]] = {}
    for rec in store.query_records(conn, dataset=dataset):
        rows_by_stem.setdefault(rec.source_stem, []).append(rec)
    consumed: set = set()

    skipped_unmatched = 0

    for pred in predictions:
        filepath = pred.get("filepath")
        if not filepath:
            continue
        stem = Path(filepath).stem
        candidates = rows_by_stem.get(stem, [])
        for det in pred.get("detections", []):
            classifications = det.get("classifications")
            if not classifications:
                # No classification block at all -> nothing to write; the row (if any)
                # stays species=NULL and is counted as n_unclassified in the summary.
                continue
            classes = classifications.get("classes") or []
            scores = classifications.get("scores") or []
            if species_index >= len(classes) or species_index >= len(scores):
                continue
            det_bbox = tuple(det.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            if len(det_bbox) != 4:
                skipped_unmatched += 1
                continue

            # Nearest unconsumed row for this stem within tolerance (greedy).
            best = None
            best_d = None
            for rec in candidates:
                if rec.record_id in consumed:
                    continue
                d = _bbox_l2(det_bbox, (rec.bbox_x, rec.bbox_y, rec.bbox_w, rec.bbox_h))
                if best_d is None or d < best_d:
                    best_d = d
                    best = rec
            if best is None or best_d > BBOX_MATCH_TOLERANCE:
                skipped_unmatched += 1
                continue

            consumed.add(best.record_id)
            _apply_classification(
                conn, best.record_id, classes[species_index], scores[species_index],
                target_species=target_species, keep_threshold=keep_threshold,
            )

    return _summarize(
        conn, dataset=dataset, target_species=target_species,
        drop_nontarget=drop_nontarget, skipped_unmatched=skipped_unmatched,
    )


# --------------------------------------------------------------------------- #
# SECONDARY path: run SpeciesNet live on the dataset's crops
# --------------------------------------------------------------------------- #

def classify_and_filter(
    conn,
    *,
    dataset: str,
    target_species: str,
    keep_threshold: float = 0.0,
    country: str = "HRV",
    batch_size: int = 16,
    model: str = "kaggle:google/speciesnet/pyTorch/v4.0.1a",
    reuse_existing_json: Optional[str] = None,
    drop_nontarget: bool = False,
) -> SpeciesFilterResult:
    """End-to-end species classification + target filter.

    If ``reuse_existing_json`` is set, delegates to ``ingest_speciesnet_json`` (no
    model). Otherwise runs the SpeciesNet CLI once on a temp folder of the dataset's
    crops (named ``<record_id>.jpg`` so the stitch back is unambiguous), then writes
    the per-crop classifications. Raises a clear ``RuntimeError`` naming SpeciesNet if
    the CLI/model (or PIL) is unavailable.
    """
    if reuse_existing_json is not None:
        return ingest_speciesnet_json(
            conn, reuse_existing_json, dataset=dataset, target_species=target_species,
            keep_threshold=keep_threshold, drop_nontarget=drop_nontarget,
        )

    if target_species not in TARGET_SPECIES_ALIASES:
        raise KeyError(
            f"unknown target_species {target_species!r}; "
            f"valid targets: {sorted(TARGET_SPECIES_ALIASES)}"
        )

    try:
        from PIL import Image  # noqa: F401  (presence check; live path only)
    except ImportError as exc:
        raise RuntimeError(
            "SpeciesNet CLI/model not available: Pillow (PIL) is required to stage "
            f"crops for the live path ({exc}). Use --reuse-json with a precomputed "
            "predictions file, or install speciesnet + Pillow."
        ) from exc

    rows = store.query_records(conn, dataset=dataset)
    rows = [r for r in rows if r.crop_path and os.path.exists(r.crop_path)]
    if not rows:
        raise RuntimeError(
            f"SpeciesNet CLI/model not available: no crop files found for dataset "
            f"{dataset!r}. Use --reuse-json with a precomputed predictions file."
        )

    crops_dir = Path(tempfile.mkdtemp(prefix="reid_t03_crops_"))
    # Map temp crop filename (== record_id) -> record_id for unambiguous stitch-back.
    name_to_rid: Dict[str, str] = {}
    crop_pred_json = Path(tempfile.gettempdir()) / f"reid_t03_pred_{os.getpid()}.json"
    try:
        for r in rows:
            dest = crops_dir / f"{r.record_id}.jpg"
            shutil.copy(r.crop_path, dest)
            name_to_rid[dest.name] = r.record_id

        cmd = [
            sys.executable, "-m", "speciesnet.scripts.run_model",
            "--folders", str(crops_dir),
            "--predictions_json", str(crop_pred_json),
            "--batch_size", str(batch_size),
            "--model", model,
            "--country", country,
        ]
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"SpeciesNet CLI/model not available: {exc}. Use --reuse-json with a "
                "precomputed predictions file, or install speciesnet."
            ) from exc

        try:
            with open(crop_pred_json, "r", encoding="utf-8") as fh:
                crop_preds = json.load(fh).get("predictions", [])
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"SpeciesNet CLI/model not available: could not read its predictions "
                f"output ({exc}). Use --reuse-json with a precomputed predictions file."
            ) from exc

        skipped_unmatched = 0
        for cp in crop_preds:
            classifications = cp.get("classifications")
            if not classifications:
                continue
            classes = classifications.get("classes") or []
            scores = classifications.get("scores") or []
            if not classes or not scores:
                continue
            crop_file = Path(cp.get("filepath", "")).name
            rid = name_to_rid.get(crop_file)
            if rid is None:
                skipped_unmatched += 1
                continue
            _apply_classification(
                conn, rid, classes[0], scores[0],
                target_species=target_species, keep_threshold=keep_threshold,
            )

        return _summarize(
            conn, dataset=dataset, target_species=target_species,
            drop_nontarget=drop_nontarget, skipped_unmatched=skipped_unmatched,
        )
    finally:
        shutil.rmtree(crops_dir, ignore_errors=True)
        try:
            crop_pred_json.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Known-species shortcut (LeopardID2022 / ATRW) — B-track, no model (D7d)
# --------------------------------------------------------------------------- #

def set_known_species(
    conn,
    *,
    dataset: str,
    species: str,
    species_conf: float = 1.0,
) -> int:
    """Stamp a fixed ``species`` on EVERY T01 row of ``dataset`` (no model).

    The B-track species stage (still run for LeopardID2022 -> ``leopard`` / ATRW ->
    ``tiger`` even though the SpeciesNet model branch is skipped, D7d). Sets
    ``species_class`` to a synthetic ``<species>`` string and ``species_kept=1`` for
    all rows via ``store.update_species`` + ``store.update_extra``. Does NOT touch
    ``gt_identity`` or ``orientation`` (T02 is their sole owner, D1). Returns rows
    written.
    """
    rows = store.query_records(conn, dataset=dataset)
    n = 0
    for r in rows:
        store.update_species(conn, r.record_id, species, float(species_conf), species)
        store.update_extra(conn, r.record_id, "species_kept", 1)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_result(res: SpeciesFilterResult, *, top: int = 20) -> None:
    print(f"dataset:           {res.dataset}")
    print(f"target_species:    {res.target_species}")
    print(f"n_classified:      {res.n_classified}")
    print(f"n_kept:            {res.n_kept}")
    print(f"n_dropped:         {res.n_dropped}")
    print(f"n_unclassified:    {res.n_unclassified}")
    print(f"skipped_unmatched: {res.skipped_unmatched}")
    print(f"species_breakdown (top {top}):")
    ranked = sorted(res.species_breakdown.items(), key=lambda kv: -kv[1])[:top]
    for name, count in ranked:
        print(f"  {count:6d}  {name}")


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reid_demo.species_filter",
        description="SpeciesNet species-filter adapter (T03).",
    )
    parser.add_argument("--dataset", required=True, help="dataset (T01 rows) to update")
    parser.add_argument("--target", default="lynx",
                        help="target species key into TARGET_SPECIES_ALIASES")
    parser.add_argument("--reuse-json", default=None,
                        help="precomputed SpeciesNet predictions JSON (PRIMARY path)")
    parser.add_argument("--set-known", default=None, metavar="SPECIES",
                        help="stamp a fixed species on every row (no model); "
                             "e.g. --set-known leopard")
    parser.add_argument("--keep-threshold", type=float, default=0.0,
                        help="min species_conf to keep a target-species crop")
    parser.add_argument("--drop-nontarget", action="store_true",
                        help="exclude non-kept rows from kept_record_ids (never deletes)")
    parser.add_argument("--db", default=store.DEFAULT_DB_PATH, help="DB path")
    args = parser.parse_args(argv)

    conn = store.connect(args.db)

    if args.set_known:
        n = set_known_species(conn, dataset=args.dataset, species=args.set_known)
        print(f"stamped {n} rows of {args.dataset!r} with species={args.set_known!r}, "
              f"species_kept=1")
        return 0

    res = classify_and_filter(
        conn,
        dataset=args.dataset,
        target_species=args.target,
        keep_threshold=args.keep_threshold,
        reuse_existing_json=args.reuse_json,
        drop_nontarget=args.drop_nontarget,
    )
    _print_result(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
