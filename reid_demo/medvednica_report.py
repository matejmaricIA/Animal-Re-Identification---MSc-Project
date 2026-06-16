"""T09 — Medvednica filtering report.

Read-only consumer of the already-computed Medvednica camera-trap artifacts under
``data/MedvednicaDS/`` (MegaDetector + SpeciesNet output) that renders a
non-technical "look, this works on YOUR cameras" pitch page:

    * ``medvednica_report.md``           — plain-language funnel + species story
    * ``figures/detection_funnel.png``   — bar chart of the triage funnel
    * ``figures/species_breakdown.png``  — bar chart of species among kept animals
    * ``figures/example_crops.png``      — montage of example animal crops
    * ``medvednica_summary.json``        — every number the page cites (T10 contract)

This module is **read-only**: it never re-runs any model (MegaDetector / SpeciesNet /
cropping) and never writes detection records into the T01 store. All numbers come
straight from the on-disk JSON artifacts. The optional ``--use-store`` path is a
non-authoritative cross-check only and is imported lazily so the module runs with
``use_store=False`` even when ``data/reid_demo/reid_demo.sqlite`` is absent.

Design decisions honoured (see STATUS_BOARD.md):

* **D7b** — "empty frames" = frames with ZERO detections of ANY category
  (``total_frames - frames_with_any_detection``). Frames whose only detections are
  person/vehicle are a SEPARATE ``person_or_vehicle_frames`` bucket, never folded
  into ``empty_frames``.
* **D3** — ``animal_detections_kept`` / ``kept_frames`` are read STRAIGHT from
  ``detections_cleaned.json`` on disk, never recomputed from the raw MD or by
  re-running ``utils/clean_detections.py``.

CLI::

    python -m reid_demo.medvednica_report [--data-dir ...] [--out-dir ...]
        [--top-k-species 12] [--n-example-crops 12] [--species-filter "eurasian lynx"]
        [--use-store] [--db <path>] [--seed 0]
    python -m reid_demo.medvednica_report --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

SCHEMA_VERSION = 1

# Required input artifact filenames (relative to --data-dir).
MD_RESULTS_NAME = "megadetector_results.json"
CLEANED_NAME = "detections_cleaned.json"
CLASSIFIED_NAME = "animals_classified.json"
CSV_NAME = "trail_cam_data.csv"
CROPS_DIRNAME = "animal_crops"

# MegaDetector detection category codes (see detection_categories in the artifacts).
CAT_ANIMAL = "1"
CAT_PERSON = "2"
CAT_VEHICLE = "3"

# SpeciesNet "no identifiable animal" sentinel — a legitimate class, reported apart.
BLANK_LABEL = "blank"


# --------------------------------------------------------------------------- #
# Pure helpers (contracted, unit-tested surface)
# --------------------------------------------------------------------------- #

_CONF_RE = re.compile(r"_conf(\d+)$")
_CROP_RE = re.compile(r"_crop(\d+)$")


def parse_crop_filename(name: str) -> Tuple[str, int, Optional[int]]:
    """Parse a crop filename into ``(source_stem, crop_index, conf_percent_or_None)``.

    ``'02020401_crop1_conf92.jpg' -> ('02020401', 1, 92)``
    ``'IMG_0066_crop1_conf78.jpg' -> ('IMG_0066', 1, 78)``

    Strips the trailing ``_conf\\d+`` to recover the confidence percent (``None`` if
    absent), then everything before ``_crop\\d+`` to recover the stem. Tolerates a
    missing ``_conf`` suffix and a missing ``_crop`` index (``crop_index`` defaults to
    ``0`` if no ``_crop`` token is present). Operates on the basename only.
    """
    base = os.path.basename(name)
    # Drop a trailing image extension if present (case-insensitive).
    root, _ext = os.path.splitext(base)

    conf_percent: Optional[int] = None
    m_conf = _CONF_RE.search(root)
    if m_conf:
        conf_percent = int(m_conf.group(1))
        root = root[: m_conf.start()]

    crop_index = 0
    m_crop = _CROP_RE.search(root)
    if m_crop:
        crop_index = int(m_crop.group(1))
        stem = root[: m_crop.start()]
    else:
        stem = root

    return stem, crop_index, conf_percent


def species_from_classes(classes: List[str]) -> str:
    """Map a SpeciesNet taxonomy-string list to the human-readable common name.

    Returns ``classes[0].split(';')[-1]`` (e.g.
    ``'uuid;mammalia;...;wild boar' -> 'wild boar'``), or ``''`` when ``classes`` is
    empty/falsy. Single source of truth for the taxonomy-string -> common-name rule
    (mirrors how the T03 store ``species`` column is populated).
    """
    if not classes:
        return ""
    first = classes[0]
    if not first:
        return ""
    return first.split(";")[-1]


def compute_funnel(md_results: dict, cleaned: dict) -> dict:
    """Compute the detection funnel from raw ``megadetector_results.json`` + ``detections_cleaned.json``.

    Returns the ``funnel`` sub-dict of the summary schema. Per **D7b**,
    ``empty_frames = total_frames - frames_with_any_detection`` (frames with ZERO
    detections of ANY category) and ``person_or_vehicle_frames`` is a SEPARATE bucket
    (non-empty frames with no animal but >=1 person/vehicle), never folded into
    ``empty_frames``. Per **D3**, ``animal_detections_kept`` / ``kept_frames`` are read
    straight from ``cleaned`` (the on-disk detections_cleaned.json), NOT recomputed.
    """
    images = md_results.get("images", [])
    total_frames = len(images)

    frames_with_any_detection = 0
    frames_with_animal = 0
    person_detections = 0
    vehicle_detections = 0
    animal_detections_raw = 0
    person_or_vehicle_frames = 0

    for im in images:
        dets = im.get("detections") or []
        if not dets:
            continue  # empty frame (zero detections of any category)
        frames_with_any_detection += 1

        has_animal = False
        has_person_or_vehicle = False
        for d in dets:
            cat = d.get("category")
            if cat == CAT_ANIMAL:
                animal_detections_raw += 1
                has_animal = True
            elif cat == CAT_PERSON:
                person_detections += 1
                has_person_or_vehicle = True
            elif cat == CAT_VEHICLE:
                vehicle_detections += 1
                has_person_or_vehicle = True

        if has_animal:
            frames_with_animal += 1
        elif has_person_or_vehicle:
            # Non-empty, no animal, has person/vehicle: the SEPARATE D7b bucket.
            person_or_vehicle_frames += 1

    empty_frames = total_frames - frames_with_any_detection
    pct_empty_removed = (
        round(empty_frames / total_frames * 100, 1) if total_frames else 0.0
    )

    # D3: trust the on-disk cleaned file verbatim — do NOT re-filter the raw MD.
    cleaned_preds = cleaned.get("predictions", [])
    animal_detections_kept = sum(len(p.get("detections") or []) for p in cleaned_preds)
    kept_frames = sum(1 for p in cleaned_preds if (p.get("detections") or []))

    return {
        "total_frames": total_frames,
        "frames_with_any_detection": frames_with_any_detection,
        "empty_frames": empty_frames,
        "pct_empty_removed": pct_empty_removed,
        "frames_with_animal": frames_with_animal,
        "person_detections": person_detections,
        "vehicle_detections": vehicle_detections,
        "person_or_vehicle_frames": person_or_vehicle_frames,
        "animal_detections_raw": animal_detections_raw,
        "animal_detections_kept": animal_detections_kept,
        "kept_frames": kept_frames,
    }


def compute_species_counts(classified: dict, *, include_blank: bool = False) -> dict:
    """Tally ``{common_name: count}`` over classified detections in animals_classified.json.

    Iterates ``predictions[*].detections[*].classifications.classes`` and counts the
    common name from :func:`species_from_classes`. ``'blank'`` is excluded unless
    ``include_blank=True``. Returns a plain ``dict`` (insertion order = first-seen).
    """
    counts: "Counter[str]" = Counter()
    for pred in classified.get("predictions", []):
        for det in pred.get("detections") or []:
            clf = det.get("classifications")
            if not clf:
                continue
            name = species_from_classes(clf.get("classes") or [])
            if not name:
                continue
            if name == BLANK_LABEL and not include_blank:
                continue
            counts[name] += 1
    return dict(counts)


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #

def _basename_stem(filepath: str) -> str:
    """``'animal_images/IMG_0066.JPG' -> 'IMG_0066'`` (basename, extension dropped)."""
    return os.path.splitext(os.path.basename(filepath or ""))[0]


def _load_json(path: str, label: str) -> dict:
    """Load a required JSON artifact, raising a clear FileNotFoundError if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required Medvednica artifact missing: {label} expected at {path!r}. "
            "T09 reads pre-computed artifacts and never re-runs any model; "
            "ensure the MedvednicaDS dump is complete."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_stem_species_index(classified: dict) -> Dict[str, List[dict]]:
    """Map ``source_stem -> [ {species, species_conf, detector_conf, crop_index}, ... ]``.

    Joins crops back to species/confidence by the basename stem of each classified
    record's ``filepath``. Detections within a frame are ordered as stored, so
    ``crop_index`` (1-based, matching the ``_cropN`` suffix on disk) maps to position.
    """
    index: Dict[str, List[dict]] = defaultdict(list)
    for pred in classified.get("predictions", []):
        stem = _basename_stem(pred.get("filepath", ""))
        if not stem:
            continue
        crop_index = 0
        for det in pred.get("detections") or []:
            clf = det.get("classifications")
            if not clf:
                continue
            crop_index += 1  # 1-based, mirrors animal_crops naming
            classes = clf.get("classes") or []
            scores = clf.get("scores") or []
            species = species_from_classes(classes)
            species_conf = float(scores[0]) if scores else None
            index[stem].append(
                {
                    "species": species,
                    "species_conf": species_conf,
                    "detector_conf": det.get("conf"),
                    "crop_index": crop_index,
                }
            )
    return index


def _parse_datetime(value: str) -> Optional[datetime]:
    """Lenient datetime parse for the artifact timestamp / CSV datetime fields."""
    if not value or value in ("Not available",):
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Last resort: first 10 chars as a date.
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _compute_temporal(classified: dict, csv_path: str) -> dict:
    """Date span + per-camera tally.

    Prefers per-record ``timestamp`` in animals_classified.json; tallies cameras from
    ``trail_cam_data.csv`` when present (``cameras={}`` if the CSV is missing).
    """
    dates: List[datetime] = []
    for pred in classified.get("predictions", []):
        dt = _parse_datetime(pred.get("timestamp", ""))
        if dt is not None:
            dates.append(dt)

    cameras: Dict[str, int] = {}
    csv_dates: List[datetime] = []
    if os.path.exists(csv_path):
        import csv as _csv

        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                cam = (row.get("camera") or "").strip()
                if cam:
                    cameras[cam] = cameras.get(cam, 0) + 1
                dt = _parse_datetime(row.get("datetime", ""))
                if dt is not None:
                    csv_dates.append(dt)

    # Prefer JSON timestamps; fall back to CSV datetimes for span if JSON has none.
    span_dates = dates if dates else csv_dates
    n_dated = len(dates) if dates else len(csv_dates)

    date_min = min(span_dates).strftime("%Y-%m-%d") if span_dates else None
    date_max = max(span_dates).strftime("%Y-%m-%d") if span_dates else None

    return {
        "date_min": date_min,
        "date_max": date_max,
        "n_dated_records": n_dated,
        "cameras": cameras,
    }


def _build_species_section(
    classified: dict, *, top_k_species: int
) -> Tuple[dict, List[Tuple[str, int]]]:
    """Build the ``species`` summary sub-dict + the chart series (top-K + 'other')."""
    counts = compute_species_counts(classified, include_blank=False)
    blank_counts = compute_species_counts(classified, include_blank=True)
    total_classified = sum(blank_counts.values())
    blank_detections = blank_counts.get(BLANK_LABEL, 0)
    real_species_detections = sum(counts.values())

    # Deterministic ordering: count desc, then name asc.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    top_k = []
    for name, count in ordered[:top_k_species]:
        pct = (
            round(count / real_species_detections * 100, 1)
            if real_species_detections
            else 0.0
        )
        top_k.append({"species": name, "count": count, "pct": pct})

    species = {
        "total_classified_detections": total_classified,
        "blank_detections": blank_detections,
        "real_species_detections": real_species_detections,
        "n_distinct_species": len(counts),  # excludes blank
        "counts": dict(ordered),  # blank excluded; ordered by count desc, name asc
        "top_k": top_k,
    }

    # Chart series: top-K bars, remainder folded into "other" (chart only, not counts).
    chart_series: List[Tuple[str, int]] = [(n, c) for n, c in ordered[:top_k_species]]
    other = sum(c for _, c in ordered[top_k_species:])
    if other > 0:
        chart_series.append(("other", other))
    return species, chart_series


def _compute_target_species(
    classified: dict, species_filter: List[str]
) -> dict:
    """Honest detection/frame counts for a species filter (e.g. ['eurasian lynx'])."""
    wanted = {s.strip().lower() for s in species_filter if s.strip()}
    detections = 0
    frames = 0
    for pred in classified.get("predictions", []):
        frame_hit = False
        for det in pred.get("detections") or []:
            clf = det.get("classifications")
            if not clf:
                continue
            name = species_from_classes(clf.get("classes") or [])
            if name.lower() in wanted:
                detections += 1
                frame_hit = True
        if frame_hit:
            frames += 1
    return {
        "filter": list(species_filter),
        "detections": detections,
        "frames": frames,
    }


def _select_example_crops(
    crops_dir: str,
    stem_index: Dict[str, List[dict]],
    *,
    n_example_crops: int,
    seed: int,
    species_filter: Optional[List[str]],
    data_dir: str,
) -> Tuple[List[dict], bool]:
    """Deterministically sample example crops, joining each to its species/conf.

    Returns ``(examples, used_filter_fallback)``. When ``species_filter`` is set, prefers
    crops whose species is in the filter; if none match, falls back to any crops and
    flags it. The sample is reproducible for a fixed ``seed`` (sorted candidate list +
    seeded RNG).
    """
    if not os.path.isdir(crops_dir):
        return [], False

    crop_names = sorted(
        f
        for f in os.listdir(crops_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    def species_for(name: str) -> dict:
        stem, idx, conf_pct = parse_crop_filename(name)
        records = stem_index.get(stem, [])
        rec = None
        for r in records:
            if r.get("crop_index") == idx:
                rec = r
                break
        if rec is None and records:
            rec = records[0]
        species = rec.get("species") if rec else None
        species_conf = rec.get("species_conf") if rec else None
        detector_conf = conf_pct / 100.0 if conf_pct is not None else None
        return {
            "crop_path": os.path.join(data_dir, CROPS_DIRNAME, name),
            "source_stem": stem,
            "species": species,
            "species_conf": species_conf,
            "detector_conf": detector_conf,
        }

    wanted: Optional[set] = None
    if species_filter:
        wanted = {s.strip().lower() for s in species_filter if s.strip()}

    used_filter_fallback = False
    candidates = crop_names
    if wanted:
        filtered = [
            n
            for n in crop_names
            if (species_for(n)["species"] or "").lower() in wanted
        ]
        if filtered:
            candidates = filtered
        else:
            used_filter_fallback = True
            candidates = crop_names

    rng = random.Random(seed)
    if len(candidates) <= n_example_crops:
        chosen = list(candidates)
    else:
        chosen = rng.sample(candidates, n_example_crops)
    # Stable, deterministic output ordering regardless of sample order.
    chosen = sorted(chosen)

    examples = [species_for(n) for n in chosen]
    return examples, used_filter_fallback


# --------------------------------------------------------------------------- #
# Figure rendering
# --------------------------------------------------------------------------- #

def _bar_value_labels(ax, bars, values, fmt="{:,}"):
    """Annotate bars with their integer value labels (plain numbers, no chart-junk)."""
    for bar, val in zip(bars, values):
        ax.annotate(
            fmt.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _render_funnel_figure(funnel: dict, out_path: str) -> None:
    """Bar chart of the triage funnel, in plain-language terms (photos / animals)."""
    stages = [
        ("All photos", funnel["total_frames"]),
        ("Empty frames\n(nothing in them)", funnel["empty_frames"]),
        ("People / vehicles\n(set aside)", funnel["person_or_vehicle_frames"]),
        ("Photos with\nanimals", funnel["frames_with_animal"]),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = ["#4C72B0", "#BBBBBB", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    _bar_value_labels(ax, bars, values)
    ax.set_ylabel("Number of photos")
    ax.set_title("Automatic triage of Medvednica camera-trap photos")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_species_figure(chart_series: List[Tuple[str, int]], out_path: str) -> None:
    """Horizontal bar chart of species counts among kept animal detections."""
    if not chart_series:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No classified species", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Largest at top.
    names = [n for n, _ in chart_series][::-1]
    values = [c for _, c in chart_series][::-1]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(names) + 1)))
    bars = ax.barh(names, values, color="#55A868")
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:,}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Number of animal detections")
    ax.set_title("Species found in the photos that contain animals")
    ax.set_xlim(0, max(values) * 1.12 if values else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_examples_montage(examples: List[dict], out_path: str) -> None:
    """Contact sheet of example animal crops with species/conf captions."""
    from PIL import Image  # local import; Pillow confirmed available

    readable: List[Tuple[dict, Any]] = []
    for ex in examples:
        path = ex["crop_path"]
        try:
            img = Image.open(path)
            img.load()
            readable.append((ex, img.convert("RGB")))
        except Exception:
            continue  # skip unreadable/missing crops gracefully

    if not readable:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(
            0.5,
            0.5,
            "No example crops available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    n = len(readable)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    # Normalise axes to a flat list.
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, (ex, img) in zip(axes, readable):
        ax.imshow(img)
        ax.axis("off")
        species = ex.get("species")
        conf = ex.get("species_conf")
        if species and conf is not None:
            caption = f"{species} ({conf:.0%})"
        elif species:
            caption = species
        else:
            caption = ex.get("source_stem", "")
        ax.set_title(caption, fontsize=9)

    for ax in axes[len(readable):]:
        ax.axis("off")

    fig.suptitle("Example animal detections", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown rendering (single source: the summary dict)
# --------------------------------------------------------------------------- #

def _fmt_int(n: Optional[int]) -> str:
    # Plain, comma-free integers so the exact figures (e.g. 8208) are greppable in
    # the rendered Markdown and the page/JSON numbers match literally.
    return str(n) if isinstance(n, (int,)) else str(n)


def render_markdown(summary: dict) -> str:
    """Render the report Markdown purely from the summary dict (page == JSON)."""
    f = summary["funnel"]
    sp = summary["species"]
    temporal = summary["temporal"]
    figures = summary["figures"]

    n_empty = f["empty_frames"]
    pct_empty = f["pct_empty_removed"]
    n_people_vehicle = f["person_or_vehicle_frames"]
    n_animal_frames = f["frames_with_animal"]
    n_species = sp["n_distinct_species"]

    lines: List[str] = []
    lines.append("# Automated triage of Medvednica camera-trap footage")
    lines.append("")
    lines.append(
        "_Turning a raw pile of trail-camera photos into a clean, sorted set of "
        "wildlife sightings — automatically._"
    )
    lines.append("")

    # 2. Headline — empty vs people/vehicle kept strictly distinct (D7b).
    lines.append("## The headline")
    lines.append("")
    lines.append(
        f"Of **{_fmt_int(f['total_frames'])}** photos, the system automatically "
        f"discarded **{_fmt_int(n_empty)} empty frames ({pct_empty}%)** — frames with "
        f"nothing in them at all — and separately set aside "
        f"**{_fmt_int(n_people_vehicle)} photos of people/vehicles**, leaving "
        f"**{_fmt_int(n_animal_frames)} photos containing animals**, which it sorted "
        f"into **{_fmt_int(n_species)} species**."
    )
    lines.append("")
    lines.append(
        "_Empty photos and people/vehicle photos are counted separately: an empty "
        "photo has nothing in it at all, while a people/vehicle photo did contain "
        "something — just not wildlife._"
    )
    lines.append("")

    # 3. Detection funnel.
    lines.append("## Detection funnel")
    lines.append("")
    lines.append(f"![Detection funnel]({figures['funnel']})")
    lines.append("")
    lines.append("| Stage | Photos |")
    lines.append("| --- | ---: |")
    lines.append(f"| All photos in | {_fmt_int(f['total_frames'])} |")
    lines.append(
        f"| Empty frames (nothing detected) | {_fmt_int(n_empty)} ({pct_empty}%) |"
    )
    lines.append(f"| People / vehicles (set aside) | {_fmt_int(n_people_vehicle)} |")
    lines.append(f"| Photos containing animals | {_fmt_int(n_animal_frames)} |")
    lines.append(
        f"| Animal detections kept (after quality filter) | "
        f"{_fmt_int(f['animal_detections_kept'])} across "
        f"{_fmt_int(f['kept_frames'])} photos |"
    )
    lines.append("")
    lines.append(
        f"_People and vehicles were seen as **{_fmt_int(f['person_detections'])}** "
        f"person and **{_fmt_int(f['vehicle_detections'])}** vehicle detections; "
        "those photos are removed from the wildlife set but reported here for "
        "transparency._"
    )
    lines.append("")

    # 4. What species were found.
    lines.append("## What species were found")
    lines.append("")
    lines.append(f"![Species breakdown]({figures['species']})")
    lines.append("")
    lines.append("| Species | Detections | Share |")
    lines.append("| --- | ---: | ---: |")
    for entry in sp["top_k"]:
        lines.append(
            f"| {entry['species']} | {_fmt_int(entry['count'])} | {entry['pct']}% |"
        )
    lines.append("")
    lines.append(
        f"Across **{_fmt_int(sp['real_species_detections'])}** identified animal "
        f"detections, the most common species above account for the bulk of the "
        f"wildlife seen. Separately, the classifier itself rejected "
        f"**{_fmt_int(sp['blank_detections'])}** crops as having no identifiable "
        "animal (reported apart from the species totals)."
    )
    lines.append("")

    # 5. Example detections.
    lines.append("## Example detections")
    lines.append("")
    lines.append(f"![Example animal crops]({figures['examples']})")
    lines.append("")
    lines.append(
        "_A sample of the animal crops the system pulled out of the footage, with "
        "the species the classifier assigned._"
    )
    lines.append("")

    # 6. When the footage was taken.
    lines.append("## When the footage was taken")
    lines.append("")
    if temporal.get("date_min") and temporal.get("date_max"):
        lines.append(
            f"The footage spans **{temporal['date_min']} to {temporal['date_max']}** "
            f"(**{_fmt_int(temporal['n_dated_records'])}** time-stamped photos with "
            "animals)."
        )
    else:
        lines.append("Timestamps were not available for this footage.")
    cams = temporal.get("cameras") or {}
    if cams:
        if set(cams) == {"unknown_camera"}:
            lines.append("")
            lines.append(
                "_Camera identities were not recorded in this dump (all listed as one "
                "`unknown_camera`); a per-camera breakdown appears automatically once "
                "real park footage carries camera labels._"
            )
        else:
            lines.append("")
            lines.append("| Camera | Photos |")
            lines.append("| --- | ---: |")
            for cam, n in sorted(cams.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"| {cam} | {_fmt_int(n)} |")
    lines.append("")

    # 7. Target species callout (only when --species-filter).
    target = summary.get("target_species")
    if target is not None:
        names = ", ".join(target["filter"])
        lines.append(f"## Target species: {names}")
        lines.append("")
        if target["detections"] > 0:
            lines.append(
                f"Found **{_fmt_int(target['detections'])}** {names} detection(s) "
                f"across **{_fmt_int(target['frames'])}** photo(s) in this sample."
            )
        else:
            lines.append(
                f"No {names} in this particular sample — but the same pipeline detects "
                "and identifies them once they appear, on park footage that includes "
                "them."
            )
        lines.append("")

    # 8. Method footnote.
    lines.append("## How this was done")
    lines.append("")
    lines.append(
        "_Animals were located with MegaDetector and identified with SpeciesNet. "
        "Only confident animal detections were kept (detector confidence at least "
        "0.5, animal category only); people and vehicles were removed. No accuracy "
        "tuning or model retraining was applied to this footage — these are the raw "
        "results of running the standard pipeline on the park's own photos._"
    )
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional store cross-check (lazy, never required)
# --------------------------------------------------------------------------- #

def _store_crosscheck(db_path: Optional[str], dataset: str) -> Optional[dict]:
    """Read-only cross-check from the T01 store; returns None / warns on any problem.

    Never authoritative; the JSON-derived funnel always wins. Imported lazily so the
    module loads without the store package state and runs when the DB is absent.
    """
    try:
        from reid_demo import store as _store
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[T09] store import failed; skipping cross-check: {exc}", file=sys.stderr)
        return None

    path = db_path or _store.DEFAULT_DB_PATH
    if not os.path.exists(path):
        print(
            f"[T09] --use-store given but no store at {path!r}; "
            "falling back to JSON-only numbers.",
            file=sys.stderr,
        )
        return None
    try:
        conn = _store.connect(path, create=False)
    except Exception as exc:
        print(
            f"[T09] could not open store at {path!r} ({exc}); JSON-only numbers used.",
            file=sys.stderr,
        )
        return None
    try:
        species_counts = _store.count_by(conn, "species", dataset=dataset)
        n_records = sum(species_counts.values())
        if n_records == 0:
            print(
                f"[T09] store has no dataset={dataset!r} rows; JSON-only numbers used.",
                file=sys.stderr,
            )
            return None
        return {
            "db_path": path,
            "n_records": n_records,
            "species_counts": species_counts,
        }
    except Exception as exc:
        print(f"[T09] store cross-check failed ({exc}); JSON-only numbers used.",
              file=sys.stderr)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def generate_medvednica_report(
    data_dir: str = "data/MedvednicaDS",
    out_dir: str = "Output/medvednica_report",
    *,
    top_k_species: int = 12,
    n_example_crops: int = 12,
    species_filter: Optional[List[str]] = None,
    use_store: bool = False,
    db_path: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """Read the Medvednica artifacts under ``data_dir`` and render the report.

    Computes the funnel + species + temporal summary, renders ``medvednica_report.md``,
    the three PNG figures, and ``medvednica_summary.json`` into ``out_dir``. Returns the
    summary dict (== contents of ``medvednica_summary.json``). Pure-read on inputs;
    never mutates ``data_dir`` or the store.
    """
    md_path = os.path.join(data_dir, MD_RESULTS_NAME)
    cleaned_path = os.path.join(data_dir, CLEANED_NAME)
    classified_path = os.path.join(data_dir, CLASSIFIED_NAME)
    csv_path = os.path.join(data_dir, CSV_NAME)
    crops_dir = os.path.join(data_dir, CROPS_DIRNAME)

    md_results = _load_json(md_path, MD_RESULTS_NAME)
    cleaned = _load_json(cleaned_path, CLEANED_NAME)
    classified = _load_json(classified_path, CLASSIFIED_NAME)

    funnel = compute_funnel(md_results, cleaned)
    species, chart_series = _build_species_section(
        classified, top_k_species=top_k_species
    )
    temporal = _compute_temporal(classified, csv_path)

    stem_index = _build_stem_species_index(classified)
    examples, used_filter_fallback = _select_example_crops(
        crops_dir,
        stem_index,
        n_example_crops=n_example_crops,
        seed=seed,
        species_filter=species_filter,
        data_dir=data_dir,
    )

    notes = ["All numbers computed from data/MedvednicaDS artifacts; no models re-run."]
    if not examples:
        notes.append(
            "animal_crops/ was empty or unreadable; the example montage is a "
            "placeholder."
        )
    if used_filter_fallback and species_filter:
        notes.append(
            "No crops matched --species-filter "
            f"{species_filter}; example montage falls back to all species."
        )

    # Optional, non-authoritative store cross-check.
    store_info = None
    if use_store:
        store_info = _store_crosscheck(db_path, "MedvednicaDS")
        if store_info is not None:
            notes.append(
                f"Store cross-check: {store_info['n_records']} MedvednicaDS rows at "
                f"{store_info['db_path']} (non-authoritative; JSON funnel is the "
                "source of truth)."
            )

    figures = {
        "funnel": os.path.join("figures", "detection_funnel.png"),
        "species": os.path.join("figures", "species_breakdown.png"),
        "examples": os.path.join("figures", "example_crops.png"),
    }

    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "MedvednicaDS",
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "data_dir": data_dir,
        "funnel": funnel,
        "species": species,
        "temporal": temporal,
        "examples": examples,
        "figures": figures,
        "report_md": "medvednica_report.md",
        "notes": notes,
    }
    if species_filter is not None:
        summary["target_species"] = _compute_target_species(classified, species_filter)
    if store_info is not None:
        summary["store_crosscheck"] = store_info

    # --- Write outputs ------------------------------------------------------ #
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    _render_funnel_figure(funnel, os.path.join(figures_dir, "detection_funnel.png"))
    _render_species_figure(
        chart_series, os.path.join(figures_dir, "species_breakdown.png")
    )
    _render_examples_montage(
        examples, os.path.join(figures_dir, "example_crops.png")
    )

    md_text = render_markdown(summary)
    with open(os.path.join(out_dir, "medvednica_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md_text)

    # Deterministic JSON: sorted keys so same inputs+seed => byte-identical file.
    with open(
        os.path.join(out_dir, "medvednica_summary.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    return summary


# --------------------------------------------------------------------------- #
# CLI / selftest
# --------------------------------------------------------------------------- #

def _selftest(data_dir: str = "data/MedvednicaDS") -> int:
    """Smoke test on the real data; asserts core numbers > 0. Returns process code."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        summary = generate_medvednica_report(data_dir, tmp, seed=0)
        f = summary["funnel"]
        sp = summary["species"]
        assert f["total_frames"] > 0, "total_frames must be > 0"
        assert f["frames_with_animal"] > 0, "frames_with_animal must be > 0"
        assert f["empty_frames"] == f["total_frames"] - f["frames_with_any_detection"]
        assert isinstance(f["person_or_vehicle_frames"], int)
        assert f["animal_detections_kept"] > 0, "kept detections must be > 0"
        assert sp["total_classified_detections"] > 0
        assert sp["real_species_detections"] > 0
        assert "blank" not in sp["counts"], "blank must be excluded from counts"
        for fn in (
            "medvednica_report.md",
            os.path.join("figures", "detection_funnel.png"),
            os.path.join("figures", "species_breakdown.png"),
            os.path.join("figures", "example_crops.png"),
            "medvednica_summary.json",
        ):
            p = os.path.join(tmp, fn)
            assert os.path.exists(p) and os.path.getsize(p) > 0, f"missing/empty: {p}"
    print("[T09] selftest OK")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reid_demo.medvednica_report",
        description="Render the Medvednica camera-trap filtering report (T09).",
    )
    parser.add_argument("--data-dir", default="data/MedvednicaDS")
    parser.add_argument("--out-dir", default="Output/medvednica_report")
    parser.add_argument("--top-k-species", type=int, default=12)
    parser.add_argument("--n-example-crops", type=int, default=12)
    parser.add_argument(
        "--species-filter",
        default=None,
        help="Comma-separated species names to focus the montage/callout on.",
    )
    parser.add_argument("--use-store", action="store_true")
    parser.add_argument("--db", default=None, help="Store DB path (with --use-store).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the smoke test on --data-dir and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.selftest:
        return _selftest(args.data_dir)

    species_filter: Optional[List[str]] = None
    if args.species_filter:
        species_filter = [s.strip() for s in args.species_filter.split(",") if s.strip()]

    summary = generate_medvednica_report(
        args.data_dir,
        args.out_dir,
        top_k_species=args.top_k_species,
        n_example_crops=args.n_example_crops,
        species_filter=species_filter,
        use_store=args.use_store,
        db_path=args.db,
        seed=args.seed,
    )
    f = summary["funnel"]
    print(
        f"[T09] wrote report to {args.out_dir} | "
        f"{f['total_frames']} photos in, {f['empty_frames']} empty, "
        f"{f['frames_with_animal']} with animals, "
        f"{summary['species']['n_distinct_species']} species."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
