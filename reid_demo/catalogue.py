"""reid_demo.catalogue — visual individual catalogue generator (T06).

READ-ONLY consumer of the T01 store. Given a populated store and a ``dataset``
name, render a **static, self-contained HTML catalogue directory** a non-technical
park biologist can browse offline by double-clicking ``index.html``:

* ``index.html`` — overview: plain-language headline counts + a grid of individual
  cards (best crop thumbnail + id + photo count + flank badge).
* ``individuals/individual_<cluster_id>.html`` — one contact sheet per discovered
  individual (distinct ``cluster_id >= 0``), all crops + per-crop metadata.
* ``unassigned.html`` — candidate-new singletons + DBSCAN noise
  (``is_candidate_new == 1`` / ``cluster_id == -1``, per D5).
* ``thumbs/`` — downsized crop thumbnails (PIL ``Image.thumbnail``) referenced by
  RELATIVE path, so the bundle is portable (zip it, move it, open it elsewhere).
* ``assets/style.css`` — local stylesheet (no CDN).
* ``catalogue_summary.json`` — machine-readable summary with STABLE keys (T09/T10).
* ``montages/individual_<id>.png`` — optional PNG contact sheets (``make_montages``).

No model loading, no clustering, no network — pure rendering of already-computed
results. All store access goes through :mod:`reid_demo.store`; only SELECTs are
issued (``connect``/``query_records``/``count_by``). Heavy/optional deps (matplotlib
montage path) are lazy-imported so this module imports cleanly without them.

Binding facts honored (DATA_CONTRACT + D1-D8):
* ``cluster_id >= 0``  => a multi-crop discovered individual.
* ``cluster_id == -1`` (with ``is_candidate_new == 1``) => singleton OR DBSCAN noise.
* ``cluster_id IS NULL`` => clustering has not run for that row; skip it.
* ``by_flank`` is computed over ``cluster_id >= 0`` rows ONLY; every crop counted
  once by its own ``orientation``; NULL/empty/non-canonical -> ``unknown``; all six
  canonical keys zero-filled; ``sum(by_flank.values()) == counts.crops_clustered``
  (D7c sum invariant).
* ``mixed_flank`` True iff a cluster holds both ``left`` and ``right``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from reid_demo.store import (
    DEFAULT_DB_PATH,
    DetectionRecord,
    connect,
    query_records,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Canonical orientation/flank buckets (DATA_CONTRACT D4). by_flank always
#: contains exactly these six keys, zero-filled. Anything else -> "unknown".
CANONICAL_FLANKS: List[str] = ["left", "right", "front", "back", "down", "unknown"]
_CANONICAL_FLANK_SET = set(CANONICAL_FLANKS)

#: Default cap on tiles drawn into a montage PNG (HTML may still show all).
DEFAULT_MONTAGE_CAP = 25

#: Output sub-directory names.
THUMBS_DIRNAME = "thumbs"
ASSETS_DIRNAME = "assets"
INDIVIDUALS_DIRNAME = "individuals"
MONTAGES_DIRNAME = "montages"
SUMMARY_FILENAME = "catalogue_summary.json"
INDEX_FILENAME = "index.html"
UNASSIGNED_FILENAME = "unassigned.html"
STYLE_FILENAME = "style.css"
PLACEHOLDER_FILENAME = "_placeholder.png"

NOISE_CLUSTER_ID = -1


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class CatalogueResult:
    """Return value of :func:`build_catalogue` — output paths + the summary dict."""

    out_dir: str
    index_html: str            # absolute path to index.html
    summary_json: str          # absolute path to catalogue_summary.json
    summary: dict              # same content as catalogue_summary.json (schema below)
    individual_pages: dict     # {cluster_id(int): absolute_html_path(str)}
    montage_pngs: dict = field(default_factory=dict)  # {cluster_id(int): abs_png_path}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm_flank(orientation: Optional[str]) -> str:
    """Map any orientation value to a canonical flank bucket (D7c).

    NULL / empty / non-canonical -> ``unknown``.
    """
    if orientation is None:
        return "unknown"
    o = str(orientation).strip().lower()
    if o in _CANONICAL_FLANK_SET:
        return o
    return "unknown"


def _is_low_conf(conf: Optional[float], threshold: float) -> bool:
    """A crop is low-confidence if cluster_conf is NULL or < threshold."""
    if conf is None:
        return True
    try:
        return float(conf) < threshold
    except (TypeError, ValueError):
        return True


def _safe_id_token(record_id: str) -> str:
    """Filesystem-safe token from a record_id for thumbnail filenames."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(record_id))


def _esc(value) -> str:
    """HTML-escape any value (None -> '')."""
    if value is None:
        return ""
    return html.escape(str(value))


def _rep_sort_key(rec: DetectionRecord):
    """Representative-crop ranking: highest cluster_conf, then detector_conf, then id.

    NULL confidences sort last (treated as -inf) so a record with a real conf wins;
    final fallback is the first crop by record_id.
    """
    cc = rec.cluster_conf if rec.cluster_conf is not None else float("-inf")
    dc = rec.detector_conf if rec.detector_conf is not None else float("-inf")
    return (-float(cc), -float(dc), str(rec.record_id))


# --------------------------------------------------------------------------- #
# Thumbnail / placeholder generation (PIL, lazy)
# --------------------------------------------------------------------------- #

def _load_pil():
    """Import Pillow lazily (it is a confirmed repo dependency)."""
    from PIL import Image  # noqa: WPS433 (local import on purpose)
    return Image


def _make_placeholder(thumbs_dir: Path, thumb_size: int) -> str:
    """Generate a gray placeholder PNG once; return its filename (in thumbs/)."""
    Image = _load_pil()
    path = thumbs_dir / PLACEHOLDER_FILENAME
    if not path.exists():
        side = max(32, min(int(thumb_size), 512))
        img = Image.new("RGB", (side, side), color=(64, 64, 70))
        img.save(path, format="PNG")
    return PLACEHOLDER_FILENAME


def _make_thumbnail(crop_path: Optional[str], thumbs_dir: Path, record_id: str,
                    thumb_size: int) -> Optional[str]:
    """Generate a downsized thumbnail for one crop into ``thumbs/``.

    Returns the thumbnail FILENAME (relative to thumbs/) on success, or ``None`` if
    the crop file is missing/unreadable (caller then uses the placeholder). Never
    raises on a bad image.
    """
    if not crop_path:
        return None
    src = Path(crop_path)
    if not src.is_file():
        return None
    Image = _load_pil()
    out_name = f"{_safe_id_token(record_id)}.jpg"
    out_path = thumbs_dir / out_name
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((int(thumb_size), int(thumb_size)), Image.LANCZOS)
            im.save(out_path, format="JPEG", quality=85)
    except Exception as exc:  # noqa: BLE001 — never abort the build for one bad image
        warnings.warn(f"could not read crop {crop_path!r} for {record_id!r}: {exc}")
        return None
    return out_name


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@dataclass
class _IndividualAgg:
    """Per-cluster aggregation built from the rows we loaded (no per-cluster SQL)."""

    cluster_id: int
    records: List[DetectionRecord] = field(default_factory=list)

    @property
    def n_crops(self) -> int:
        return len(self.records)

    def flank_set(self) -> List[str]:
        seen = []
        for r in self.records:
            f = _norm_flank(r.orientation)
            if f not in seen:
                seen.append(f)
        # stable canonical ordering
        return [f for f in CANONICAL_FLANKS if f in seen]

    @property
    def mixed_flank(self) -> bool:
        raw = {_norm_flank(r.orientation) for r in self.records}
        return "left" in raw and "right" in raw

    def species_list(self) -> List[str]:
        out = []
        for r in self.records:
            if r.species and r.species not in out:
                out.append(r.species)
        return out

    def camera_list(self) -> List[str]:
        out = []
        for r in self.records:
            if r.camera_id and r.camera_id not in out:
                out.append(r.camera_id)
        return out

    def timestamps(self) -> List[str]:
        return sorted(r.timestamp for r in self.records if r.timestamp)

    def mean_cluster_conf(self) -> Optional[float]:
        vals = [float(r.cluster_conf) for r in self.records if r.cluster_conf is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    def n_low_conf(self, threshold: float) -> int:
        return sum(1 for r in self.records if _is_low_conf(r.cluster_conf, threshold))

    def n_confirmed(self) -> int:
        return sum(1 for r in self.records if r.review_status == "confirmed")

    def representative(self) -> DetectionRecord:
        return sorted(self.records, key=_rep_sort_key)[0]


def _group_individuals(records: Sequence[DetectionRecord]) -> Dict[int, _IndividualAgg]:
    """Group rows with ``cluster_id >= 0`` by cluster_id (the individuals)."""
    groups: Dict[int, _IndividualAgg] = {}
    for r in records:
        cid = r.cluster_id
        if cid is None or cid < 0:
            continue
        groups.setdefault(cid, _IndividualAgg(cluster_id=cid)).records.append(r)
    return groups


def _candidate_new(records: Sequence[DetectionRecord]) -> List[DetectionRecord]:
    """Candidate-new / unassigned rows: ``is_candidate_new == 1`` (de-duped by id)."""
    seen = set()
    out: List[DetectionRecord] = []
    for r in records:
        if r.is_candidate_new == 1 and r.record_id not in seen:
            seen.add(r.record_id)
            out.append(r)
    return out


def _compute_by_flank(records: Sequence[DetectionRecord]) -> Dict[str, int]:
    """``by_flank`` over ``cluster_id >= 0`` rows only (D7c).

    Each clustered crop counted once by its own orientation; NULL/empty/non-canonical
    -> ``unknown``; all six canonical keys zero-filled. Invariant enforced by caller:
    ``sum(by_flank.values()) == crops_clustered``.
    """
    counts = {k: 0 for k in CANONICAL_FLANKS}
    for r in records:
        cid = r.cluster_id
        if cid is None or cid < 0:
            continue
        counts[_norm_flank(r.orientation)] += 1
    return counts


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def _build_summary(
    records: Sequence[DetectionRecord],
    groups: Dict[int, _IndividualAgg],
    candidate_new: Sequence[DetectionRecord],
    *,
    dataset: Optional[str],
    species_filter: Optional[str],
    low_conf_threshold: float,
    rep_thumb_rel: Dict[int, str],
) -> dict:
    """Assemble the catalogue_summary.json dict (stable keys)."""
    total_crops = len(records)
    crops_clustered = sum(
        1 for r in records if r.cluster_id is not None and r.cluster_id >= 0
    )
    n_individuals = len(groups)
    n_candidate_new = sum(1 for r in records if r.is_candidate_new == 1)
    n_unassigned_noise = sum(
        1 for r in records if r.cluster_id is not None and r.cluster_id == NOISE_CLUSTER_ID
    )
    n_low_conf = sum(1 for r in records if _is_low_conf(r.cluster_conf, low_conf_threshold))
    n_confirmed = sum(1 for r in records if r.review_status == "confirmed")
    n_rejected = sum(1 for r in records if r.review_status == "rejected")

    by_flank = _compute_by_flank(records)
    # D7c sum invariant (assert-grade; build is read-only so this is purely derived).
    assert sum(by_flank.values()) == crops_clustered, (by_flank, crops_clustered)

    # individuals sorted by n_crops desc, then cluster_id asc
    ordered = sorted(groups.values(), key=lambda g: (-g.n_crops, g.cluster_id))

    individuals = []
    for g in ordered:
        ts = g.timestamps()
        individuals.append({
            "cluster_id": int(g.cluster_id),
            "n_crops": int(g.n_crops),
            "flanks": g.flank_set(),
            "mixed_flank": bool(g.mixed_flank),
            "species": g.species_list(),
            "cameras": g.camera_list(),
            "first_seen": ts[0] if ts else None,
            "last_seen": ts[-1] if ts else None,
            "mean_cluster_conf": g.mean_cluster_conf(),
            "n_low_conf": int(g.n_low_conf(low_conf_threshold)),
            "n_confirmed": int(g.n_confirmed()),
            "representative_crop": rep_thumb_rel.get(g.cluster_id),
            "page": f"{INDIVIDUALS_DIRNAME}/individual_{g.cluster_id}.html",
        })

    headline = (
        f"Found {n_individuals} "
        f"{'individual' if n_individuals == 1 else 'individuals'} across "
        f"{crops_clustered} {'photo' if crops_clustered == 1 else 'photos'} "
        f"({n_candidate_new} possible new, {n_unassigned_noise} unassigned)."
    )

    return {
        "dataset": dataset,
        "species_filter": species_filter,
        "generated_at": _now_iso(),
        "low_conf_threshold": low_conf_threshold,
        "counts": {
            "total_crops": total_crops,
            "crops_clustered": crops_clustered,
            "individuals": n_individuals,
            "candidate_new": n_candidate_new,
            "unassigned_noise": n_unassigned_noise,
            "low_confidence_crops": n_low_conf,
            "reviewed_confirmed": n_confirmed,
            "reviewed_rejected": n_rejected,
        },
        "headline": headline,
        "by_flank": by_flank,
        "individuals": individuals,
    }


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

_STYLE_CSS = """\
/* reid_demo catalogue — self-contained, no CDN. */
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
    Arial, sans-serif;
  margin: 0; padding: 0 0 4rem 0; color: #1c2128; background: #f6f8fa;
}
header { background: #11334d; color: #fff; padding: 1.5rem 2rem; }
header h1 { margin: 0 0 .3rem 0; font-size: 1.6rem; }
.headline { font-size: 1.15rem; margin: .25rem 0 0 0; }
.subtle { color: #9fb3c8; font-size: .85rem; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem; }
a { color: #11578c; text-decoration: none; }
a:hover { text-decoration: underline; }
.stats { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0 2rem 0; }
.stat {
  background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
  padding: .8rem 1.1rem; min-width: 9rem;
}
.stat .n { font-size: 1.7rem; font-weight: 700; }
.stat .l { color: #57606a; font-size: .8rem; text-transform: uppercase;
  letter-spacing: .03em; }
.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 1rem;
}
.card {
  background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
  overflow: hidden; display: flex; flex-direction: column;
}
.card img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #eaeef2; }
.card .body { padding: .6rem .7rem; }
.card .id { font-weight: 700; }
.card .meta { color: #57606a; font-size: .82rem; margin-top: .2rem; }
.badges { margin-top: .4rem; display: flex; flex-wrap: wrap; gap: .3rem; }
.badge {
  font-size: .72rem; padding: .12rem .45rem; border-radius: 10px;
  background: #eaeef2; color: #24292f; border: 1px solid #d0d7de;
}
.badge.left { background: #dbeafe; color: #1e40af; border-color: #bfdbfe; }
.badge.right { background: #fde68a; color: #92400e; border-color: #fcd34d; }
.badge.unknown { background: #eaeef2; color: #57606a; }
.badge.mixed { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
.badge.review { background: #fef3c7; color: #92400e; border-color: #fde68a; }
.badge.confirmed { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
.badge.rejected { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
.tiles {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem; margin-top: 1rem;
}
.tile {
  background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
  overflow: hidden; position: relative;
}
.tile.low_conf { border-color: #f59e0b; box-shadow: 0 0 0 2px #fde68a inset; }
.tile img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #eaeef2; }
.tile .cap { padding: .45rem .55rem; font-size: .76rem; color: #57606a; }
.tile .cap .rid { color: #24292f; font-weight: 600; word-break: break-all; }
.flag {
  position: absolute; top: .4rem; right: .4rem; font-size: .68rem;
  padding: .1rem .4rem; border-radius: 8px; font-weight: 600;
}
.flag.review { background: #f59e0b; color: #fff; }
.flag.confirmed { background: #16a34a; color: #fff; }
.flag.rejected { background: #dc2626; color: #fff; }
.note {
  background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412;
  padding: .6rem .8rem; border-radius: 8px; margin: 1rem 0;
}
.back { display: inline-block; margin-bottom: 1rem; }
footer { text-align: center; color: #8b949e; font-size: .78rem; padding: 2rem; }
table.kv { border-collapse: collapse; margin: .5rem 0; }
table.kv td { padding: .15rem .8rem .15rem 0; font-size: .85rem; }
table.kv td.k { color: #57606a; }
"""


def _flank_badges(flanks: Sequence[str], mixed: bool) -> str:
    parts = []
    for f in flanks:
        cls = f if f in ("left", "right", "unknown") else "unknown"
        parts.append(f'<span class="badge {cls}">{_esc(f)}</span>')
    if mixed:
        parts.append('<span class="badge mixed">mixed flank</span>')
    return "".join(parts)


def _page_shell(title: str, heading: str, headline: str, body: str,
                css_href: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="{_esc(css_href)}">
</head>
<body>
<header>
  <h1>{_esc(heading)}</h1>
  <p class="headline">{_esc(headline)}</p>
</header>
<main>
{body}
</main>
<footer>Generated by reid_demo.catalogue (T06) · static offline catalogue</footer>
</body>
</html>
"""


def _render_index(summary: dict, title: str) -> str:
    c = summary["counts"]
    stats = [
        (c["individuals"], "individuals"),
        (c["crops_clustered"], "photos clustered"),
        (c["candidate_new"], "possible new"),
        (c["unassigned_noise"], "unassigned"),
        (c["low_confidence_crops"], "need review"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="n">{n}</div>'
        f'<div class="l">{_esc(label)}</div></div>'
        for n, label in stats
    )

    cards = []
    for ind in summary["individuals"]:
        cid = ind["cluster_id"]
        rep = ind.get("representative_crop") or f"{THUMBS_DIRNAME}/{PLACEHOLDER_FILENAME}"
        flanks = ind.get("flanks") or ["unknown"]
        badges = _flank_badges(flanks, ind.get("mixed_flank", False))
        n = ind["n_crops"]
        mean_conf = ind.get("mean_cluster_conf")
        conf_str = f"{mean_conf:.2f}" if mean_conf is not None else "n/a"
        species = ", ".join(ind.get("species") or []) or "unknown species"
        n_low = ind.get("n_low_conf", 0)
        low_badge = (
            f'<span class="badge review">{n_low} need review</span>' if n_low else ""
        )
        n_conf = ind.get("n_confirmed", 0)
        conf_badge = (
            f'<span class="badge confirmed">{n_conf} confirmed</span>' if n_conf else ""
        )
        cards.append(f"""
<a class="card" href="{_esc(ind['page'])}">
  <img src="{_esc(rep)}" alt="Individual {cid}" loading="lazy">
  <div class="body">
    <div class="id">Individual {cid}</div>
    <div class="meta">{n} {'photo' if n == 1 else 'photos'} · {_esc(species)}</div>
    <div class="meta subtle">confidence {conf_str}</div>
    <div class="badges">{badges}{low_badge}{conf_badge}</div>
  </div>
</a>""")

    grid = (
        '<div class="grid">' + "".join(cards) + "</div>"
        if cards else
        '<p class="note">No discovered individuals (no clusters with cluster_id &gt;= 0).</p>'
    )

    unassigned_link = (
        f'<p><a href="{UNASSIGNED_FILENAME}">'
        f'&#9888; {c["candidate_new"]} possible new / unassigned crops '
        f'&rarr; review</a></p>'
        if c["candidate_new"] or c["unassigned_noise"] else ""
    )

    body = f"""
<div class="stats">{stat_html}</div>
{unassigned_link}
<h2>Discovered individuals</h2>
{grid}
"""
    css_href = f"{ASSETS_DIRNAME}/{STYLE_FILENAME}"
    return _page_shell(title, title, summary["headline"], body, css_href)


def _tile_html(rec: DetectionRecord, thumb_rel: str, threshold: float) -> str:
    low = _is_low_conf(rec.cluster_conf, threshold)
    tile_cls = "tile low_conf" if low else "tile"

    flag = ""
    if rec.review_status == "confirmed":
        flag = '<span class="flag confirmed">&#10003; confirmed</span>'
    elif rec.review_status == "rejected":
        flag = '<span class="flag rejected">&#10007; rejected</span>'
    elif low:
        flag = '<span class="flag review">review</span>'

    conf = rec.cluster_conf
    conf_str = f"{float(conf):.2f}" if conf is not None else "n/a"
    flank = _norm_flank(rec.orientation)
    cap = f"""<div class="cap">
  <div class="rid">{_esc(rec.record_id)}</div>
  <div>flank: {_esc(flank)} · conf: {conf_str}</div>
  <div>{_esc(rec.species or 'unknown species')}</div>
  <div>{_esc(rec.camera_id or 'unknown camera')} · {_esc(rec.timestamp or '')}</div>
  <div>review: {_esc(rec.review_status)}</div>
</div>"""
    return f"""<div class="{tile_cls}">
  {flag}
  <img src="{_esc(thumb_rel)}" alt="{_esc(rec.record_id)}" loading="lazy">
  {cap}
</div>"""


def _render_individual_page(
    agg: _IndividualAgg,
    ind_summary: dict,
    thumb_rel: Dict[str, str],
    *,
    title: str,
    low_conf_threshold: float,
    max_crops: Optional[int],
) -> str:
    cid = agg.cluster_id
    # tiles ordered by representative ranking so best crops lead
    recs = sorted(agg.records, key=_rep_sort_key)
    if max_crops is not None:
        recs = recs[: max_crops]

    flanks = ind_summary.get("flanks") or ["unknown"]
    badges = _flank_badges(flanks, ind_summary.get("mixed_flank", False))
    mixed_note = (
        '<p class="note">&#9888; Mixed flank: this cluster contains both '
        '<b>left</b> and <b>right</b> crops. Lynx left/right flanks are different '
        'patterns and normally cluster separately — worth a human check.</p>'
        if ind_summary.get("mixed_flank") else ""
    )

    mean_conf = ind_summary.get("mean_cluster_conf")
    conf_str = f"{mean_conf:.2f}" if mean_conf is not None else "n/a"
    kv = f"""<table class="kv">
  <tr><td class="k">Photos</td><td>{ind_summary['n_crops']}</td></tr>
  <tr><td class="k">Species</td><td>{_esc(', '.join(ind_summary.get('species') or []) or 'unknown')}</td></tr>
  <tr><td class="k">Cameras</td><td>{_esc(', '.join(ind_summary.get('cameras') or []) or 'unknown')}</td></tr>
  <tr><td class="k">First seen</td><td>{_esc(ind_summary.get('first_seen') or 'n/a')}</td></tr>
  <tr><td class="k">Last seen</td><td>{_esc(ind_summary.get('last_seen') or 'n/a')}</td></tr>
  <tr><td class="k">Mean confidence</td><td>{conf_str}</td></tr>
  <tr><td class="k">Need review</td><td>{ind_summary.get('n_low_conf', 0)}</td></tr>
  <tr><td class="k">Confirmed</td><td>{ind_summary.get('n_confirmed', 0)}</td></tr>
</table>"""

    tiles = "".join(
        _tile_html(
            r,
            f"../{thumb_rel.get(r.record_id, f'{THUMBS_DIRNAME}/{PLACEHOLDER_FILENAME}')}",
            low_conf_threshold,
        )
        for r in recs
    )
    capped_note = (
        f'<p class="subtle">Showing first {len(recs)} of {agg.n_crops} crops.</p>'
        if max_crops is not None and agg.n_crops > len(recs) else ""
    )

    body = f"""
<a class="back" href="../{INDEX_FILENAME}">&larr; Back to catalogue</a>
<h2>Individual {cid} <span class="badges">{badges}</span></h2>
{mixed_note}
{kv}
{capped_note}
<div class="tiles">{tiles}</div>
"""
    css_href = f"../{ASSETS_DIRNAME}/{STYLE_FILENAME}"
    heading = f"{title} — Individual {cid}"
    headline = f"{agg.n_crops} {'photo' if agg.n_crops == 1 else 'photos'} of individual {cid}"
    return _page_shell(heading, heading, headline, body, css_href)


def _render_unassigned_page(
    candidate_records: Sequence[DetectionRecord],
    thumb_rel: Dict[str, str],
    *,
    title: str,
    low_conf_threshold: float,
) -> str:
    recs = sorted(candidate_records, key=_rep_sort_key)
    tiles = "".join(
        _tile_html(
            r,
            thumb_rel.get(r.record_id, f"{THUMBS_DIRNAME}/{PLACEHOLDER_FILENAME}"),
            low_conf_threshold,
        )
        for r in recs
    )
    intro = (
        '<p class="note">These crops were not confidently matched to a discovered '
        'individual. Each may be a brand-new animal the system has not seen before, '
        'or a hard-to-match photo. They are flagged for human attention.</p>'
        if recs else
        '<p>No candidate-new or unassigned crops.</p>'
    )
    body = f"""
<a class="back" href="{INDEX_FILENAME}">&larr; Back to catalogue</a>
<h2>Possible new individuals &amp; unassigned</h2>
{intro}
<div class="tiles">{tiles}</div>
"""
    css_href = f"{ASSETS_DIRNAME}/{STYLE_FILENAME}"
    heading = f"{title} — Possible new / unassigned"
    headline = (
        f"{len(recs)} crop{'s' if len(recs) != 1 else ''} flagged as possible new "
        f"or unassigned"
    )
    return _page_shell(heading, heading, headline, body, css_href)


# --------------------------------------------------------------------------- #
# Montages (optional, matplotlib — lazy, fail-soft)
# --------------------------------------------------------------------------- #

def _build_montages(
    groups: Dict[int, _IndividualAgg],
    montages_dir: Path,
    *,
    cap: int,
) -> Dict[int, str]:
    """Render one PNG contact sheet per individual via visualization_suite.collage.

    Lazy-imports matplotlib/cv2/collage; any failure downgrades to HTML-only with a
    warning instead of crashing the build.
    """
    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        from visualization_suite.collage import make_grid  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"montages disabled (import failed): {exc}")
        return {}

    montages_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[int, str] = {}
    for cid, agg in groups.items():
        recs = sorted(agg.records, key=_rep_sort_key)[: max(1, cap)]
        images: List["np.ndarray"] = []
        titles: List[str] = []
        for r in recs:
            img = None
            if r.crop_path and Path(r.crop_path).is_file():
                try:
                    img = cv2.imread(r.crop_path, cv2.IMREAD_COLOR)
                except Exception:  # noqa: BLE001
                    img = None
            if img is None:
                img = np.full((128, 128, 3), 70, dtype=np.uint8)  # gray placeholder (BGR)
            images.append(img)
            titles.append(str(r.record_id))
        try:
            cols = min(5, max(1, len(images)))
            grid_img, _ = make_grid(images, titles=titles, cols=cols)
            png_path = montages_dir / f"individual_{cid}.png"
            cv2.imwrite(str(png_path), grid_img)
            out[cid] = str(png_path.resolve())
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"montage for cluster {cid} failed: {exc}")
            continue
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def build_catalogue(
    db_path: str = DEFAULT_DB_PATH,
    *,
    dataset: Optional[str] = None,
    out_dir: Optional[str] = None,
    species: Optional[str] = None,
    low_conf_threshold: float = 0.5,
    thumb_size: int = 256,
    max_crops_per_individual: Optional[int] = None,
    make_montages: bool = False,
    title: str = "Individual Catalogue",
) -> CatalogueResult:
    """Render a static HTML catalogue directory from the T01 store (read-only).

    Reads clustered records, groups by ``cluster_id``, writes ``index.html`` +
    per-individual pages + ``unassigned.html`` + ``thumbs/`` + ``assets/style.css``
    + ``catalogue_summary.json`` (+ optional ``montages/*.png``). Never raises on a
    missing crop file (renders a placeholder); raises only on an unreadable store or
    an empty result set after filtering.

    Parameters mirror the Interface contract in tickets/T06; see the module docstring.
    """
    # ----- read (SELECT-only) -----
    conn = connect(db_path, create=False)
    try:
        records = query_records(
            conn, dataset=dataset, species=species, order_by="record_id"
        )
    finally:
        conn.close()

    if not records:
        raise ValueError(
            "build_catalogue: empty result set after filtering "
            f"(dataset={dataset!r}, species={species!r}). Nothing to render."
        )

    # ----- output dir layout -----
    if out_dir is None:
        out_dir = os.path.join(
            "data", "reid_demo", "catalogue", dataset if dataset else "all"
        )
    out_root = Path(out_dir).resolve()
    thumbs_dir = out_root / THUMBS_DIRNAME
    assets_dir = out_root / ASSETS_DIRNAME
    indiv_dir = out_root / INDIVIDUALS_DIRNAME
    for d in (out_root, thumbs_dir, assets_dir, indiv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ----- group -----
    groups = _group_individuals(records)
    candidate_records = _candidate_new(records)

    # ----- thumbnails (one per record) + placeholder -----
    placeholder_name = _make_placeholder(thumbs_dir, thumb_size)
    placeholder_rel = f"{THUMBS_DIRNAME}/{placeholder_name}"
    thumb_rel: Dict[str, str] = {}   # record_id -> "thumbs/<file>" (relative to out_root)
    missing_crops = 0
    for r in records:
        name = _make_thumbnail(r.crop_path, thumbs_dir, r.record_id, thumb_size)
        if name is None:
            missing_crops += 1
            thumb_rel[r.record_id] = placeholder_rel
        else:
            thumb_rel[r.record_id] = f"{THUMBS_DIRNAME}/{name}"
    if missing_crops:
        warnings.warn(
            f"{missing_crops} crop file(s) missing/unreadable; placeholder tiles used."
        )

    # representative thumbnail per cluster (relative to out_root)
    rep_thumb_rel: Dict[int, str] = {}
    for cid, agg in groups.items():
        rep = agg.representative()
        rep_thumb_rel[cid] = thumb_rel.get(rep.record_id, placeholder_rel)

    # ----- summary -----
    summary = _build_summary(
        records, groups, candidate_records,
        dataset=dataset, species_filter=species,
        low_conf_threshold=low_conf_threshold, rep_thumb_rel=rep_thumb_rel,
    )
    summary["counts"]["missing_crops"] = missing_crops  # extra (non-breaking) diagnostic

    # ----- write assets/style.css -----
    (assets_dir / STYLE_FILENAME).write_text(_STYLE_CSS, encoding="utf-8")

    # ----- write index.html -----
    index_path = out_root / INDEX_FILENAME
    index_path.write_text(_render_index(summary, title), encoding="utf-8")

    # ----- per-individual pages -----
    ind_summary_by_cid = {i["cluster_id"]: i for i in summary["individuals"]}
    individual_pages: Dict[int, str] = {}
    for cid, agg in groups.items():
        page_html = _render_individual_page(
            agg, ind_summary_by_cid[cid], thumb_rel,
            title=title, low_conf_threshold=low_conf_threshold,
            max_crops=max_crops_per_individual,
        )
        page_path = indiv_dir / f"individual_{cid}.html"
        page_path.write_text(page_html, encoding="utf-8")
        individual_pages[cid] = str(page_path.resolve())

    # ----- unassigned.html -----
    unassigned_path = out_root / UNASSIGNED_FILENAME
    unassigned_path.write_text(
        _render_unassigned_page(
            candidate_records, thumb_rel,
            title=title, low_conf_threshold=low_conf_threshold,
        ),
        encoding="utf-8",
    )

    # ----- catalogue_summary.json -----
    summary_path = out_root / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ----- optional montages -----
    montage_pngs: Dict[int, str] = {}
    if make_montages:
        cap = max_crops_per_individual if max_crops_per_individual else DEFAULT_MONTAGE_CAP
        montage_pngs = _build_montages(groups, out_root / MONTAGES_DIRNAME, cap=cap)

    return CatalogueResult(
        out_dir=str(out_root),
        index_html=str(index_path.resolve()),
        summary_json=str(summary_path.resolve()),
        summary=summary,
        individual_pages=individual_pages,
        montage_pngs=montage_pngs,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _seed_selftest_store(db_path: str) -> str:
    """Seed a tiny throwaway store: 3 individuals (cluster_id 0/1/2) + 1 candidate-new
    singleton (cluster_id == -1 / is_candidate_new == 1, per D5). Returns dataset name.
    """
    from reid_demo.store import make_record_id, upsert_records  # local import

    for p in (db_path, db_path + "-wal", db_path + "-shm"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    ds = "SelftestDS"
    conn = connect(db_path)

    def mk(stem, idx, cid, conf, flank, cand=0):
        return DetectionRecord(
            record_id=make_record_id(stem, idx),
            source_image=f"data/{ds}/images/{stem}.JPG",
            source_stem=stem, det_index=idx,
            crop_path=f"/tmp/_reid_cat_selftest_missing/{stem}__crop{idx}.jpg",
            bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
            detector_conf=0.9, camera_id="unknown_camera",
            timestamp="2025-06-02 04:27:51", species="eurasian lynx",
            species_conf=0.95, cluster_id=cid, cluster_conf=conf,
            is_candidate_new=cand, orientation=flank, dataset=ds,
        )

    recs = []
    # individual 0: 3 left crops
    for i in range(3):
        recs.append(mk(f"A{i}", 1, 0, 0.9, "left"))
    # individual 1: 2 right crops (low-conf cluster)
    for i in range(2):
        recs.append(mk(f"B{i}", 1, 1, 0.4, "right"))
    # individual 2: mixed flank (left + right) + one empty-orientation crop -> unknown
    recs.append(mk("Cmix", 1, 2, 0.8, "left"))
    recs.append(mk("Cmix2", 1, 2, 0.8, "right"))
    recs.append(mk("Cnull", 1, 2, 0.8, ""))  # '' normalizes to 'unknown' at ingest
    # 1 candidate-new singleton (noise id, cluster_id == -1 per D5)
    recs.append(mk("S", 1, -1, 0.3, "unknown", cand=1))

    upsert_records(conn, recs)
    conn.close()
    return ds


def _selftest(db_path: str) -> bool:
    """Seed a throwaway store, build the catalogue into a temp dir, assert invariants."""
    import tempfile

    ds = _seed_selftest_store(db_path)
    out_dir = tempfile.mkdtemp(prefix="reid_cat_selftest_")
    res = build_catalogue(db_path, dataset=ds, out_dir=out_dir, low_conf_threshold=0.5)
    s = res.summary
    c = s["counts"]

    assert os.path.exists(res.index_html), res.index_html
    assert os.path.exists(res.summary_json), res.summary_json
    assert os.path.exists(os.path.join(out_dir, ASSETS_DIRNAME, STYLE_FILENAME))
    assert c["individuals"] == 3, c
    assert c["candidate_new"] == 1, c
    assert c["unassigned_noise"] == 1, c
    assert c["crops_clustered"] == 8, c  # 3 + 2 + 3 (cluster 2 has 3 crops)
    assert c["total_crops"] == 9, c

    bf = s["by_flank"]
    assert set(bf) == set(CANONICAL_FLANKS), bf
    assert sum(bf.values()) == c["crops_clustered"], (bf, c)
    assert bf["unknown"] == 1, bf  # the empty-orientation clustered crop; noise excluded

    # individuals sorted by n_crops desc then cluster_id asc
    ns = [i["n_crops"] for i in s["individuals"]]
    assert ns == sorted(ns, reverse=True), ns

    # mixed flank on cluster 2
    ind2 = next(i for i in s["individuals"] if i["cluster_id"] == 2)
    assert ind2["mixed_flank"] is True, ind2

    # relative paths only
    html_text = Path(res.index_html).read_text(encoding="utf-8")
    assert "file://" not in html_text
    assert 'src="/' not in html_text and 'href="/' not in html_text

    print("[selftest] OK")
    print("HEADLINE:", s["headline"])
    print("out_dir:", res.out_dir)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reid_demo.catalogue",
        description="Static visual individual catalogue generator (T06, read-only).",
    )
    parser.add_argument("--selftest", action="store_true",
                        help="seed a throwaway store, build, assert counts; exit 0 ok")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="store DB path")
    parser.add_argument("--dataset", default=None, help="filter to one dataset/run")
    parser.add_argument("--out", default=None, help="output catalogue directory")
    parser.add_argument("--species", default=None, help="optional species filter")
    parser.add_argument("--low-conf", type=float, default=0.5,
                        help="cluster_conf below this is flagged 'review'")
    parser.add_argument("--thumb-size", type=int, default=256,
                        help="longest-edge px for thumbnails")
    parser.add_argument("--max-crops", type=int, default=None,
                        help="cap tiles per contact sheet")
    parser.add_argument("--montages", action="store_true",
                        help="also render montages/*.png via visualization_suite")
    parser.add_argument("--title", default="Individual Catalogue",
                        help="page H1 / report title")
    args = parser.parse_args(argv)

    if args.selftest:
        db = args.db if args.db != DEFAULT_DB_PATH else "/tmp/reid_cat_selftest.sqlite"
        try:
            ok = _selftest(db)
        except AssertionError as exc:
            print(f"[selftest] FAILED: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[selftest] ERROR: {exc}", file=sys.stderr)
            return 1
        return 0 if ok else 1

    try:
        res = build_catalogue(
            args.db,
            dataset=args.dataset,
            out_dir=args.out,
            species=args.species,
            low_conf_threshold=args.low_conf,
            thumb_size=args.thumb_size,
            max_crops_per_individual=args.max_crops,
            make_montages=args.montages,
            title=args.title,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"build_catalogue failed: {exc}", file=sys.stderr)
        return 1

    print(res.index_html)
    print(res.summary["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
