"""T10 — Demo assembly & end-to-end runner (the conductor).

One command runs the open-set re-ID pipeline on a configured dataset and assembles a
shareable, static demo bundle (a folder + ``index.html`` + ``SUMMARY.md`` + ``manifest.json``).

T10 contains NO new detection / embedding / clustering / catalogue / eval / report logic.
It only *orders* the T02–T12 stages, plumbs the shared ``--db``/dataset/out args, captures
each stage's result, and gathers the outputs into one bundle. Every heavy operation is a
call into the T02–T12 modules (subprocess CLI or in-process function); the single source of
wiring truth is :data:`STAGE_REGISTRY`.

Two honest tracks (binding decision; see STATUS_BOARD.md):

* **B-track** (LeopardID2022 default, ATRW optional) — the individual-ID proof:
  ``ingest -> species -> embed -> [fisher -> fusion] -> cluster -> {catalogue, eval}``.
  The ``species`` stage is NOT skipped: it runs the cheap, model-free
  ``T03.set_known_species(species="leopard"|"tiger")`` so every row carries a uniform
  ``species`` column. ``fisher`` (T11) and ``fusion`` (T12) run ONLY when ``--signals`` is
  ``global+fisher`` or ``full-funnel``.
* **A-track** (MedvednicaDS) — the real-field-data filtering report: ``report`` only (T09).

Signal layering (D8, ``--signals``): ``global`` (default; T04 embeddings only — the
standalone backbone), ``global+fisher`` (T11 Fisher fused into the calibrated affinity),
``full-funnel`` (the fused affinity PLUS T12 GV reranking on borderline pairs). For the
non-``global`` sets T10 also re-clusters/evaluates on the global-only backbone so the bundle
can report the **lift** ("global finds N individuals; with geometric verification, M").

Importable contract used by downstream/tests:
``build_config``, ``plan_stages``, ``run_demo``, ``assemble_bundle``, ``main`` and the
``DemoConfig``/``StageSpec``/``StageResult``/``BundleResult`` dataclasses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import store

# --------------------------------------------------------------------------- #
# Constants / tracks / defaults
# --------------------------------------------------------------------------- #

MANIFEST_SCHEMA_VERSION: int = 1

DEFAULT_DB_PATH: str = store.DEFAULT_DB_PATH                # data/reid_demo/reid_demo.sqlite
DEFAULT_OUT_DIR: str = "demo_bundle"
DEFAULT_DATASETS: List[str] = ["LeopardID2022", "MedvednicaDS"]
DEFAULT_PRIMARY: str = "LeopardID2022"
SMOKE_MAX_IDENTITIES: int = 8

# Signal sets (D8). ``global`` is the default — the standalone backbone — so the default
# dry-run plan is exactly ingest->species->embed->cluster->catalogue->eval (no T11/T12).
SIGNAL_GLOBAL = "global"
SIGNAL_GLOBAL_FISHER = "global+fisher"
SIGNAL_FULL_FUNNEL = "full-funnel"
SIGNAL_CHOICES = (SIGNAL_GLOBAL, SIGNAL_GLOBAL_FISHER, SIGNAL_FULL_FUNNEL)
DEFAULT_SIGNALS = SIGNAL_GLOBAL

# Which track a dataset belongs to (extensible).
DATASET_TRACKS: Dict[str, str] = {
    "LeopardID2022": "B",
    "ATRW": "B",
    "MedvednicaDS": "A",
}
# Per-B-track known species, stamped model-free by T03.set_known_species.
B_TRACK_SPECIES: Dict[str, str] = {
    "LeopardID2022": "leopard",
    "ATRW": "tiger",
}

EVAL_OUT_DIR: str = "evaluations/clustering"               # T07's single-file output dir
CATALOGUE_STAGING: str = os.path.join("data", "reid_demo", "catalogue")
FUSION_STAGING: str = os.path.join("data", "reid_demo", "fusion")
REPORT_STAGING: str = os.path.join("data", "reid_demo", "medvednica_report")

# Where each copied stage artifact lands inside the bundle, and how to copy it.
#   "dir"  -> copy the directory that contains the artifact (catalogue / report)
#   "file" -> copy the single file (eval json)
BUNDLE_LAYOUT: Dict[str, tuple] = {
    "catalogue": ("catalogue", "dir"),
    "eval": ("accuracy", "file"),
    "report": ("medvednica_report", "dir"),
}


def _eval_tag(signals: str) -> str:
    """Filename-safe tag for the funnel eval (baseline eval is always tagged ``global``)."""
    return signals.replace("+", "_").replace("-", "_")


# --------------------------------------------------------------------------- #
# Dataclasses (interface contract)
# --------------------------------------------------------------------------- #

@dataclass
class StageSpec:
    """A single stage's resolved invocation. Exactly one of ``cli``/``func`` is set."""

    name: str                      # ingest|species|embed|fisher|fusion|cluster|catalogue|eval|report
    ticket: str                    # "T02".."T12"
    critical: bool                 # if True, failure aborts the run (unless --continue-on-error)
    output_paths: List[str]        # declared artifacts; used for skip-if-exists
    cli: Optional[List[str]] = None        # argv WITHOUT the python exe, run as subprocess
    func: Optional[str] = None             # "module:function" dotted path, called in-process
    kwargs: dict = field(default_factory=dict)
    # additive (non-contract) wiring helpers:
    dataset: Optional[str] = None
    label: Optional[str] = None            # disambiguates duplicate stage names in logs
    skip_store_column: Optional[str] = None  # store column whose non-NULL presence = "done"

    @property
    def display_label(self) -> str:
        return self.label or self.name


@dataclass
class DemoConfig:
    datasets: List[str]
    primary_dataset: str
    db_path: str
    out_dir: str
    run_name: str
    stages: Optional[List[str]] = None     # None => all applicable
    smoke: bool = False
    max_identities: Optional[int] = None
    force: bool = False
    continue_on_error: bool = False
    signals: str = DEFAULT_SIGNALS         # additive (D8): global | global+fisher | full-funnel


@dataclass
class StageResult:
    name: str
    ticket: str
    status: str                    # "ok" | "skipped" | "failed"
    seconds: float
    output_paths: List[str]        # resolved, existing artifacts after the stage
    log_path: Optional[str]
    error: Optional[str] = None
    # additive:
    dataset: Optional[str] = None
    label: Optional[str] = None


@dataclass
class BundleResult:
    run_name: str
    out_dir: str                   # absolute path to demo_bundle/<run_name>/
    status: str                    # "ok" | "partial" | "failed"
    stages: List[StageResult]
    headline: dict
    manifest_path: str
    index_html_path: str


# --------------------------------------------------------------------------- #
# Config building
# --------------------------------------------------------------------------- #

def _load_config_file(path: str) -> dict:
    """Load a JSON or YAML config override file. YAML requires PyYAML; JSON always works."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"--config {path!r} is YAML but PyYAML is not installed; "
                "use a .json config or `pip install pyyaml`."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def build_config(args) -> DemoConfig:
    """Build a :class:`DemoConfig` from parsed CLI args (an ``argparse.Namespace``-like).

    Precedence: explicit CLI flags > config-file values > built-in defaults. Applies the
    ``--smoke`` default of ``--max-identities 8`` and the ``<primary>_<timestamp>`` run name.
    """
    def _get(name, default=None):
        return getattr(args, name, default)

    file_cfg: dict = {}
    cfg_path = _get("config")
    if cfg_path:
        file_cfg = _load_config_file(cfg_path)

    datasets = _get("datasets") or file_cfg.get("datasets") or list(DEFAULT_DATASETS)
    primary = _get("primary_dataset") or file_cfg.get("primary_dataset")
    if not primary:
        primary = primary_in(datasets)

    db_path = _get("db") or file_cfg.get("db_path") or DEFAULT_DB_PATH
    out_dir = _get("out_dir") or file_cfg.get("out_dir") or DEFAULT_OUT_DIR
    signals = _get("signals") or file_cfg.get("signals") or DEFAULT_SIGNALS
    stages = _get("stages") or file_cfg.get("stages")

    smoke = bool(_get("smoke") or file_cfg.get("smoke", False))
    max_identities = _get("max_identities")
    if max_identities is None:
        max_identities = file_cfg.get("max_identities")
    if smoke and max_identities is None:
        max_identities = SMOKE_MAX_IDENTITIES

    run_name = _get("run_name") or file_cfg.get("run_name")
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{primary}_{stamp}"

    return DemoConfig(
        datasets=list(datasets),
        primary_dataset=primary,
        db_path=db_path,
        out_dir=out_dir,
        run_name=run_name,
        stages=list(stages) if stages else None,
        smoke=smoke,
        max_identities=max_identities,
        force=bool(_get("force") or file_cfg.get("force", False)),
        continue_on_error=bool(_get("continue_on_error") or file_cfg.get("continue_on_error", False)),
        signals=signals,
    )


def primary_in(datasets: List[str]) -> str:
    """Pick a sensible B-track primary dataset from a list (first B-track, else first)."""
    for ds in datasets:
        if DATASET_TRACKS.get(ds) == "B":
            return ds
    return datasets[0] if datasets else DEFAULT_PRIMARY


# --------------------------------------------------------------------------- #
# Stage registry (single source of wiring truth)
# --------------------------------------------------------------------------- #
#
# Each builder returns a fully-resolved StageSpec for one (config, dataset). This is the ONE
# place to edit if a downstream signature changes. Flags below were verified against the
# T02–T12 module CLIs (see each module's argparse block):
#   ingest    : python -m reid_demo.ingest    --wildlife-subset DS --db DB --dataset DS [--max-identities N]
#   species   : python -m reid_demo.species_filter --dataset DS --set-known SPECIES --db DB
#   embed     : python -m reid_demo.embed     --db DB --dataset DS
#   fisher    : python -m reid_demo.fisher    --db DB --dataset DS
#   fusion    : python -m reid_demo.fusion    --db DB --dataset DS --signals SIG --out-dir DIR
#   cluster   : python -m reid_demo.cluster   --db DB --dataset DS --flank-policy separate [--force]
#               (fused affinity has NO CLI flag -> run in-process via _cluster_with_affinity)
#   catalogue : python -m reid_demo.catalogue --db DB --dataset DS --out DIR
#   eval      : python -m reid_demo.eval      --dataset DS --db DB --tag TAG --flank-aware --out-dir DIR
#   report    : python -m reid_demo.medvednica_report --data-dir data/DS --out-dir DIR


def _py_module(mod: str, *args: str) -> List[str]:
    """argv (without the python exe) for ``python -m <mod> ...``."""
    return ["-m", mod, *[str(a) for a in args]]


def _catalogue_out(dataset: str) -> str:
    return os.path.join(CATALOGUE_STAGING, dataset)


def _eval_json_path(dataset: str, tag: str) -> str:
    return os.path.join(EVAL_OUT_DIR, f"{dataset}_{tag}.json")


def _fusion_npz_path(dataset: str, signals: str) -> str:
    return os.path.join(FUSION_STAGING, f"{dataset}_{signals}.npz")


def _spec_ingest(cfg: DemoConfig, ds: str) -> StageSpec:
    argv = _py_module("reid_demo.ingest", "--wildlife-subset", ds, "--db", cfg.db_path,
                      "--dataset", ds)
    if cfg.max_identities is not None:
        argv += ["--max-identities", str(cfg.max_identities)]
    return StageSpec(name="ingest", ticket="T02", critical=True, output_paths=[],
                     cli=argv, dataset=ds, skip_store_column="crop_path")


def _spec_species(cfg: DemoConfig, ds: str) -> StageSpec:
    species = B_TRACK_SPECIES.get(ds, "lynx")
    argv = _py_module("reid_demo.species_filter", "--dataset", ds, "--set-known", species,
                      "--db", cfg.db_path)
    return StageSpec(name="species", ticket="T03", critical=True, output_paths=[],
                     cli=argv, dataset=ds, label=f"species(set_known={species})",
                     skip_store_column="species")


def _spec_embed(cfg: DemoConfig, ds: str) -> StageSpec:
    argv = _py_module("reid_demo.embed", "--db", cfg.db_path, "--dataset", ds)
    return StageSpec(name="embed", ticket="T04", critical=True, output_paths=[],
                     cli=argv, dataset=ds, skip_store_column="embedding_ref")


def _spec_fisher(cfg: DemoConfig, ds: str) -> StageSpec:
    argv = _py_module("reid_demo.fisher", "--db", cfg.db_path, "--dataset", ds)
    # accuracy layer never blocks the run (degrades to global clustering on failure)
    return StageSpec(name="fisher", ticket="T11", critical=False, output_paths=[],
                     cli=argv, dataset=ds, skip_store_column="fisher_ref")


def _spec_fusion(cfg: DemoConfig, ds: str) -> StageSpec:
    npz = _fusion_npz_path(ds, cfg.signals)
    argv = _py_module("reid_demo.fusion", "--db", cfg.db_path, "--dataset", ds,
                      "--signals", cfg.signals, "--out-dir", FUSION_STAGING)
    return StageSpec(name="fusion", ticket="T12", critical=False, output_paths=[npz],
                     cli=argv, dataset=ds, label=f"fusion({cfg.signals})")


def _spec_cluster(cfg: DemoConfig, ds: str, *, affinity_path: Optional[str] = None,
                  force: bool = False, label: Optional[str] = None) -> StageSpec:
    if affinity_path is None:
        argv = _py_module("reid_demo.cluster", "--db", cfg.db_path, "--dataset", ds,
                          "--flank-policy", "separate")
        if force:
            argv += ["--force"]
        return StageSpec(name="cluster", ticket="T05", critical=True, output_paths=[],
                         cli=argv, dataset=ds, label=label,
                         skip_store_column=None if force else "cluster_id")
    # fused/GV affinity has no CLI flag -> in-process call with the loaded (N,N) matrix.
    return StageSpec(
        name="cluster", ticket="T05", critical=True, output_paths=[],
        func="reid_demo.run_demo:_cluster_with_affinity",
        kwargs={"db_path": cfg.db_path, "dataset": ds, "affinity_path": affinity_path,
                "force": force, "flank_policy": "separate"},
        dataset=ds, label=label or "cluster(fused-affinity)",
        skip_store_column=None if force else "cluster_id",
    )


def _spec_catalogue(cfg: DemoConfig, ds: str) -> StageSpec:
    out = _catalogue_out(ds)
    argv = _py_module("reid_demo.catalogue", "--db", cfg.db_path, "--dataset", ds, "--out", out)
    return StageSpec(name="catalogue", ticket="T06", critical=True,
                     output_paths=[os.path.join(out, "index.html")], cli=argv, dataset=ds)


def _spec_eval(cfg: DemoConfig, ds: str, *, tag: str, label: Optional[str] = None) -> StageSpec:
    out = _eval_json_path(ds, tag)
    argv = _py_module("reid_demo.eval", "--dataset", ds, "--db", cfg.db_path,
                      "--tag", tag, "--flank-aware", "--out-dir", EVAL_OUT_DIR)
    return StageSpec(name="eval", ticket="T07", critical=False, output_paths=[out],
                     cli=argv, dataset=ds, label=label or f"eval(tag={tag})")


def _spec_report(cfg: DemoConfig, ds: str) -> StageSpec:
    data_dir = os.path.join("data", ds)
    argv = _py_module("reid_demo.medvednica_report", "--data-dir", data_dir,
                      "--out-dir", REPORT_STAGING)
    return StageSpec(name="report", ticket="T09", critical=False,
                     output_paths=[os.path.join(REPORT_STAGING, "medvednica_summary.json")],
                     cli=argv, dataset=ds)


# Registry: stage name -> the builder used by plan_stages / --dry-run inspection.
STAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ingest": {"ticket": "T02", "track": "B", "build": _spec_ingest},
    "species": {"ticket": "T03", "track": "B", "build": _spec_species},
    "embed": {"ticket": "T04", "track": "B", "build": _spec_embed},
    "fisher": {"ticket": "T11", "track": "B", "build": _spec_fisher},
    "fusion": {"ticket": "T12", "track": "B", "build": _spec_fusion},
    "cluster": {"ticket": "T05", "track": "B", "build": _spec_cluster},
    "catalogue": {"ticket": "T06", "track": "B", "build": _spec_catalogue},
    "eval": {"ticket": "T07", "track": "B", "build": _spec_eval},
    "report": {"ticket": "T09", "track": "A", "build": _spec_report},
}


# --------------------------------------------------------------------------- #
# Planning (pure)
# --------------------------------------------------------------------------- #

def plan_stages(config: DemoConfig) -> List[StageSpec]:
    """Resolve, per dataset, the ordered list of :class:`StageSpec` in dependency order.

    Pure: no filesystem or subprocess side effects.

    * B-track ``global``: ``ingest -> species -> embed -> cluster -> {catalogue, eval}``.
    * B-track ``global+fisher`` / ``full-funnel``: baseline global ``cluster -> eval`` first
      (for the lift), then ``fisher -> fusion`` and the fused ``cluster -> {catalogue, eval}``.
    * A-track: ``report`` only.

    A ``--stages`` whitelist, if given, filters the resolved specs by name (order preserved).
    """
    specs: List[StageSpec] = []
    funnel = config.signals != SIGNAL_GLOBAL

    for ds in config.datasets:
        track = DATASET_TRACKS.get(ds, "B")
        if track == "A":
            specs.append(_spec_report(config, ds))
            continue

        # B-track
        specs.append(_spec_ingest(config, ds))
        specs.append(_spec_species(config, ds))
        specs.append(_spec_embed(config, ds))

        if not funnel:
            specs.append(_spec_cluster(config, ds))
            specs.append(_spec_catalogue(config, ds))
            specs.append(_spec_eval(config, ds, tag=SIGNAL_GLOBAL))
        else:
            # Baseline global pass first so the bundle can report the lift.
            specs.append(_spec_cluster(config, ds, label="cluster(baseline-global)"))
            specs.append(_spec_eval(config, ds, tag=SIGNAL_GLOBAL,
                                    label="eval(baseline-global)"))
            # Accuracy layer, then re-cluster on the fused affinity.
            specs.append(_spec_fisher(config, ds))
            specs.append(_spec_fusion(config, ds))
            affinity = _fusion_npz_path(ds, config.signals)
            specs.append(_spec_cluster(config, ds, affinity_path=affinity, force=True,
                                       label=f"cluster({config.signals})"))
            specs.append(_spec_catalogue(config, ds))
            specs.append(_spec_eval(config, ds, tag=_eval_tag(config.signals),
                                    label=f"eval({config.signals})"))

    if config.stages:
        wanted = set(config.stages)
        specs = [s for s in specs if s.name in wanted]
    return specs


# --------------------------------------------------------------------------- #
# In-process helpers (func stages)
# --------------------------------------------------------------------------- #

def _cluster_with_affinity(*, db_path: str, dataset: str, affinity_path: str,
                           force: bool = False, flank_policy: str = "separate", **_) -> Any:
    """Cluster on a precomputed fused/GV affinity (T12 -> T05's pluggable ``affinity=``)."""
    from . import cluster, fusion

    matrix, _ids = fusion.load_affinity(affinity_path)
    return cluster.run_clustering(
        db_path, dataset=dataset, affinity=matrix, force=force, flank_policy=flank_policy,
    )


# --------------------------------------------------------------------------- #
# Stage execution
# --------------------------------------------------------------------------- #

def _outputs_exist(spec: StageSpec) -> bool:
    return bool(spec.output_paths) and all(os.path.exists(p) for p in spec.output_paths)


def _store_stage_done(spec: StageSpec, db_path: str) -> bool:
    """For store-mutating stages: is the relevant column already populated for this dataset?"""
    if not spec.skip_store_column or not spec.dataset:
        return False
    if not os.path.exists(db_path):
        return False
    try:
        conn = store.connect(db_path, create=False)
    except Exception:
        return False
    try:
        rows = store.query_records(
            conn, dataset=spec.dataset,
            where_sql=f"{spec.skip_store_column} IS NOT NULL", limit=1,
        )
        return len(rows) > 0
    except Exception:
        return False
    finally:
        conn.close()


def _resolve_func(dotted: str):
    mod_name, _, func_name = dotted.partition(":")
    import importlib

    module = importlib.import_module(mod_name)
    return getattr(module, func_name)


def _run_stage(spec: StageSpec, config: DemoConfig, logs_dir: str) -> StageResult:
    """Execute one stage (subprocess or in-process), timing it and capturing logs."""
    log_name = f"{spec.name}_{spec.dataset or 'all'}"
    if spec.label and spec.label != spec.name:
        log_name = f"{spec.name}_{spec.dataset or 'all'}_{_eval_tag(spec.label)[:24]}"
    log_path = os.path.join(logs_dir, f"{log_name}.log")

    # Skip-if-exists (unless --force).
    if not config.force and (_outputs_exist(spec) or _store_stage_done(spec, config.db_path)):
        return StageResult(
            name=spec.name, ticket=spec.ticket, status="skipped", seconds=0.0,
            output_paths=[p for p in spec.output_paths if os.path.exists(p)],
            log_path=None, dataset=spec.dataset, label=spec.label,
        )

    os.makedirs(logs_dir, exist_ok=True)
    start = time.perf_counter()
    error: Optional[str] = None
    ok = False

    if spec.cli is not None:
        argv = [sys.executable, *spec.cli]
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(argv)}\n\n")
            logf.flush()
            try:
                proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, check=False)
                ok = proc.returncode == 0
                if not ok:
                    error = _stage_error_message(spec, f"non-zero exit {proc.returncode}", log_path)
            except FileNotFoundError as exc:
                error = _stage_error_message(spec, f"module/exe not found: {exc}", log_path)
    elif spec.func is not None:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# in-process: {spec.func}  kwargs={spec.kwargs}\n\n")
            logf.flush()
            try:
                fn = _resolve_func(spec.func)
                with redirect_stdout(logf), redirect_stderr(logf):
                    fn(**spec.kwargs)
                ok = True
            except Exception:
                tb = traceback.format_exc()
                logf.write("\n" + tb)
                error = _stage_error_message(spec, "in-process call raised", log_path)
    else:  # pragma: no cover - registry guarantees cli or func
        error = _stage_error_message(spec, "no cli/func wired", log_path)

    seconds = round(time.perf_counter() - start, 3)
    status = "ok" if ok else "failed"
    out_existing = [p for p in spec.output_paths if os.path.exists(p)]
    return StageResult(
        name=spec.name, ticket=spec.ticket, status=status, seconds=seconds,
        output_paths=out_existing, log_path=log_path, error=error,
        dataset=spec.dataset, label=spec.label,
    )


def _stage_error_message(spec: StageSpec, reason: str, log_path: str) -> str:
    invocation = ("python " + " ".join(spec.cli)) if spec.cli else f"in-process {spec.func}"
    return (
        f"stage '{spec.name}' ({spec.ticket}) failed: {reason}. "
        f"Expected invocation: `{invocation}`. "
        f"See {log_path}. If a downstream signature changed, update "
        f"STAGE_REGISTRY['{spec.name}'] in reid_demo/run_demo.py."
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_demo(config: DemoConfig) -> BundleResult:
    """Execute :func:`plan_stages`, then :func:`assemble_bundle`. Idempotent unless force."""
    specs = plan_stages(config)
    bundle_dir = os.path.abspath(os.path.join(config.out_dir, config.run_name))
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    results: List[StageResult] = []
    aborted = False
    for spec in specs:
        if aborted:
            results.append(StageResult(
                name=spec.name, ticket=spec.ticket, status="skipped", seconds=0.0,
                output_paths=[], log_path=None, dataset=spec.dataset, label=spec.label,
                error="skipped: a prior critical stage failed",
            ))
            continue

        res = _run_stage(spec, config, logs_dir)
        results.append(res)

        if res.status == "failed" and spec.critical and not config.continue_on_error:
            aborted = True  # stop launching further stages; still write the bundle

    return assemble_bundle(config, results)


# --------------------------------------------------------------------------- #
# Bundle assembly (pure-ish; testable with stubbed StageResults)
# --------------------------------------------------------------------------- #

def _copy_artifact(stage: StageResult, bundle_dir: str) -> List[str]:
    """Copy a stage's primary artifact into the bundle; return bundle-relative paths."""
    subdir, mode = BUNDLE_LAYOUT[stage.name]
    dest_root = os.path.join(bundle_dir, subdir)
    rels: List[str] = []

    if mode == "dir":
        src = _artifact_dir(stage.output_paths)
        if src and os.path.isdir(src):
            shutil.copytree(src, dest_root, dirs_exist_ok=True)
            for root, _dirs, files in os.walk(dest_root):
                for f in files:
                    rels.append(os.path.relpath(os.path.join(root, f), bundle_dir))
    elif mode == "file":
        os.makedirs(dest_root, exist_ok=True)
        for p in stage.output_paths:
            if os.path.isfile(p):
                dest = os.path.join(dest_root, os.path.basename(p))
                shutil.copy2(p, dest)
                rels.append(os.path.relpath(dest, bundle_dir))
    return rels


def _artifact_dir(output_paths: List[str]) -> Optional[str]:
    """The directory to copy for a 'dir' stage: parent of index.html, else first path's dir."""
    for p in output_paths:
        if os.path.basename(p) == "index.html":
            return os.path.dirname(p)
    for p in output_paths:
        if os.path.isdir(p):
            return p
        if os.path.isfile(p):
            return os.path.dirname(p)
    return None


def _read_eval_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _headline_from_eval(report: dict) -> dict:
    """Map a T07 ClusteringReport dict to the biologist-facing headline trio (no recompute)."""
    return {
        "individuals_found": report.get("n_found_clusters"),
        "individuals_true": report.get("n_true_individuals"),
        "pct_photos_correctly_grouped": report.get("pct_photos_correctly_grouped"),
    }


def _store_block(config: DemoConfig) -> tuple:
    """Return (store_counts, candidate_new, crops_total) for the primary dataset; defensive."""
    counts: Dict[str, Any] = {}
    candidate_new = None
    crops_total = None
    if not os.path.exists(config.db_path):
        return counts, candidate_new, crops_total
    try:
        conn = store.connect(config.db_path, create=False)
    except Exception:
        return counts, candidate_new, crops_total
    try:
        ds = config.primary_dataset
        counts["by_species"] = store.count_by(conn, "species", dataset=ds)
        counts["review_status"] = store.count_by(conn, "review_status", dataset=ds)
        cand = store.count_by(conn, "is_candidate_new", dataset=ds)
        # count_by keys are stringified column values; sum the truthy ("1") bucket.
        candidate_new = int(cand.get(1, cand.get("1", 0)))
        crops_total = sum(int(v) for v in store.count_by(conn, "dataset", dataset=ds).values()) or None
        # cluster-size histogram (size -> #clusters), excluding noise/candidate-new (-1).
        rows = store.query_records(conn, dataset=ds, where_sql="cluster_id IS NOT NULL")
        sizes: Dict[Any, int] = {}
        for r in rows:
            cid = getattr(r, "cluster_id", None)
            if cid is None or cid < 0:
                continue
            sizes[cid] = sizes.get(cid, 0) + 1
        hist: Dict[str, int] = {}
        for n in sizes.values():
            hist[str(n)] = hist.get(str(n), 0) + 1
        counts["by_cluster_size_histogram"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
    except Exception:
        pass
    finally:
        conn.close()
    return counts, candidate_new, crops_total


def assemble_bundle(config: DemoConfig, stage_results: List[StageResult]) -> BundleResult:
    """Gather stage artifacts into ``demo_bundle/<run_name>/``; render index/summary/manifest.

    Pure-ish: tests call this with hand-built :class:`StageResult`s pointing at fixture files,
    so no models run. Artifacts are COPIED (not symlinked) for portability.
    """
    bundle_dir = os.path.abspath(os.path.join(config.out_dir, config.run_name))
    os.makedirs(bundle_dir, exist_ok=True)

    # 1. Copy stage artifacts; record bundle-relative paths per stage.
    copied: Dict[int, List[str]] = {}
    for i, st in enumerate(stage_results):
        if st.name in BUNDLE_LAYOUT and st.status in ("ok", "skipped") and st.output_paths:
            try:
                copied[i] = _copy_artifact(st, bundle_dir)
            except Exception:
                copied[i] = []

    # 2. Headline + lift from the eval JSON(s) for the primary dataset.
    headline = _build_headline(config, stage_results)

    # 3. Store-derived counts.
    store_counts, candidate_new, crops_total = _store_block(config)
    headline["candidate_new_individuals"] = candidate_new
    headline["crops_total"] = crops_total
    headline["dataset"] = config.primary_dataset

    # 4. Overall status.
    status = _overall_status(config, stage_results)

    # 5. Manifest.
    manifest = _build_manifest(config, stage_results, copied, headline, store_counts, status)
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # 6. index.html + SUMMARY.md.
    index_path = os.path.join(bundle_dir, "index.html")
    summary_path = os.path.join(bundle_dir, "SUMMARY.md")
    has_catalogue = os.path.isfile(os.path.join(bundle_dir, "catalogue", "index.html"))
    has_report = os.path.isdir(os.path.join(bundle_dir, "medvednica_report"))
    report_rel = _report_index_rel(bundle_dir)
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(_render_index_html(config, headline, manifest, has_catalogue, report_rel))
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(_render_summary_md(config, headline, manifest, has_catalogue, has_report))

    return BundleResult(
        run_name=config.run_name, out_dir=bundle_dir, status=status,
        stages=stage_results, headline=headline,
        manifest_path=manifest_path, index_html_path=index_path,
    )


def _build_headline(config: DemoConfig, stage_results: List[StageResult]) -> dict:
    """Read T07's single eval JSON (verbatim) for the primary dataset; add lift if present."""
    headline: Dict[str, Any] = {
        "dataset": config.primary_dataset,
        "individuals_found": None,
        "individuals_true": None,
        "pct_photos_correctly_grouped": None,
        "candidate_new_individuals": None,
        "crops_total": None,
    }

    # Collect eval reports for the primary dataset, keyed by their own 'tag'.
    by_tag: Dict[str, dict] = {}
    for st in stage_results:
        if st.name != "eval" or st.dataset != config.primary_dataset:
            continue
        if st.status not in ("ok", "skipped"):
            continue
        for p in st.output_paths:
            if p.endswith(".json"):
                rep = _read_eval_json(p)
                if rep:
                    by_tag[str(rep.get("tag", "default"))] = rep

    if not by_tag:
        return headline

    funnel_tag = _eval_tag(config.signals)
    primary_report = by_tag.get(funnel_tag) or by_tag.get(SIGNAL_GLOBAL) or next(iter(by_tag.values()))
    headline.update(_headline_from_eval(primary_report))

    # Lift: baseline (global) vs the funnel run, when both exist.
    if config.signals != SIGNAL_GLOBAL and SIGNAL_GLOBAL in by_tag and funnel_tag in by_tag:
        headline["lift"] = {
            "baseline_signals": SIGNAL_GLOBAL,
            "global": _headline_from_eval(by_tag[SIGNAL_GLOBAL]),
            "full_funnel_signals": config.signals,
            "full_funnel": _headline_from_eval(by_tag[funnel_tag]),
        }
    return headline


def _overall_status(config: DemoConfig, stage_results: List[StageResult]) -> str:
    """ok = no failures; partial = only non-critical failed; failed = a critical failed."""
    critical_names = {s.name for s in plan_stages(config) if s.critical}
    any_failed = any(r.status == "failed" for r in stage_results)
    critical_failed = any(r.status == "failed" and r.name in critical_names for r in stage_results)
    if critical_failed:
        return "failed"
    if any_failed:
        return "partial"
    return "ok"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None


def _build_manifest(config: DemoConfig, stage_results: List[StageResult],
                    copied: Dict[int, List[str]], headline: dict,
                    store_counts: dict, status: str) -> dict:
    stages_json = []
    for i, st in enumerate(stage_results):
        stages_json.append({
            "name": st.name,
            "ticket": st.ticket,
            "dataset": st.dataset,
            "label": st.label,
            "status": st.status,
            "seconds": st.seconds,
            "output_paths": copied.get(i, []),   # bundle-relative; [] for store-mutating stages
            "log": (os.path.relpath(st.log_path, os.path.join(config.out_dir, config.run_name))
                    if st.log_path else None),
            "error": st.error,
        })

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_name": config.run_name,
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "status": status,
        "db_path": config.db_path,
        "primary_dataset": config.primary_dataset,
        "datasets": list(config.datasets),
        "signals": config.signals,
        "command": "python -m reid_demo.run_demo " + " ".join(_config_to_argv(config)),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "headline": headline,
        "stages": stages_json,
        "store_counts": store_counts,
        "bundle": {
            "index_html": "index.html",
            "summary_md": "SUMMARY.md",
            "catalogue_dir": "catalogue",
            "accuracy_dir": "accuracy",
            "medvednica_report_dir": "medvednica_report",
        },
    }


def _config_to_argv(config: DemoConfig) -> List[str]:
    argv = ["--datasets", *config.datasets, "--primary-dataset", config.primary_dataset,
            "--db", config.db_path, "--out-dir", config.out_dir, "--run-name", config.run_name,
            "--signals", config.signals]
    if config.smoke:
        argv.append("--smoke")
    if config.max_identities is not None:
        argv += ["--max-identities", str(config.max_identities)]
    if config.force:
        argv.append("--force")
    if config.continue_on_error:
        argv.append("--continue-on-error")
    return argv


# --------------------------------------------------------------------------- #
# Rendering (offline, no JS/CDN)
# --------------------------------------------------------------------------- #

def _report_index_rel(bundle_dir: str) -> Optional[str]:
    """Best link target inside medvednica_report/ (index.html > report .md > summary json)."""
    rdir = os.path.join(bundle_dir, "medvednica_report")
    if not os.path.isdir(rdir):
        return None
    for cand in ("index.html", "medvednica_report.md", "medvednica_summary.json"):
        if os.path.isfile(os.path.join(rdir, cand)):
            return f"medvednica_report/{cand}"
    return "medvednica_report/"


def _headline_sentence(headline: dict) -> Optional[str]:
    found = headline.get("individuals_found")
    true = headline.get("individuals_true")
    pct = headline.get("pct_photos_correctly_grouped")
    if found is None and pct is None:
        return None
    parts = []
    if found is not None and true is not None:
        parts.append(f"Found {found} individuals vs {true} known")
    elif found is not None:
        parts.append(f"Found {found} individuals")
    if pct is not None:
        parts.append(f"{pct:g}% of photos correctly grouped")
    cand = headline.get("candidate_new_individuals")
    if cand:
        parts.append(f"{cand} possible new individuals to review")
    return "; ".join(parts) + "."


def _render_index_html(config: DemoConfig, headline: dict, manifest: dict,
                       has_catalogue: bool, report_rel: Optional[str]) -> str:
    sentence = _headline_sentence(headline)
    headline_html = (f"<p class='headline'>{sentence}</p>" if sentence
                     else "<p class='headline muted'>Individual-ID accuracy not available for this run.</p>")

    cat_card = (
        "<div class='card'><h3>(B) Proof we can ID individuals</h3>"
        "<p>Open-set clustering grouped an unlabeled pile of photos into individual animals, "
        "validated against known identities.</p>"
        + ("<p><a href='catalogue/index.html'>Open the visual catalogue &rarr;</a></p>"
           if has_catalogue else "<p class='muted'>Catalogue not produced.</p>")
        + _accuracy_table_html(headline)
        + "</div>"
    )

    report_card = (
        "<div class='card'><h3>(A) Your footage, filtered</h3>"
        "<p>Empty frames removed and detections sorted by species on real camera-trap data.</p>"
        + (f"<p><a href='{report_rel}'>Open the filtering report &rarr;</a></p>"
           if report_rel else "<p class='muted'>Filtering report not produced.</p>")
        + "</div>"
    )

    lift_html = _lift_html(headline)
    tech_html = _tech_details_html(manifest)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Re-ID demo — {config.primary_dataset}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
   background:#f5f6f8;color:#1c2530;line-height:1.5}}
 .wrap{{max-width:920px;margin:0 auto;padding:32px 20px 64px}}
 h1{{font-size:1.6rem;margin:0 0 4px}}
 .sub{{color:#5b6675;margin:0 0 24px}}
 .headline{{font-size:1.3rem;font-weight:600;background:#0b6b4f;color:#fff;
   padding:18px 22px;border-radius:10px}}
 .cards{{display:flex;gap:18px;flex-wrap:wrap;margin-top:24px}}
 .card{{flex:1 1 360px;background:#fff;border:1px solid #e2e6ec;border-radius:10px;
   padding:18px 20px}}
 .card h3{{margin:0 0 8px}}
 a{{color:#0b6b4f;font-weight:600;text-decoration:none}}
 a:hover{{text-decoration:underline}}
 table{{border-collapse:collapse;margin-top:10px;width:100%}}
 td,th{{border:1px solid #e2e6ec;padding:6px 10px;text-align:left;font-size:.92rem}}
 .muted{{color:#8a94a3}}
 details{{margin-top:28px;background:#fff;border:1px solid #e2e6ec;border-radius:10px;padding:8px 18px}}
 summary{{cursor:pointer;font-weight:600;padding:8px 0}}
 .lift{{margin-top:24px;background:#fff;border:1px solid #e2e6ec;border-radius:10px;padding:16px 20px}}
 code{{background:#eef1f4;padding:1px 5px;border-radius:4px}}
</style></head>
<body><div class="wrap">
 <h1>Animal Re-Identification — Demo</h1>
 <p class="sub">{config.primary_dataset} &middot; signals: <code>{config.signals}</code>
   &middot; status: {manifest['status']} &middot; {manifest['created_at']}</p>
 {headline_html}
 <div class="cards">{report_card}{cat_card}</div>
 {lift_html}
 {tech_html}
</div></body></html>
"""


def _accuracy_table_html(headline: dict) -> str:
    rows = [
        ("Individuals found", headline.get("individuals_found")),
        ("Known individuals", headline.get("individuals_true")),
        ("% photos correctly grouped", headline.get("pct_photos_correctly_grouped")),
        ("Possible new individuals", headline.get("candidate_new_individuals")),
        ("Total photos", headline.get("crops_total")),
    ]
    body = "".join(
        f"<tr><td>{k}</td><td>{'' if v is None else (f'{v:g}' if isinstance(v, float) else v)}</td></tr>"
        for k, v in rows if v is not None
    )
    if not body:
        return ""
    return f"<table>{body}</table>"


def _lift_html(headline: dict) -> str:
    lift = headline.get("lift")
    if not lift:
        return ""
    g = lift["global"]
    f = lift["full_funnel"]
    return (
        "<div class='lift'><h3>Accuracy lift</h3>"
        f"<p>Global backbone finds <b>{g.get('individuals_found')}</b> individuals "
        f"({g.get('pct_photos_correctly_grouped')}% correctly grouped); "
        f"with geometric verification (<code>{lift.get('full_funnel_signals')}</code>) "
        f"&rarr; <b>{f.get('individuals_found')}</b> individuals "
        f"({f.get('pct_photos_correctly_grouped')}% correctly grouped).</p></div>"
    )


def _tech_details_html(manifest: dict) -> str:
    stage_rows = "".join(
        f"<tr><td>{s['name']}</td><td>{s['ticket']}</td><td>{s.get('dataset') or ''}</td>"
        f"<td>{s['status']}</td><td>{s['seconds']:g}</td></tr>"
        for s in manifest.get("stages", [])
    )
    return (
        "<details><summary>Technical details</summary>"
        f"<p>Command: <code>{manifest.get('command','')}</code><br>"
        f"Git: <code>{manifest.get('git_commit')}</code> &middot; "
        f"Python {manifest.get('python')} &middot; DB <code>{manifest.get('db_path')}</code></p>"
        "<table><tr><th>stage</th><th>ticket</th><th>dataset</th><th>status</th><th>sec</th></tr>"
        f"{stage_rows}</table></details>"
    )


def _render_summary_md(config: DemoConfig, headline: dict, manifest: dict,
                       has_catalogue: bool, has_report: bool) -> str:
    sentence = _headline_sentence(headline) or "_Individual-ID accuracy not available for this run._"
    lines = [
        f"# Re-ID demo — {config.primary_dataset}",
        "",
        f"**{sentence}**",
        "",
        f"- Signals: `{config.signals}`",
        f"- Status: {manifest['status']}",
        f"- Generated: {manifest['created_at']}",
        "",
        "## (A) Your footage, filtered",
        ("See `medvednica_report/`." if has_report else "_Not produced._"),
        "",
        "## (B) Proof we can ID individuals",
        ("See `catalogue/`." if has_catalogue else "_Not produced._"),
        "",
    ]
    lift = headline.get("lift")
    if lift:
        g, f = lift["global"], lift["full_funnel"]
        lines += [
            "## Accuracy lift",
            f"- Global backbone: {g.get('individuals_found')} individuals, "
            f"{g.get('pct_photos_correctly_grouped')}% correctly grouped",
            f"- `{lift.get('full_funnel_signals')}`: {f.get('individuals_found')} individuals, "
            f"{f.get('pct_photos_correctly_grouped')}% correctly grouped",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m reid_demo.run_demo",
        description="T10 — run the open-set re-ID pipeline and assemble a shareable demo bundle.",
    )
    p.add_argument("--config", default=None, help="YAML or JSON config file; overrides defaults.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help=f"datasets to process (default: {' '.join(DEFAULT_DATASETS)}).")
    p.add_argument("--primary-dataset", default=None,
                   help=f"B-track dataset driving the headline (default: {DEFAULT_PRIMARY}).")
    p.add_argument("--db", default=None, help=f"shared T01 store (default: {DEFAULT_DB_PATH}).")
    p.add_argument("--out-dir", default=None, help=f"bundle root (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--run-name", default=None, help="default: <primary-dataset>_<timestamp>.")
    p.add_argument("--stages", nargs="+", default=None, help="subset of stage names to run.")
    p.add_argument("--signals", default=None, choices=SIGNAL_CHOICES,
                   help=f"clustering signal set (default: {DEFAULT_SIGNALS}).")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset (caps identities) for a fast test run.")
    p.add_argument("--max-identities", type=int, default=None,
                   help=f"cap distinct identities (default when --smoke: {SMOKE_MAX_IDENTITIES}).")
    p.add_argument("--force", action="store_true", help="re-run stages even if outputs exist.")
    p.add_argument("--dry-run", action="store_true", help="print the ordered plan; execute nothing.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="keep going if a non-critical stage fails (flagged in manifest).")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose output.")
    return p


def _print_plan(config: DemoConfig) -> None:
    specs = plan_stages(config)
    print(f"Demo plan — run_name={config.run_name}  signals={config.signals}  db={config.db_path}")
    print(f"datasets={config.datasets}  primary={config.primary_dataset}")
    print("-" * 78)
    for i, s in enumerate(specs, 1):
        invocation = ("python " + " ".join(s.cli)) if s.cli else f"in-process {s.func} {s.kwargs}"
        crit = "critical" if s.critical else "optional"
        print(f"{i:2}. [{s.ticket}] {s.display_label}  ({s.dataset or '-'}, {crit})")
        print(f"      {invocation}")
        if s.output_paths:
            print(f"      outputs: {s.output_paths}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)

    if getattr(args, "dry_run", False):
        _print_plan(config)
        return 0

    try:
        result = run_demo(config)
    except Exception:
        traceback.print_exc()
        return 1

    print(f"Bundle: {result.out_dir}")
    print(f"Status: {result.status}")
    sentence = _headline_sentence(result.headline)
    if sentence:
        print(sentence)

    # Exit 0 iff status != failed and the bundle (index + manifest) was written.
    wrote_bundle = (os.path.exists(result.index_html_path)
                    and os.path.exists(result.manifest_path))
    if result.status == "failed" or not wrote_bundle:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
