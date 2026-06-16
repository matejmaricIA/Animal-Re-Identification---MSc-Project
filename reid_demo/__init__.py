"""reid_demo — open-set lynx re-ID demo/pilot package.

T01 ships the shared store + data contract. Downstream tickets (T02–T12) add
ingestion, species filtering, embedding, clustering, catalogue, eval, review,
reports and the end-to-end runner. See ``DATA_CONTRACT.md`` and ``STATUS_BOARD.md``.
"""

from .store import (
    DetectionRecord,
    SCHEMA_VERSION,
    DEFAULT_DB_PATH,
    TABLE_NAME,
    COLUMNS,
    REVIEW_STATUSES,
    ORIENTATIONS,
    connect,
    init_db,
    upsert_record,
    upsert_records,
    get_record,
    query_records,
    update_species,
    update_embedding,
    update_cluster,
    update_review,
    update_extra,
    count_by,
    make_record_id,
    export_records,
    import_records,
    to_dataframe,
)

from .embed import (
    embed_records,
    embed_crops,
    get_embedding_matrix,
    load_embeddings,
    EmbedResult,
    embedding_cache_path,
    DEFAULT_EMB_DIR,
    DEFAULT_MODEL_NAME,
)

# T11 — local-feature + Fisher-vector service (additive; heavy deps lazy-imported).
from .fisher import (
    build_fisher_records,
    build_fisher_vectors,
    get_fisher_matrix,
    load_fisher_vectors,
    FisherResult,
    fisher_cache_path,
    fisher_cache_label,
    DEFAULT_FISHER_DIR,
    DEFAULT_METHOD,
    DEFAULT_PCA_DIM,
)

# T06 — visual individual catalogue generator (additive; read-only consumer).
from .catalogue import (
    build_catalogue,
    CatalogueResult,
)

# T07 — clustering evaluation harness (additive; read-only consumer).
from .eval import (
    ClusteringReport,
    load_eval_frame,
    build_label_arrays,
    standard_metrics,
    plain_language_metrics,
    evaluate_clustering,
    save_report,
)

# T08 — human-in-the-loop review tool (additive).
from .review import (
    build_review_queue,
    apply_decisions,
    serve_review_ui,
    review_status_summary,
    build_pair_image,
    ReviewItem,
    ReviewDecision,
    DECISIONS,
    DEFAULT_QUEUE_SIZE,
    LOW_CONF_THRESHOLD,
)

# T12 — multi-signal fusion + GV reranking (additive; heavy deps lazy-imported).
from .fusion import (
    build_fused_affinity,
    affinity_provider,
    select_borderline_pairs,
    gv_rerank,
    refine_affinity_with_gv,
    run_fusion,
    load_affinity,
    FusionResult,
    PairScore,
    DEFAULT_SIGNALS,
    SIGNAL_SETS,
    FUSION_DIR,
)

__all__ = [
    "DetectionRecord",
    "SCHEMA_VERSION",
    "DEFAULT_DB_PATH",
    "TABLE_NAME",
    "COLUMNS",
    "REVIEW_STATUSES",
    "ORIENTATIONS",
    "connect",
    "init_db",
    "upsert_record",
    "upsert_records",
    "get_record",
    "query_records",
    "update_species",
    "update_embedding",
    "update_cluster",
    "update_review",
    "update_extra",
    "count_by",
    "make_record_id",
    "export_records",
    "import_records",
    "to_dataframe",
    "embed_records",
    "embed_crops",
    "get_embedding_matrix",
    "load_embeddings",
    "EmbedResult",
    "embedding_cache_path",
    "DEFAULT_EMB_DIR",
    "DEFAULT_MODEL_NAME",
    "build_fisher_records",
    "build_fisher_vectors",
    "get_fisher_matrix",
    "load_fisher_vectors",
    "FisherResult",
    "fisher_cache_path",
    "fisher_cache_label",
    "DEFAULT_FISHER_DIR",
    "DEFAULT_METHOD",
    "DEFAULT_PCA_DIM",
    # T06 catalogue
    "build_catalogue",
    "CatalogueResult",
    # T07 eval
    "ClusteringReport",
    "load_eval_frame",
    "build_label_arrays",
    "standard_metrics",
    "plain_language_metrics",
    "evaluate_clustering",
    "save_report",
    # T08 review
    "build_review_queue",
    "apply_decisions",
    "serve_review_ui",
    "review_status_summary",
    "build_pair_image",
    "ReviewItem",
    "ReviewDecision",
    "DECISIONS",
    "DEFAULT_QUEUE_SIZE",
    "LOW_CONF_THRESHOLD",
    # T12 fusion
    "build_fused_affinity",
    "affinity_provider",
    "select_borderline_pairs",
    "gv_rerank",
    "refine_affinity_with_gv",
    "run_fusion",
    "load_affinity",
    "FusionResult",
    "PairScore",
    "DEFAULT_SIGNALS",
    "SIGNAL_SETS",
    "FUSION_DIR",
]
