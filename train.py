import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from wildlife_datasets import splits

import preprocessing
from constants import (
    EVAL_RESULTS_XLSX,
    GEOMETRIC_CANDIDATES,
    MD_DATASET_SPLITS,
    N_COMPONENTS_PCA,
    UNION_CANDIDATES,
)
from evaluate import evaluate_predictions, save_evaluation_results
from feature_aggregation import (
    compute_fisher_vectors,
    ensure_local_descriptors,
    feature_descriptor_dir,
    load_descriptors,
    load_keypoints,
    load_or_train_fisher_vectors,
)
from feature_extraction import get_image_paths
from global_embedding import global_embedding_cache_label, load_or_build_global_embeddings
from predict import classify_single_image_late_fusion, classify_test_images_late_fusion
from train_late_fusion import train_calibrators_two_stage
from utils.make_classification_pipeline_assets import build_assets_from_funnel
from utility_functions import combine_fisher_vectors, save_count_results_wrapper


def run_training_for_dataset(
    dataset_name: str,
    df_raw: pd.DataFrame,
    *,
    args,
    train_use_fisher: bool,
    method: str,
    method_tag: str,
    ensemble_methods: list[str],
    ensamble_weights: list[float],
    gv_features: str,
    gv_matcher: str,
    seg_tag: str,
    tag: str,
    use_cuda: bool,
) -> None:
    print(f"Training mode selected. Using dataset: {dataset_name}")
    print("training...")

    md_meta = MD_DATASET_SPLITS.get(str(dataset_name).strip().lower(), None)
    if md_meta is not None:
        print(
            "MD split metadata: "
            f"trained_on={md_meta['trained_on']}, "
            f"split_type={md_meta['split_type']}, "
            f"random_split={md_meta['random_split']}"
        )

    feature_method = method
    start_eval_time = time.time()
    use_md_baseline_split = bool(md_meta is not None and md_meta.get("trained_on"))
    if use_md_baseline_split:
        print("[MD] Using MegaDescriptor baseline split (split_md_baseline).")

    df, csv_path, _ = preprocessing.prepare_processed_dataset(
        dataset_name,
        df_raw,
        remove_background=args.remove_background,
        use_mantiuk=args.use_mantiuk,
        require_processed_paths=True,
    )

    if use_md_baseline_split:
        repo_root = Path(__file__).resolve().parent
        baseline_root = repo_root / "test-scripts" / "wildlife-tools-data"
        metadata_root = baseline_root / "metadata" / "datasets"

        def _resolve_baseline_dir(root: Path, name: str) -> Path | None:
            direct = root / name
            if direct.exists():
                return direct
            name_lower = str(name).lower()
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == name_lower:
                    return child
            return None

        def _normalize_split_path(value: str) -> str:
            path = str(value).strip().replace("\\", "/")
            if path.startswith("./"):
                path = path[2:]
            if path.lower().startswith("images/"):
                parts = path.split("/", 2)
                if len(parts) >= 3:
                    path = parts[2]
            return path.lower()

        baseline_dir = _resolve_baseline_dir(metadata_root, str(dataset_name))
        if baseline_dir is None:
            print(f"[ERROR] MD baseline metadata not found for dataset: {dataset_name}")
            sys.exit(1)
        metadata_csv = baseline_dir / "metadata.csv"
        if not metadata_csv.exists():
            print(f"[ERROR] Missing MD baseline metadata CSV: {metadata_csv}")
            sys.exit(1)

        df_baseline = pd.read_csv(metadata_csv, dtype={"image_id": str, "identity": str})
        unnamed_cols = [c for c in df_baseline.columns if str(c).lower().startswith("unnamed")]
        if unnamed_cols:
            df_baseline = df_baseline.drop(columns=unnamed_cols)
        if "path" not in df_baseline.columns or "split" not in df_baseline.columns:
            print(f"[ERROR] MD baseline metadata missing required columns in: {metadata_csv}")
            sys.exit(1)

        baseline_split_map = dict(
            zip(
                df_baseline["path"].astype(str).map(_normalize_split_path),
                df_baseline["split"].astype(str),
            )
        )

        if "path" not in df.columns:
            print("[ERROR] Dataset metadata missing 'path' column; cannot apply MD baseline split.")
            sys.exit(1)

        key_series = df["path"].astype(str).map(_normalize_split_path)
        mapped_split = key_series.map(baseline_split_map)
        matched = int(mapped_split.notna().sum())
        total = int(len(df))
        if matched == 0:
            print(
                f"[ERROR] Could not match any paths to MD baseline metadata for dataset: {dataset_name}."
            )
            sys.exit(1)
        if matched != total:
            print(
                f"[WARN] MD baseline split matched {matched}/{total} rows; leaving {total - matched} unchanged."
            )

        existing_split = (
            df["split"].astype(str).str.lower()
            if "split" in df.columns
            else pd.Series(["train"] * len(df), index=df.index)
        )
        df["split_md_baseline"] = mapped_split.fillna(existing_split).astype(str).str.lower()
        if md_meta is not None:
            df["split_type"] = md_meta.get("split_type")
            df["trained_on"] = bool(md_meta.get("trained_on"))
            df["random_split"] = bool(md_meta.get("random_split"))

        df.to_csv(csv_path, index=False)

    if ("split" in df.columns) or (use_md_baseline_split and "split_md_baseline" in df.columns):
        print("Using predefined split in WildlifeReID10k.")
        split_col = "split"
        if use_md_baseline_split and "split_md_baseline" in df.columns:
            split_col = "split_md_baseline"
        train_mask = df[split_col].astype(str).str.lower() != "test"
        test_mask = df[split_col].astype(str).str.lower() == "test"
        df_train, df_test = df[train_mask], df[test_mask]

        if args.split_type == "closed":
            print("Using closed set split.")
            train_identities = set(df_train["identity"].unique())
            df_test = df_test[df_test["identity"].isin(train_identities)].copy()
        splits.analyze_split(df, df_train.index, df_test.index)
    else:
        print("[ERROR] Missing 'split' information in dataset metadata.")
        sys.exit(1)

    train_image_items = get_image_paths(df_train, args.remove_background)
    test_image_items = get_image_paths(df_test, args.remove_background)
    train_paths_map = dict(zip(df_train["image_id"].astype(str), train_image_items))
    test_paths_map = dict(zip(df_test["image_id"].astype(str), test_image_items))

    image_paths = None
    if gv_matcher == "loftr":
        image_paths = dict(
            zip(df["image_id"].astype(str), get_image_paths(df, args.remove_background))
        )

    ds_tag = dataset_name
    base_dir = f"./data/{ds_tag}"
    os.makedirs(base_dir, exist_ok=True)

    pca_dim = int(getattr(args, "pca_dim", N_COMPONENTS_PCA))
    # Always include PCA dim to avoid reusing old cached PCA/GMM/FVs.
    pca_cache_suffix = f"{seg_tag}_pca{pca_dim}"

    fusion_signals_set = set(args.fusion_signals or [])
    use_fisher_signal = bool(train_use_fisher)
    use_gv_signal = "gv" in fusion_signals_set

    train_dict, test_dict = {}, {}
    train_keypoints, test_keypoints = {}, {}
    fv_tr, fv_te = {}, {}
    if use_fisher_signal:
        if feature_method == "ensamble":
            methods = ensemble_methods
            for m in methods:
                dir_tr = feature_descriptor_dir(base_dir, m, "train", seg_tag)
                dir_te = feature_descriptor_dir(base_dir, m, "test", seg_tag)
                ensure_local_descriptors(train_image_items, m, dir_tr)
                ensure_local_descriptors(test_image_items, m, dir_te)

            fv_tr_list = []
            fv_te_list = []
            for m in methods:
                dir_tr = feature_descriptor_dir(base_dir, m, "train", seg_tag)
                dir_te = feature_descriptor_dir(base_dir, m, "test", seg_tag)
                train_dict_m = load_descriptors(os.path.join(dir_tr, "descriptors.h5"))
                test_dict_m = load_descriptors(os.path.join(dir_te, "descriptors.h5"))
                pca_m, gmm_m, fv_tr_m = load_or_train_fisher_vectors(
                    ds_tag=ds_tag,
                    method_name=m,
                    cache_suffix=pca_cache_suffix,
                    descriptors=train_dict_m,
                    pca_dim=pca_dim,
                )
                fv_te_m = compute_fisher_vectors(test_dict_m, pca_m, gmm_m)
                fv_tr_list.append(fv_tr_m)
                fv_te_list.append(fv_te_m)

            fv_tr = combine_fisher_vectors(fv_tr_list, ensamble_weights)
            fv_te = combine_fisher_vectors(fv_te_list, ensamble_weights)

        else:
            dir_tr = feature_descriptor_dir(base_dir, feature_method, "train", seg_tag)
            dir_te = feature_descriptor_dir(base_dir, feature_method, "test", seg_tag)
            ensure_local_descriptors(train_image_items, feature_method, dir_tr)
            ensure_local_descriptors(test_image_items, feature_method, dir_te)

            train_dict = load_descriptors(os.path.join(dir_tr, "descriptors.h5"))
            test_dict = load_descriptors(os.path.join(dir_te, "descriptors.h5"))
            train_keypoints = load_keypoints(os.path.join(dir_tr, "keypoints.h5"))
            test_keypoints = load_keypoints(os.path.join(dir_te, "keypoints.h5"))

            pca, gmm, fv_tr = load_or_train_fisher_vectors(
                ds_tag=ds_tag,
                method_name=feature_method,
                cache_suffix=pca_cache_suffix,
                descriptors=train_dict,
                pca_dim=pca_dim,
            )
            fv_te = compute_fisher_vectors(test_dict, pca, gmm)

    if use_gv_signal and (
        not train_dict or not test_dict or not train_keypoints or not test_keypoints
    ):
        gv_dir_tr = feature_descriptor_dir(base_dir, gv_features, "train", seg_tag)
        gv_dir_te = feature_descriptor_dir(base_dir, gv_features, "test", seg_tag)
        ensure_local_descriptors(train_image_items, gv_features, gv_dir_tr)
        ensure_local_descriptors(test_image_items, gv_features, gv_dir_te)
        train_dict = load_descriptors(os.path.join(gv_dir_tr, "descriptors.h5"))
        test_dict = load_descriptors(os.path.join(gv_dir_te, "descriptors.h5"))
        train_keypoints = load_keypoints(os.path.join(gv_dir_tr, "keypoints.h5"))
        test_keypoints = load_keypoints(os.path.join(gv_dir_te, "keypoints.h5"))

    if args.use_global_embedding:
        print("Extracting global embeddings...")
        global_ckpt = getattr(args, "global_ckpt", "") or None
        emb_model_label = global_embedding_cache_label(args.embedding_model, global_ckpt)
        emb_tr_path = f"{base_dir}/global_embeddings_train_{emb_model_label}_{seg_tag}.pkl"
        emb_te_path = f"{base_dir}/global_embeddings_test_{emb_model_label}_{seg_tag}.pkl"
        emb_tr = load_or_build_global_embeddings(
            train_paths_map,
            emb_tr_path,
            model_name=args.embedding_model,
            checkpoint_path=global_ckpt,
        )
        emb_te = load_or_build_global_embeddings(
            test_paths_map,
            emb_te_path,
            model_name=args.embedding_model,
            checkpoint_path=global_ckpt,
        )
    else:
        emb_tr, emb_te = {}, {}

    train_labels = dict(zip(df_train["image_id"], df_train["identity"]))
    test_labels = dict(zip(df_test["image_id"], df_test["identity"]))
    calibrators = train_calibrators_two_stage(
        train_labels=train_labels,
        global_emb=emb_tr,
        fisher_vectors=fv_tr,
        keypoints=train_keypoints,
        descriptors=train_dict,
        calib_ids=args.calib_ids,
        calibration_method=args.calibration_method,
        use_lightglue=args.use_lightglue,
        method=gv_features,
        fusion_signals=args.fusion_signals,
        gv_matcher=gv_matcher,
    )
    preds = classify_test_images_late_fusion(
        test_global_emb=emb_te,
        train_global_emb=emb_tr,
        test_fisher=fv_te,
        train_fisher=fv_tr,
        test_keypoints=test_keypoints,
        train_keypoints=train_keypoints,
        test_descriptors=test_dict,
        train_descriptors=train_dict,
        train_labels=train_labels,
        calibrators=calibrators,
        top_n=5,
        shortlist_size=UNION_CANDIDATES,
        use_lightglue=args.use_lightglue,
        method=gv_features,
        gv_matcher=gv_matcher,
        image_paths=image_paths,
        fusion_signals=args.fusion_signals,
        test_labels=test_labels,
        debug=args.debug,
        dataset_name=dataset_name,
        calibration_method=args.calibration_method,
        calib_ids=args.calib_ids,
    )

    metrics = evaluate_predictions(preds, test_labels)
    if md_meta is not None:
        metrics["md_trained_on"] = md_meta["trained_on"]
        metrics["md_split_type"] = md_meta["split_type"]
        metrics["md_random_split"] = md_meta["random_split"]

    if use_cuda:
        torch.cuda.synchronize()

    runtime_sec = time.time() - start_eval_time
    metrics["eval_runtime_sec"] = runtime_sec

    if args.save_eval:
        save_evaluation_results(metrics, ds_tag, tag=tag)
        row = {
            "Dataset": dataset_name,
            "Training Examples": len(df_train.index),
            "Num Classes": df_train["identity"].nunique(),
            "Method": method_tag if use_fisher_signal else "N/A",
            "Use Fisher": use_fisher_signal,
            "Use Global Embedding": args.use_global_embedding,
            "Embedding Model": args.embedding_model if args.use_global_embedding else "None",
            "Global Checkpoint": getattr(args, "global_ckpt", "") if args.use_global_embedding else "",
            "Use GV": use_gv_signal,
            "Geom. Candidates": GEOMETRIC_CANDIDATES,
            "Calibration Method": args.calibration_method,
            "Calibration IDs": int(args.calib_ids),
            "Run Time (minutes)": round((float(metrics["eval_runtime_sec"]) / 60), 2),
            "Accuracy": round(float(metrics["accuracy"]), 4),
            "Top-5 Accuracy": round(float(metrics["top_n_accuracy"]), 4),
            "F-1 Score": round(float(metrics["classification_metrics"]["weighted avg"]["f1-score"]), 4),
        }
        if md_meta is not None:
            row["MD Trained On"] = md_meta["trained_on"]
            row["MD Split Type"] = md_meta["split_type"]
            row["MD Random Split"] = md_meta["random_split"]
        save_count_results_wrapper(row, EVAL_RESULTS_XLSX)


def run_query_visualization_for_dataset(
    dataset_name: str,
    df_raw: pd.DataFrame,
    *,
    args,
    train_use_fisher: bool,
    method: str,
    method_tag: str,
    ensemble_methods: list[str],
    ensamble_weights: list[float],
    gv_features: str,
    gv_matcher: str,
    seg_tag: str,
) -> dict:
    """Run the classification funnel for one or more queries and export visualization assets."""
    print(f"Visualization mode selected. Dataset: {dataset_name}")
    md_meta = MD_DATASET_SPLITS.get(str(dataset_name).strip().lower(), None)
    if md_meta is not None:
        print(
            "MD split metadata: "
            f"trained_on={md_meta['trained_on']}, "
            f"split_type={md_meta['split_type']}, "
            f"random_split={md_meta['random_split']}"
        )

    feature_method = method
    use_md_baseline_split = bool(md_meta is not None and md_meta.get("trained_on"))
    if use_md_baseline_split:
        print("[MD] Using MegaDescriptor baseline split (split_md_baseline).")

    df, csv_path, _ = preprocessing.prepare_processed_dataset(
        dataset_name,
        df_raw,
        remove_background=args.remove_background,
        use_mantiuk=args.use_mantiuk,
        require_processed_paths=True,
    )

    if use_md_baseline_split:
        repo_root = Path(__file__).resolve().parent
        baseline_root = repo_root / "test-scripts" / "wildlife-tools-data"
        metadata_root = baseline_root / "metadata" / "datasets"

        def _resolve_baseline_dir(root: Path, name: str) -> Path | None:
            direct = root / name
            if direct.exists():
                return direct
            name_lower = str(name).lower()
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == name_lower:
                    return child
            return None

        def _normalize_split_path(value: str) -> str:
            path = str(value).strip().replace("\\", "/")
            if path.startswith("./"):
                path = path[2:]
            if path.lower().startswith("images/"):
                parts = path.split("/", 2)
                if len(parts) >= 3:
                    path = parts[2]
            return path.lower()

        baseline_dir = _resolve_baseline_dir(metadata_root, str(dataset_name))
        if baseline_dir is None:
            print(f"[ERROR] MD baseline metadata not found for dataset: {dataset_name}")
            sys.exit(1)
        metadata_csv = baseline_dir / "metadata.csv"
        if not metadata_csv.exists():
            print(f"[ERROR] Missing MD baseline metadata CSV: {metadata_csv}")
            sys.exit(1)

        df_baseline = pd.read_csv(metadata_csv, dtype={"image_id": str, "identity": str})
        unnamed_cols = [c for c in df_baseline.columns if str(c).lower().startswith("unnamed")]
        if unnamed_cols:
            df_baseline = df_baseline.drop(columns=unnamed_cols)
        if "path" not in df_baseline.columns or "split" not in df_baseline.columns:
            print(f"[ERROR] MD baseline metadata missing required columns in: {metadata_csv}")
            sys.exit(1)

        baseline_split_map = dict(
            zip(
                df_baseline["path"].astype(str).map(_normalize_split_path),
                df_baseline["split"].astype(str),
            )
        )

        if "path" not in df.columns:
            print("[ERROR] Dataset metadata missing 'path' column; cannot apply MD baseline split.")
            sys.exit(1)

        key_series = df["path"].astype(str).map(_normalize_split_path)
        mapped_split = key_series.map(baseline_split_map)
        matched = int(mapped_split.notna().sum())
        total = int(len(df))
        if matched == 0:
            print(
                f"[ERROR] Could not match any paths to MD baseline metadata for dataset: {dataset_name}."
            )
            sys.exit(1)
        if matched != total:
            print(
                f"[WARN] MD baseline split matched {matched}/{total} rows; leaving {total - matched} unchanged."
            )

        existing_split = (
            df["split"].astype(str).str.lower()
            if "split" in df.columns
            else pd.Series(["train"] * len(df), index=df.index)
        )
        df["split_md_baseline"] = mapped_split.fillna(existing_split).astype(str).str.lower()
        if md_meta is not None:
            df["split_type"] = md_meta.get("split_type")
            df["trained_on"] = bool(md_meta.get("trained_on"))
            df["random_split"] = bool(md_meta.get("random_split"))
        df.to_csv(csv_path, index=False)

    if ("split" in df.columns) or (use_md_baseline_split and "split_md_baseline" in df.columns):
        print("Using predefined split in WildlifeReID10k.")
        split_col = "split"
        if use_md_baseline_split and "split_md_baseline" in df.columns:
            split_col = "split_md_baseline"
        train_mask = df[split_col].astype(str).str.lower() != "test"
        test_mask = df[split_col].astype(str).str.lower() == "test"
        df_train, df_test = df[train_mask], df[test_mask]

        if args.split_type == "closed":
            print("Using closed set split.")
            train_identities = set(df_train["identity"].unique())
            df_test = df_test[df_test["identity"].isin(train_identities)].copy()
        splits.analyze_split(df, df_train.index, df_test.index)
    else:
        print("[ERROR] Missing 'split' information in dataset metadata.")
        sys.exit(1)

    def _resolve_query_id(query_value: str) -> str | None:
        token = str(query_value).strip()
        if not token:
            return None
        ids = df_test["image_id"].astype(str).tolist()
        ids_set = set(ids)
        if token in ids_set:
            return token
        token_stem = Path(token).stem.lower()
        for iid in ids:
            if str(iid).lower() == token_stem:
                return iid
        if "path" in df_test.columns:
            for _, row in df_test.iterrows():
                p = str(row.get("path", ""))
                if Path(p).stem.lower() == token_stem:
                    return str(row["image_id"])
        return None

    raw_queries: list[str] = []
    if getattr(args, "query_images", None):
        raw_queries.extend([str(x) for x in args.query_images if str(x).strip()])
    if getattr(args, "query_image", None):
        raw_queries.append(str(args.query_image))

    # Accept both space-separated and comma-separated forms.
    query_tokens: list[str] = []
    for q in raw_queries:
        parts = [p.strip() for p in str(q).split(",")]
        query_tokens.extend([p for p in parts if p])

    if not query_tokens:
        print("[ERROR] No query images were provided.")
        sys.exit(1)

    resolved_ids: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for token in query_tokens:
        rid = _resolve_query_id(token)
        if rid is None:
            unresolved.append(token)
            continue
        if rid not in seen:
            seen.add(rid)
            resolved_ids.append(rid)

    if unresolved:
        sample = ", ".join(df_test["image_id"].astype(str).head(10).tolist())
        print(
            f"[ERROR] The following query items were not found in test split for {dataset_name}: "
            f"{', '.join(unresolved)}. Sample test ids: {sample}"
        )
        sys.exit(1)
    if not resolved_ids:
        print("[ERROR] No valid query ids were resolved.")
        sys.exit(1)
    print(f"[VIS] Resolved {len(resolved_ids)} query image(s): {', '.join(resolved_ids)}")

    train_image_items = get_image_paths(df_train, args.remove_background)
    test_image_items = get_image_paths(df_test, args.remove_background)
    train_paths_map = dict(zip(df_train["image_id"].astype(str), train_image_items))
    test_paths_map = dict(zip(df_test["image_id"].astype(str), test_image_items))

    image_paths = None
    if gv_matcher == "loftr":
        image_paths = dict(
            zip(df["image_id"].astype(str), get_image_paths(df, args.remove_background))
        )

    ds_tag = dataset_name
    base_dir = f"./data/{ds_tag}"
    os.makedirs(base_dir, exist_ok=True)

    pca_dim = int(getattr(args, "pca_dim", N_COMPONENTS_PCA))
    # Always include PCA dim to avoid reusing old cached PCA/GMM/FVs.
    pca_cache_suffix = f"{seg_tag}_pca{pca_dim}"

    fusion_signals_set = set(args.fusion_signals or [])
    use_fisher_signal = bool(train_use_fisher)
    use_gv_signal = "gv" in fusion_signals_set

    train_dict, test_dict = {}, {}
    train_keypoints, test_keypoints = {}, {}
    fv_tr, fv_te = {}, {}
    if use_fisher_signal:
        if feature_method == "ensamble":
            methods = ensemble_methods
            for m in methods:
                dir_tr = feature_descriptor_dir(base_dir, m, "train", seg_tag)
                dir_te = feature_descriptor_dir(base_dir, m, "test", seg_tag)
                ensure_local_descriptors(train_image_items, m, dir_tr)
                ensure_local_descriptors(test_image_items, m, dir_te)

            fv_tr_list = []
            fv_te_list = []
            for m in methods:
                dir_tr = feature_descriptor_dir(base_dir, m, "train", seg_tag)
                dir_te = feature_descriptor_dir(base_dir, m, "test", seg_tag)
                train_dict_m = load_descriptors(os.path.join(dir_tr, "descriptors.h5"))
                test_dict_m = load_descriptors(os.path.join(dir_te, "descriptors.h5"))
                pca_m, gmm_m, fv_tr_m = load_or_train_fisher_vectors(
                    ds_tag=ds_tag,
                    method_name=m,
                    cache_suffix=pca_cache_suffix,
                    descriptors=train_dict_m,
                    pca_dim=pca_dim,
                )
                fv_te_m = compute_fisher_vectors(test_dict_m, pca_m, gmm_m)
                fv_tr_list.append(fv_tr_m)
                fv_te_list.append(fv_te_m)

            fv_tr = combine_fisher_vectors(fv_tr_list, ensamble_weights)
            fv_te = combine_fisher_vectors(fv_te_list, ensamble_weights)
        else:
            dir_tr = feature_descriptor_dir(base_dir, feature_method, "train", seg_tag)
            dir_te = feature_descriptor_dir(base_dir, feature_method, "test", seg_tag)
            ensure_local_descriptors(train_image_items, feature_method, dir_tr)
            ensure_local_descriptors(test_image_items, feature_method, dir_te)

            train_dict = load_descriptors(os.path.join(dir_tr, "descriptors.h5"))
            test_dict = load_descriptors(os.path.join(dir_te, "descriptors.h5"))
            train_keypoints = load_keypoints(os.path.join(dir_tr, "keypoints.h5"))
            test_keypoints = load_keypoints(os.path.join(dir_te, "keypoints.h5"))

            pca, gmm, fv_tr = load_or_train_fisher_vectors(
                ds_tag=ds_tag,
                method_name=feature_method,
                cache_suffix=pca_cache_suffix,
                descriptors=train_dict,
                pca_dim=pca_dim,
            )
            fv_te = compute_fisher_vectors(test_dict, pca, gmm)

    if use_gv_signal and (
        not train_dict or not test_dict or not train_keypoints or not test_keypoints
    ):
        gv_dir_tr = feature_descriptor_dir(base_dir, gv_features, "train", seg_tag)
        gv_dir_te = feature_descriptor_dir(base_dir, gv_features, "test", seg_tag)
        ensure_local_descriptors(train_image_items, gv_features, gv_dir_tr)
        ensure_local_descriptors(test_image_items, gv_features, gv_dir_te)
        train_dict = load_descriptors(os.path.join(gv_dir_tr, "descriptors.h5"))
        test_dict = load_descriptors(os.path.join(gv_dir_te, "descriptors.h5"))
        train_keypoints = load_keypoints(os.path.join(gv_dir_tr, "keypoints.h5"))
        test_keypoints = load_keypoints(os.path.join(gv_dir_te, "keypoints.h5"))

    if args.use_global_embedding:
        print("Extracting global embeddings...")
        global_ckpt = getattr(args, "global_ckpt", "") or None
        emb_model_label = global_embedding_cache_label(args.embedding_model, global_ckpt)
        emb_tr_path = f"{base_dir}/global_embeddings_train_{emb_model_label}_{seg_tag}.pkl"
        emb_te_path = f"{base_dir}/global_embeddings_test_{emb_model_label}_{seg_tag}.pkl"
        emb_tr = load_or_build_global_embeddings(
            train_paths_map,
            emb_tr_path,
            model_name=args.embedding_model,
            checkpoint_path=global_ckpt,
        )
        emb_te = load_or_build_global_embeddings(
            test_paths_map,
            emb_te_path,
            model_name=args.embedding_model,
            checkpoint_path=global_ckpt,
        )
    else:
        emb_tr, emb_te = {}, {}

    if not emb_tr and not fv_tr:
        print(
            "[ERROR] Visualization mode requires at least one Tier-1 signal. "
            "Enable --use_global_embedding and/or include 'fisher' in --fusion_signals."
        )
        sys.exit(1)

    train_labels = dict(zip(df_train["image_id"].astype(str), df_train["identity"].astype(str)))
    calibrators = train_calibrators_two_stage(
        train_labels=train_labels,
        global_emb=emb_tr,
        fisher_vectors=fv_tr,
        keypoints=train_keypoints,
        descriptors=train_dict,
        calib_ids=args.calib_ids,
        calibration_method=args.calibration_method,
        use_lightglue=args.use_lightglue,
        method=gv_features,
        fusion_signals=args.fusion_signals,
        gv_matcher=gv_matcher,
    )

    df_train_ids = df_train["image_id"].astype(str)
    train_processed_paths = dict(zip(df_train_ids, get_image_paths(df_train, args.remove_background)))
    out_root = Path(args.assets_out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_query: list[dict] = []

    for query_id in resolved_ids:
        print(f"[VIS] Query image id: {query_id}")
        single_result = classify_single_image_late_fusion(
            test_id=query_id,
            test_global_emb=emb_te,
            train_global_emb=emb_tr,
            test_fisher=fv_te,
            train_fisher=fv_tr,
            test_keypoints=test_keypoints,
            train_keypoints=train_keypoints,
            test_descriptors=test_dict,
            train_descriptors=train_dict,
            train_labels=train_labels,
            calibrators=calibrators,
            shortlist_size=UNION_CANDIDATES,
            use_lightglue=args.use_lightglue,
            method=gv_features,
            gv_matcher=gv_matcher,
            image_paths=image_paths,
            fusion_signals=args.fusion_signals,
            top_n=5,
        )

        row_q = df_test[df_test["image_id"].astype(str) == str(query_id)].head(1)
        if row_q.empty:
            print(f"[ERROR] Could not resolve query row for id={query_id}")
            sys.exit(1)
        query_processed_path = get_image_paths(row_q, args.remove_background)[0]

        query_raw_path = None
        if "processed_path" in row_q.columns:
            query_raw_path = str(row_q.iloc[0]["processed_path"]) + f"/{query_id}.jpg"

        query_segmented_path = None
        if "processed_path_segmented" in row_q.columns:
            seg_base = row_q.iloc[0].get("processed_path_segmented", None)
            if isinstance(seg_base, str) and seg_base.strip():
                query_segmented_path = seg_base + f"/{query_id}.jpg"

        query_class = None
        if "identity" in row_q.columns:
            cls_val = row_q.iloc[0].get("identity", None)
            if cls_val is not None:
                query_class = str(cls_val)

        query_out_dir = out_root / str(query_id)
        manifest = build_assets_from_funnel(
            out_dir=query_out_dir,
            query_id=query_id,
            query_class=query_class,
            query_raw_path=query_raw_path,
            query_processed_path=query_processed_path,
            query_segmented_path=query_segmented_path,
            train_processed_paths=train_processed_paths,
            query_keypoints=np.asarray(test_keypoints.get(query_id, np.empty((0, 2), dtype=np.float32))),
            query_descriptors=np.asarray(test_dict.get(query_id, np.empty((0, 0), dtype=np.float32))),
            query_global_embedding=np.asarray(emb_te.get(query_id, np.empty((0,), dtype=np.float32))),
            query_fisher_vector=np.asarray(fv_te.get(query_id, np.empty((0,), dtype=np.float32))),
            train_global_embeddings=emb_tr,
            train_fisher_vectors=fv_tr,
            train_keypoints=train_keypoints,
            train_descriptors=train_dict,
            result=single_result,
            gv_matcher=gv_matcher,
            gv_features=gv_features,
            image_paths=image_paths,
            top_k=int(args.assets_top_k),
            panel_size=int(args.assets_panel_size),
            overview_mode=bool(getattr(args, "assets_overview_mode", False)),
        )
        manifest_path = query_out_dir / "assets_manifest.json"
        print(f"[VIS] Predicted class ({query_id}): {single_result.get('predicted_class')}")
        print(f"[VIS] Assets written to: {query_out_dir}")
        print(f"[VIS] Manifest: {manifest_path}")
        per_query.append(
            {
                "query_id": str(query_id),
                "predicted_class": single_result.get("predicted_class"),
                "output_dir": str(query_out_dir),
                "manifest_path": str(manifest_path),
            }
        )

    batch_manifest = {
        "dataset": str(dataset_name),
        "num_queries": int(len(per_query)),
        "queries": per_query,
    }
    batch_manifest_path = out_root / "batch_manifest.json"
    batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    print(f"[VIS] Batch manifest: {batch_manifest_path}")
    return batch_manifest
