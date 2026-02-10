import argparse
import hashlib
import json
import sys
import os
import pickle
import time

import torch

from constants import *
from feature_extraction import (
    get_image_paths,
    get_segmentation_tag,
)
from feature_aggregation import (
    load_descriptors,
    ensure_local_descriptors,
    feature_descriptor_dir,
    load_or_train_fisher_vectors,
)
from global_embedding import load_or_build_global_embeddings
from nested_importance_sampling import nested_importance_sampling
from nested_importance_sampling import CountCalibrators
from preprocessing import prepare_processed_dataset
from train import run_query_visualization_for_dataset, run_training_for_dataset
from utility_functions import (
    load_dataset,
    combine_fisher_vectors,
    save_count_results_wrapper,
)

ensamble_weights = ENSEMBLE_WEIGHTS


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--count', action = 'store_true', help='Estimate the population size using Nested Importance Sampling')
    parser.add_argument(
        '--visualize_query_pipeline',
        action='store_true',
        help='Run classification funnel for one query and export stage assets.',
    )
    parser.add_argument(
        '--ds',
        nargs='+',
        help=(
            "Specify one or more datasets (e.g., ATRW BelugaID). "
            "Use 'full' to train on all datasets."
        ),
        default=['full'],
    )
    parser.add_argument('--save_eval', action='store_true', help='Save evaluation metrics during training', default = True)
    parser.add_argument('--use_mantiuk', action='store_true', help='Use Mantiuk tone mapping during preprocessing')
    parser.add_argument('--remove_background', action='store_true', help='Remove background during preprocessing')
    parser.add_argument('--version', type=str, default='1', help='Identifier for the current method version')
    parser.add_argument(
        '--method',
        type=str,
        nargs='+',
        choices=['disk', 'superpoint', 'aliked', 'keynet_hardnet', 'lightglue', 'ensamble'],
        default=['disk'],
        help=(
            "Feature extraction method(s) to use. "
            "Pass multiple (e.g., --method disk aliked) to ensemble those; "
            "'ensamble' uses all three (disk, superpoint, aliked)."
        ),
    )
    parser.add_argument('--use_lightglue', action ='store_true', help='Use LightGlue for feature matching when performing geometric verification', default=True)
    parser.add_argument('--gv_matcher', type=str, choices=['ratio', 'lightglue', 'loftr'],
                        default=None, help='Matcher for geometric verification (ratio, lightglue, loftr)')
    parser.add_argument(
        '--gv_features',
        type=str,
        choices=['disk', 'superpoint', 'aliked'],
        default=None,
        help="Local feature type for GV when using LightGlue (defaults to --method unless ensemble).",
    )
    parser.add_argument('--num_vertices', type=int, default=10, help='Vertices sampled in Nested-IS')
    parser.add_argument('--num_neighbors', type=int, default=100, help='Neighbors sampled per vertex in Nested-IS')
    parser.add_argument('--save_count', action='store_true', help='Save population estimation results to XLSX')
    parser.add_argument(
        '--label_error_rate',
        type=float,
        default=0.0,
        help='Fraction of pair labelings to flip during counting',
    )
    parser.add_argument(
        '--count_confirm_same_votes',
        type=int,
        default=1,
        help=(
            "Robustness to human mistakes in count mode. "
            "Declare a pair as 'same' only if the oracle returns 'same' K times in a row "
            "(only re-asks when the first vote is 'same'). "
            "Set to 3 for strict confirmation; set to 1 to disable."
        ),
    )
    parser.add_argument(
        '--count_proposal_mode',
        type=str,
        default='calibrated',
        choices=['calibrated', 'power'],
        help="HITL-NIS proposal mode: calibrated probabilities or power-transformed scores.",
    )
    parser.add_argument(
        '--count_cal_pairs',
        type=int,
        default=2000,
        help="Number of GT-simulated calibration pairs for per-signal score calibration in count mode.",
    )
    parser.add_argument(
        '--count_cal_shortlist',
        type=int,
        default=150,
        help="Shortlist size used when sampling hard negatives for count calibration.",
    )
    parser.add_argument(
        '--count_cal_negs_per_query',
        type=int,
        default=400,
        help="Number of hard negatives per calibration query image.",
    )
    parser.add_argument(
        '--count_force_recalibrate',
        action='store_true',
        help="Ignore cached count calibrators and retrain.",
    )
    parser.add_argument(
        '--count_skip_calibration',
        action='store_true',
        help="Skip count calibrator training/loading and use raw global/Fisher similarities.",
    )
    parser.add_argument('--use_global_embedding', action='store_true', help='Use global CNN')
    parser.add_argument(
        '--use_fisher',
        action='store_true',
        help='Use PCA/GMM/Fisher-vector features (count mode; training follows --fusion_signals).',
    )

    parser.add_argument(
        '--embedding_model',
        type=str,
        default='megadescriptor-l-384',
        choices=[
            'resnet50',
            'megadescriptor-l-384',
        ],
        help='Model for global embeddings',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible counting')
    parser.add_argument('--calibration_method', type=str, default='isotonic_pchip', choices=['isotonic_pchip', 'logistic', 'isotonic'], help='Calibration method to use')
    parser.add_argument('--fusion_signals', type=str, nargs = '+', default=['global', 'fisher', 'gv'], choices=['global', 'fisher', 'gv'], help='Signals to fuse')
    parser.add_argument('--calib_size', type=int, default=200, help='Size of the calibration set')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument(
        '--query_image',
        type=str,
        default=None,
        help='Query image id or filename (used with --visualize_query_pipeline).',
    )
    parser.add_argument(
        '--query_images',
        type=str,
        nargs='+',
        default=None,
        help='Multiple query image ids/filenames (space and/or comma separated).',
    )
    parser.add_argument(
        '--assets_out_dir',
        type=str,
        default='docs/Final Thesis/Figures/pipeline_assets',
        help='Output directory for query pipeline assets.',
    )
    parser.add_argument(
        '--assets_top_k',
        type=int,
        default=8,
        help='Number of candidates shown per ranking strip in exported assets.',
    )
    parser.add_argument(
        '--assets_panel_size',
        type=int,
        default=320,
        help='Square panel size (pixels) for generated asset tiles.',
    )
    parser.add_argument(
        '--assets_overview_mode',
        action='store_true',
        help='Generate simplified overview assets with larger strip titles/text and minimal annotations.',
    )
    args = parser.parse_args()
    seg_tag = get_segmentation_tag(args.remove_background)
    args.split_type = 'closed'
    dataset_name = args.ds
    raw_methods = args.method
    method_list = list(raw_methods) if isinstance(raw_methods, (list, tuple)) else [raw_methods]
    method_list = [m.lower() for m in method_list]
    is_ensemble = False
    ensemble_methods = []
    method_tag = None
    if "ensamble" in method_list:
        if len(method_list) > 1:
            print("Ensamble specified with other methods. Using full ensemble (disk, superpoint, aliked).")
        is_ensemble = True
        ensemble_methods = ["disk", "superpoint", "aliked"]
        method = "ensamble"
        method_tag = "ensamble"
    elif len(method_list) > 1:
        invalid = [m for m in method_list if m not in {"disk", "superpoint", "aliked"}]
        if invalid:
            print(f"[ERROR] Unsupported ensemble methods: {invalid}.")
            sys.exit(1)
        is_ensemble = True
        ensemble_methods = method_list
        method = "ensamble"
        method_tag = "+".join(ensemble_methods)
        ensamble_weights = [0.5, 0.5]
    else:
        method = method_list[0]
        method_tag = method
        ensemble_methods = [method]
    gv_features = args.gv_features
    #use_splitter = False

    gv_matcher = args.gv_matcher
    if gv_matcher is None:
        gv_matcher = "lightglue" if args.use_lightglue else "ratio"
    gv_matcher = gv_matcher.lower()

    if not is_ensemble:
        if gv_features and gv_features != method:
            print(
                "[GV] --gv_features ignored because --method is not ensamble; "
                "using --method."
            )
        gv_features = "aliked" if method == "lightglue" else method
    else:
        if gv_features is None:
            gv_features = "disk"
        if gv_features == "lightglue":
            gv_features = "aliked"
        if gv_features not in {"disk", "superpoint", "aliked"}:
            print(f"[GV] Unsupported gv_features '{gv_features}', falling back to 'disk'.")
            gv_features = "disk"
    
    #split_type = 'closed_split'
    fusion_signals_set = set(args.fusion_signals or [])
    use_gv_signal = "gv" in fusion_signals_set
    train_use_fisher = "fisher" in fusion_signals_set

    # create a configuration tag for saving evaluation results
    tag = (
        f"rmbkg_{args.remove_background}_tm_{args.use_mantiuk}_{method_tag}"
        f"_PCA_{N_COMPONENTS_PCA}_GMM_{N_COMPONENTS_GMM}"
        f"_gv_{use_gv_signal}_lg_{args.use_lightglue}"
        f"_v{args.version}"
    )

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.train and not dataset_name:
        print("Please specify the dataset to train on using the --ds argument.")
        sys.exit(0)

    if args.visualize_query_pipeline:
        requested = [str(name).strip() for name in dataset_name if str(name).strip()]
        if len(requested) != 1:
            print("[ERROR] --visualize_query_pipeline requires exactly one dataset via --ds <DATASET>.")
            sys.exit(1)
        query_tokens = []
        if args.query_image:
            query_tokens.append(str(args.query_image))
        if args.query_images:
            query_tokens.extend([str(x) for x in args.query_images])
        parsed_tokens = []
        for tok in query_tokens:
            parts = [p.strip() for p in str(tok).split(",")]
            parsed_tokens.extend([p for p in parts if p])
        if not parsed_tokens:
            print(
                "[ERROR] --visualize_query_pipeline requires at least one query via "
                "--query_image or --query_images."
            )
            sys.exit(1)
        # Pass normalized tokens to train.py so it can resolve IDs in dataset test split.
        args.query_images = parsed_tokens

        name = requested[0]
        if name.lower() == "full":
            print("[ERROR] --visualize_query_pipeline requires a concrete dataset name, not 'full'.")
            sys.exit(1)
        df_raw = load_dataset(name)
        run_query_visualization_for_dataset(
            name,
            df_raw,
            args=args,
            train_use_fisher=train_use_fisher,
            method=method,
            method_tag=method_tag,
            ensemble_methods=ensemble_methods,
            ensamble_weights=ensamble_weights,
            gv_features=gv_features,
            gv_matcher=gv_matcher,
            seg_tag=seg_tag,
        )
        sys.exit(0)

    if args.train:
        requested = [str(name).strip() for name in dataset_name if str(name).strip()]
        if not requested:
            print("Please specify the dataset to train on using the --ds argument.")
            sys.exit(0)

        if any(name.lower() == "full" for name in requested):
            df_all = load_dataset("full")
            if "dataset" not in df_all.columns:
                print("Expected 'dataset' column in all_datasets.csv for full mode.")
                sys.exit(1)
            for sub_name, sub_df in df_all.groupby("dataset", sort=True):
                run_training_for_dataset(
                    sub_name,
                    sub_df,
                    args=args,
                    train_use_fisher=train_use_fisher,
                    method=method,
                    method_tag=method_tag,
                    ensemble_methods=ensemble_methods,
                    ensamble_weights=ensamble_weights,
                    gv_features=gv_features,
                    gv_matcher=gv_matcher,
                    seg_tag=seg_tag,
                    tag=tag,
                    use_cuda=use_cuda,
                )
        else:
            for name in requested:
                df_raw = load_dataset(name)
                run_training_for_dataset(
                    name,
                    df_raw,
                    args=args,
                    train_use_fisher=train_use_fisher,
                    method=method,
                    method_tag=method_tag,
                    ensemble_methods=ensemble_methods,
                    ensamble_weights=ensamble_weights,
                    gv_features=gv_features,
                    gv_matcher=gv_matcher,
                    seg_tag=seg_tag,
                    tag=tag,
                    use_cuda=use_cuda,
                )
        sys.exit(0)

    if args.count:
        if not dataset_name or len(dataset_name) != 1:
            print("[ERROR] --count requires exactly one dataset: use `--ds <DATASET>`.")
            sys.exit(1)

        dataset_name = str(dataset_name[0]).strip()
        ds_tag = dataset_name.lower()
        base_dir = f"./data/{ds_tag}"
        os.makedirs(base_dir, exist_ok=True)

        print(f"[COUNT] Dataset: {dataset_name} (dir: {base_dir})")
        print(f"[COUNT] Proposal: {args.count_proposal_mode}")

        start_time = time.time()

        df_raw = load_dataset(dataset_name)
        df, _, output_dir = prepare_processed_dataset(
            dataset_name,
            df_raw,
            remove_background=args.remove_background,
            use_mantiuk=args.use_mantiuk,
            require_processed_paths=True,
            log_prefix="[COUNT]",
        )
        print(f"[COUNT] Output directory: {output_dir}")

        # --- Fisher vectors (optional signal) ---
        fisher_vectors = None
        if args.use_fisher:
            full_image_items = get_image_paths(df, args.remove_background)
            if is_ensemble:
                methods = ensemble_methods
                fv_list = []
                for m in methods:
                    feat_dir = feature_descriptor_dir(base_dir, m, "full", seg_tag)

                    def _load_full_desc(method_name=m, method_dir=feat_dir):
                        ensure_local_descriptors(full_image_items, method_name, method_dir)
                        return load_descriptors(os.path.join(method_dir, "descriptors.h5"))

                    _, _, fv_m = load_or_train_fisher_vectors(
                        ds_tag=ds_tag,
                        method_name=m,
                        cache_suffix=f"{seg_tag}_full",
                        descriptors_loader=_load_full_desc,
                    )
                    fv_list.append(fv_m)
                fisher_vectors = combine_fisher_vectors(fv_list, ENSEMBLE_WEIGHTS)
            else:
                feat_dir_fisher = feature_descriptor_dir(base_dir, method, "full", seg_tag)

                def _load_full_desc_single():
                    ensure_local_descriptors(full_image_items, method, feat_dir_fisher)
                    return load_descriptors(os.path.join(feat_dir_fisher, "descriptors.h5"))

                _, _, fisher_vectors = load_or_train_fisher_vectors(
                    ds_tag=ds_tag,
                    method_name=method,
                    cache_suffix=f"{seg_tag}_full",
                    descriptors_loader=_load_full_desc_single,
                )

        # --- Global embeddings (optional signal) ---
        global_embeddings = None
        if args.use_global_embedding:
            print("[COUNT] Loading/computing global embeddings...")
            image_paths = dict(zip(df["image_id"].astype(str), get_image_paths(df, args.remove_background)))
            emb_path = f"{base_dir}/global_embeddings_count_{args.embedding_model}_{seg_tag}_full.pkl"
            global_embeddings = load_or_build_global_embeddings(
                image_paths,
                emb_path,
                model_name=args.embedding_model,
            )

        if global_embeddings is None and fisher_vectors is None:
            print(
                "[ERROR] Count mode requires at least one cheap base signal. "
                "Enable `--use_global_embedding` and/or `--use_fisher`."
            )
            sys.exit(1)

        # --- Oracle: for now, use GT identity labels to simulate human vetting ---
        if "identity" not in df.columns:
            print("[ERROR] Counting currently requires ground-truth `identity` labels in metadata.")
            sys.exit(1)

        df["image_id"] = df["image_id"].astype(str)
        df["identity"] = df["identity"].astype(str)
        labels = dict(zip(df["image_id"], df["identity"]))

        def oracle(u_id: str, v_id: str) -> int:
            return int(labels.get(u_id) == labels.get(v_id) and u_id != v_id)

        # --- Calibration (WildFusion-style per-signal score → P(match)) ---
        calibrators_bundle = None
        cal_info = {}
        if args.count_proposal_mode == "calibrated" and not args.count_skip_calibration:
            from calibration import train_count_calibrators_gt

            cal_key = {
                "ds_tag": ds_tag,
                "seg_tag": seg_tag,
                "use_global_embedding": bool(args.use_global_embedding),
                "embedding_model": args.embedding_model if args.use_global_embedding else None,
                "use_fisher": bool(args.use_fisher),
                "fisher_method": method_tag if args.use_fisher else None,
                "count_cal_pairs": int(args.count_cal_pairs),
                "count_cal_shortlist": int(args.count_cal_shortlist),
                "count_cal_negs_per_query": int(args.count_cal_negs_per_query),
                "calibration_method": args.calibration_method,
            }
            cal_key_json = json.dumps(cal_key, sort_keys=True, separators=(",", ":"))
            cal_hash = hashlib.sha1(cal_key_json.encode("utf-8")).hexdigest()[:12]
            cal_cache_path = f"{base_dir}/count_calibrators_{cal_hash}.pkl"

            cal_loaded = False
            cal_dict = None
            if os.path.exists(cal_cache_path) and not args.count_force_recalibrate:
                try:
                    with open(cal_cache_path, "rb") as f:
                        payload = pickle.load(f)
                    if payload.get("key") == cal_key:
                        cal_dict = payload.get("calibrators", None)
                        cal_info = payload.get("info", {}) or {}
                        cal_loaded = isinstance(cal_dict, dict) and bool(cal_dict)
                except Exception:
                    cal_loaded = False

            if not cal_loaded:
                print("[COUNT] Training count calibrators (GT simulation)...")
                cal_dict, cal_info = train_count_calibrators_gt(
                    train_labels=labels,
                    image_ids=df["image_id"].astype(str).tolist(),
                    global_emb=global_embeddings,
                    fisher_vectors=fisher_vectors,
                    target_pairs=int(args.count_cal_pairs),
                    shortlist_size=int(args.count_cal_shortlist),
                    n_negatives_per_query=int(args.count_cal_negs_per_query),
                    calibration_method=args.calibration_method,
                    seed=args.seed if args.seed is not None else 42,
                )
                with open(cal_cache_path, "wb") as f:
                    pickle.dump({"key": cal_key, "calibrators": cal_dict, "info": cal_info}, f)
                print(f"[COUNT] Saved count calibrators to: {cal_cache_path}")
            else:
                print(f"[COUNT] Loaded cached count calibrators: {cal_cache_path}")

            calibrators_bundle = CountCalibrators(
                global_cal=cal_dict.get("global") if isinstance(cal_dict, dict) else None,
                fisher_cal=cal_dict.get("fisher") if isinstance(cal_dict, dict) else None,
            )

        # --- Estimate with HITL-NIS (configured proposal mode) ---
        image_ids = df["image_id"].astype(str).tolist()

        estimate, se, stats = nested_importance_sampling(
            global_embeddings,
            fisher_vectors,
            image_ids,
            oracle=oracle,
            proposal_mode=args.count_proposal_mode,
            calibrators=calibrators_bundle,
            n_vertices=args.num_vertices,
            n_neighbors=args.num_neighbors,
            label_error_rate=args.label_error_rate,
            confirm_same_votes=int(args.count_confirm_same_votes),
            seed=args.seed,
            return_stats=True,
        )

        runtime = time.time() - start_time
        ci_low = float(estimate - 1.96 * se)
        ci_high = float(estimate + 1.96 * se)

        print(f"[COUNT] Estimated individuals: {estimate:.2f} ± {se:.2f} (95% CI [{ci_low:.2f}, {ci_high:.2f}])")
        print(stats)

        if args.save_count:
            row = {
                "Dataset": dataset_name,
                "Num Images": int(len(image_ids)),
                "Ground Truth": int(df['identity'].nunique()),
                "Use Global Embedding": bool(args.use_global_embedding),
                "Embedding Model": args.embedding_model if args.use_global_embedding else "None",
                "Use Fisher": bool(args.use_fisher),
                "Fisher Method": method_tag if args.use_fisher else "None",
                "GV Features": gv_features,
                "GV Matcher": gv_matcher,
                "Remove Background": bool(args.remove_background),
                "Proposal Mode": args.count_proposal_mode,
                "Calibration Method": args.calibration_method,
                "Cal Pairs": int(args.count_cal_pairs),
                "Cal Shortlist": int(args.count_cal_shortlist),
                "Cal Negs/Query": int(args.count_cal_negs_per_query),
                "Skip Calibration": bool(args.count_skip_calibration),
                "Num Vertices": int(args.num_vertices),
                "Num Neighbors": int(args.num_neighbors),
                "Label Error Rate": float(args.label_error_rate),
                "Confirm Same Votes": int(args.count_confirm_same_votes),
                "Oracle Calls": int(stats.get("oracle_calls", 0)),
                "Unique Oracle Pairs": int(stats.get("unique_oracle_pairs", 0)),
                "Result": round(float(estimate), 2),
                "Std Error": round(float(se), 4),
                "CI Low (95%)": round(float(ci_low), 2),
                "CI High (95%)": round(float(ci_high), 2),
                "Runtime (minutes)": round(float(runtime) / 60, 2),
            }
            if cal_info:
                row["Cal Pos"] = int(cal_info.get("pos", 0))
                row["Cal Neg"] = int(cal_info.get("neg", 0))
                row["Cal Pos Rate"] = float(cal_info.get("pos_rate", 0.0))
            save_count_results_wrapper(row)

        sys.exit(0)
