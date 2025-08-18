import argparse
from wildlife_datasets import datasets, splits
from wildlife_datasets.datasets import WildlifeReID10k
from patches.elpephants_patch import PatchedELPephants
datasets.ELPephants = PatchedELPephants
from segmentation import segment_dataset, has_segmenter
import preprocessing
import sys
from feature_extraction import get_image_paths, extract_features, extract_features_keynet_hardnet, extract_features_keynet_hardnet_faster, extract_features_lightglue
from feature_aggregation import (
    load_descriptors,
    stack_all_descriptors,
    train_pca,
    train_gmm,
    compute_fisher_vectors,
    load_keypoints,
)
from color_descriptors import (
    compute_color_descriptors,
    standardize as standardize_colors,
    normalize_hsv,
)

from shape_descriptors import (
    compute_shape_descriptors,
    standardize as standardize_shapes,
)

from constants import *
import os
import pickle
from predict import classify_test_images, predict, classify_test_images_with_geometric_verification
from mixture_optimization.block_normalization import (
    apply_zscore_and_l2_train_test,
    fuse_blocks_weighted_concat,
)
from evaluate import evaluate_predictions, save_evaluation_results
#from calibration import calibrate
import pandas as pd
import shutil
import cv2
import sys
#from visualize import visualize_results
import numpy as np
import torch
import time
from nested_importance_sampling import nested_importance_sampling
from utility_functions import (
    load_dataset,
    save_stuff,
    load_stuff,
    save_count_results,
    combine_fisher_vectors,
    save_count_results_wrapper
)

from global_embedding import extract_global_embeddings


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_true')
    parser.add_argument('--count', action = 'store_true', help='Estimate the population size using Nested Importance Sampling')
    parser.add_argument('--ds', type = str, help="Specify the dataset to use (e.g., ATRW, BelugaID, etc.), full to train on the full dataset.",
                        default = 'full')
    parser.add_argument('--image_location', type = str)
    parser.add_argument('--save_eval', action='store_true', help='Save evaluation metrics during training', default = True)
    parser.add_argument('--use_mantiuk', action='store_true', help='Use Mantiuk tone mapping during preprocessing')
    parser.add_argument('--remove_background', action='store_true', help='Remove background during preprocessing')
    parser.add_argument('--version', type=str, default='1', help='Identifier for the current method version')
    parser.add_argument('--method', type = str, default = 'disk', help='Feature extraction method to use')
    parser.add_argument('--use_geometric_verification', action='store_true', help='Use geometric verification during prediction', default=False)
    parser.add_argument('--use_lightglue', action ='store_true', help='Use LightGlue for feature matching when performing geometric verification', default=True)
    parser.add_argument('--use_color', action='store_true', help='Use colour descriptors (HSV histogram and Lab moments)')
    parser.add_argument('--num_vertices', type=int, default=100, help='Vertices sampled in Nested-IS')
    parser.add_argument('--num_neighbors', type=int, default=10, help='Neighbors sampled per vertex in Nested-IS')
    parser.add_argument('--save_count', action='store_true', help='Save population estimation results to XLSX')
    parser.add_argument('--label_error_rate', type=float, default=0.0,
                        help='Fraction of pair labelings to flip during counting')
    parser.add_argument('--gv_threshold', type=float, default=0.75,
                        help='Geometric verification distance threshold')
    parser.add_argument('--automated_mode', action='store_true', default=False,
                        help='Use fully automated counting without (fake) human labels')
    parser.add_argument('--gv_method', type=str, default='RANSAC', choices=['RANSAC', 'MAGSAC'],
                        help='Geometric verification method to use (RANSAC or MAGSAC)')
    parser.add_argument('--use_global_embedding', action='store_true', help='Use global CNN/Transformer embeddings')
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='resnet50',
        choices=['resnet50', 'megadescriptor-l-384'],
        help='Model for global embeddings',
    )
    parser.add_argument('--use_shape', action='store_true', help='Use shape descriptors based on animal contours')
    parser.add_argument('--w_fisher', type=float, default=3.0, help='Weight for Fisher vectors during fusion')
    parser.add_argument('--w_color', type=float, default=1.0, help='Weight for colour descriptors during fusion')
    parser.add_argument('--w_shape', type=float, default=1.0, help='Weight for shape descriptors during fusion')
    parser.add_argument('--w_global', type=float, default=1.0, help='Weight for global embeddings during fusion')


    args = parser.parse_args()
    args.split_type = 'closed'
    dataset_name = args.ds
    method = args.method
    #use_splitter = False
    
    # Set the geometric verification method
    #import constants
    gv_method = args.gv_method
    
    already_trained = False
    #split_type = 'closed_split'

    # create a configuration tag for saving evaluation results
    tag = (
        f"rmbkg_{args.remove_background}_tm_{args.use_mantiuk}_{method}"
        f"_PCA_{N_COMPONENTS_PCA}_GMM_{N_COMPONENTS_GMM}"
        f"_gv_{args.use_geometric_verification}_lg_{args.use_lightglue}"
        f"_color_{args.use_color}_v{args.version}"
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

    if args.predict and not dataset_name:
        print("Please specify the dataset to use for prediction with the --ds argument.")
        sys.exit(0)
    
    if args.train:
        dataset_name = args.ds
        print(f"Training mode selected. Using dataset: {args.ds}")

        print('training...')
        #if not os.path.isdir(f"./data/{args.ds}"):
            #datasets.ATRW.get_data(f'./data/ATRW/')
        #    datasets.__dict__[dataset_name].get_data(f"./data/{dataset_name}/")

        #dataset = datasets.ATRW('./data/ATRW')
        #dataset = datasets.__dict__[dataset_name](f"./data/{dataset_name}/")
        #df = dataset.df
        #df['image_id'] = df['image_id'].astype(str)
        
        df_raw = load_dataset(dataset_name)
        df_raw["image_id"] = df_raw["image_id"].astype(str)
        
        start_eval_time = time.time()
        # ── Pre‑processing (tone map / background removal) ──
        if dataset_name.lower() == "full":
            processed_frames = []
        #    for sub_name, sub_df in df_raw.groupby("dataset"):
        #        sub_dir = f"./data/{sub_name}"
        #        os.makedirs(sub_dir, exist_ok=True)
        #        csv_path = f"{sub_dir}/processed_metadata.csv"
        #        if not os.path.exists(csv_path):
        #            processed = preprocessing.preprocess_dataset(
        #                sub_df.copy(),
        #                f"{sub_dir}/segmented_dataset/",
        #                sub_name,
        #                use_mantiuk=args.use_mantiuk,
        #                remove_background=args.remove_background,
        #            )
        #            processed.to_csv(csv_path, index=False)
        #        else:
        #            processed = pd.read_csv(csv_path)
        #        processed_frames.append(processed)
        #    df = pd.concat(processed_frames, ignore_index=True)
        else:
            sub_dir = f"./data/{dataset_name}"
            os.makedirs(sub_dir, exist_ok=True)

            

            csv_path = f"{sub_dir}/processed_metadata.csv"
            dataset_path = f"{sub_dir}/dataset"
            segmented_dataset_path = f"{sub_dir}/segmented_dataset"
            if not os.path.exists(csv_path) or ((not os.path.exists(dataset_path) and not args.remove_background) or (not os.path.exists(segmented_dataset_path) and args.remove_background)):
                
                
                if has_segmenter(dataset_name):
                    df = segment_dataset(
                        df_raw.copy(),
                        f"{sub_dir}/segmented_dataset/",
                        dataset_name,
                        use_mantiuk=args.use_mantiuk,
                    )
                    args.remove_background = True
                else:
                    # Keep different folders for segmented and unsegmented dataset
                    output_dir_preprocess = 'dataset'
                    if args.remove_background:
                        output_dir_preprocess = 'segmented_dataset'
                    df = preprocessing.preprocess_dataset(
                        df_raw.copy(),
                        f"{sub_dir}/{output_dir_preprocess}/",
                        dataset_name,
                        use_mantiuk=args.use_mantiuk,
                        remove_background=args.remove_background,
                    )
                df.to_csv(csv_path, index=False)
            else:
                df = pd.read_csv(csv_path)

        df["image_id"] = df["image_id"].astype(str)
        
        if "split" in df.columns:
            print("Using predefined split in WildlifeReID10k.")
            train_mask = df["split"].str.lower() != "test"
            test_mask = df["split"].str.lower() == "test"
            df_train, df_test = df[train_mask], df[test_mask]
            
            if args.split_type == 'closed':
                print('Using closed set split.')
                # After loading the dataset splits
                train_identities = set(df_train["identity"].unique())
                test_identities = set(df_test["identity"].unique())

                # Filter test set to only include identities seen in training
                df_test = df_test[df_test["identity"].isin(train_identities)].copy()
            splits.analyze_split(df, df_train.index, df_test.index)
            
        ds_tag = dataset_name
        base_dir = f"./data/{ds_tag}"
        os.makedirs(base_dir, exist_ok=True)
        
        # ── Feature extraction ──
        if method == 'ensamble':
            methods = ['disk', 'keynet_hardnet', 'lightglue']
            for m in methods:
                dir_tr = f"{base_dir}/feature_descriptors_train_{m}/"
                dir_te = f"{base_dir}/feature_descriptors_test_{m}/"
                if not os.path.isdir(dir_tr):
                    if m == 'disk':
                        extract_features(get_image_paths(df_train, args.remove_background), MODEL_PATH, dir_tr)
                        extract_features(get_image_paths(df_test, args.remove_background), MODEL_PATH, dir_te)
                    elif m == 'keynet_hardnet':
                        extract_features_keynet_hardnet_faster(get_image_paths(df_train, args.remove_background), dir_tr)
                        extract_features_keynet_hardnet_faster(get_image_paths(df_test, args.remove_background), dir_te)
                    elif m == 'lightglue':
                        extract_features_lightglue(get_image_paths(df_train, args.remove_background), dir_tr)
                        extract_features_lightglue(get_image_paths(df_test, args.remove_background), dir_te)

            fv_tr_list = []
            fv_te_list = []
            for m in methods:
                train_dict_m = load_descriptors(f"{base_dir}/feature_descriptors_train_{m}/descriptors.h5")
                test_dict_m = load_descriptors(f"{base_dir}/feature_descriptors_test_{m}/descriptors.h5")
                if m == 'disk':
                    train_dict = train_dict_m
                    test_dict = test_dict_m
                    train_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_train_{m}/keypoints.h5")
                    test_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_test_{m}/keypoints.h5")
                desc_tr_m = stack_all_descriptors(train_dict_m)
                pca_path_m = f"{base_dir}/pca_model_{m}.pkl"
                gmm_path_m = f"{base_dir}/gmm_model_{m}.pkl"
                fv_path_m  = f"{base_dir}/fisher_vectors_{m}.pkl"
                if os.path.exists(pca_path_m) and os.path.exists(gmm_path_m) and os.path.exists(fv_path_m):
                    pca_m, gmm_m, fv_tr_m = load_stuff(pca_path_m, gmm_path_m, fv_path_m)
                    fv_te_m = compute_fisher_vectors(test_dict_m, pca_m, gmm_m)
                else:
                    pca_m = train_pca(desc_tr_m)
                    gmm_m = train_gmm(pca_m.transform(desc_tr_m))
                    fv_tr_m = compute_fisher_vectors(train_dict_m, pca_m, gmm_m)
                    fv_te_m = compute_fisher_vectors(test_dict_m, pca_m, gmm_m)
                    save_stuff(pca_m, gmm_m, fv_tr_m,
                        (PCA_PATH.format(ds_tag, m), GMM_PATH.format(ds_tag, m), FISHER_VECTORS.format(ds_tag, m)))
                fv_tr_list.append(fv_tr_m)
                fv_te_list.append(fv_te_m)

            fv_tr = combine_fisher_vectors(fv_tr_list, ENSEMBLE_WEIGHTS)
            fv_te = combine_fisher_vectors(fv_te_list, ENSEMBLE_WEIGHTS)

        else:
            if not os.path.isdir(f"{base_dir}/feature_descriptors_train_{method}/"):
                if method == 'disk':
                    extract_features(get_image_paths(df_train, args.remove_background), MODEL_PATH, f"{base_dir}/feature_descriptors_train_{method}/")
                    extract_features(get_image_paths(df_test, args.remove_background), MODEL_PATH, f"{base_dir}/feature_descriptors_test_{method}/")
                elif method == 'keynet_hardnet':
                    extract_features_keynet_hardnet_faster(get_image_paths(df_train, args.remove_background), f"{base_dir}/feature_descriptors_train_{method}/")
                    extract_features_keynet_hardnet_faster(get_image_paths(df_test, args.remove_background), f"{base_dir}/feature_descriptors_test_{method}/")
                elif method == 'lightglue':
                    extract_features_lightglue(get_image_paths(df_train, args.remove_background), f"{base_dir}/feature_descriptors_train_{method}/")
                    extract_features_lightglue(get_image_paths(df_test, args.remove_background), f"{base_dir}/feature_descriptors_test_{method}/")

            train_dict = load_descriptors(f"{base_dir}/feature_descriptors_train_{method}/descriptors.h5")
            test_dict = load_descriptors(f"{base_dir}/feature_descriptors_test_{method}/descriptors.h5")
            train_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_train_{method}/keypoints.h5")
            test_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_test_{method}/keypoints.h5")
            desc_tr = stack_all_descriptors(train_dict)
            pca_path = f"{base_dir}/pca_model_{method}.pkl"
            gmm_path = f"{base_dir}/gmm_model_{method}.pkl"
            fv_path  = f"{base_dir}/fisher_vectors_{method}.pkl"
            if os.path.exists(pca_path) and os.path.exists(gmm_path) and os.path.exists(fv_path):
                print("Using already trained PCA and GMM models.")
                pca, gmm, fv_tr = load_stuff(pca_path, gmm_path, fv_path)
                fv_te = compute_fisher_vectors(test_dict, pca, gmm)
                print("Fisher vectors computed for test set.")
            else:
                pca = train_pca(desc_tr)
                gmm = train_gmm(pca.transform(desc_tr))
                fv_tr = compute_fisher_vectors(train_dict, pca, gmm)
                fv_te = compute_fisher_vectors(test_dict, pca, gmm)
                print("Fisher vectors computed for training and test sets.")
                save_stuff(pca, gmm, fv_tr,
                    (PCA_PATH.format(ds_tag, method), GMM_PATH.format(ds_tag, method), FISHER_VECTORS.format(ds_tag, method)))
            
            #params = calibrate(
            #    dataset_tag = dataset_name,
            #    fisher_vecs = fv_tr,
            #    descriptors = train_dict,
            #    keypoints = train_keypoints,
            #)
            #print("Dataset-specific GV params: ", params)
            

        if args.use_color:
            print("Extracting colour descriptors...")
            train_paths = get_image_paths(df_train, args.remove_background)
            test_paths = get_image_paths(df_test, args.remove_background)
            color_tr = compute_color_descriptors(train_paths)
            color_te = compute_color_descriptors(test_paths)
            color_tr, mean_c, std_c = standardize_colors(color_tr)
            color_te, _, _ = standardize_colors(color_te, mean_c, std_c)
            color_tr = normalize_hsv(color_tr)
            color_te = normalize_hsv(color_te)
        else:
            color_tr, color_te = {}, {}

        if args.use_global_embedding:
            print("Extracting global embeddings...")
            train_paths = dict(zip(df_train["image_id"].astype(str), get_image_paths(df_train, args.remove_background)))
            test_paths = dict(zip(df_test["image_id"].astype(str), get_image_paths(df_test, args.remove_background)))
            emb_tr_path = f"{base_dir}/global_embeddings_train_{args.embedding_model}.pkl"
            emb_te_path = f"{base_dir}/global_embeddings_test_{args.embedding_model}.pkl"
            if os.path.exists(emb_tr_path) and os.path.exists(emb_te_path):
                with open(emb_tr_path, "rb") as f:
                    emb_tr = pickle.load(f)
                with open(emb_te_path, "rb") as f:
                    emb_te = pickle.load(f)
            else:
                emb_tr = extract_global_embeddings(train_paths, model_name=args.embedding_model)
                emb_te = extract_global_embeddings(test_paths, model_name=args.embedding_model)
                with open(emb_tr_path, "wb") as f:
                    pickle.dump(emb_tr, f)
                with open(emb_te_path, "wb") as f:
                    pickle.dump(emb_te, f)
        else:
            emb_tr, emb_te = {}, {}

        train_labels = dict(zip(df_train["image_id"], df_train["identity"]))
        test_labels = dict(zip(df_test["image_id"], df_test["identity"]))
        if args.use_shape:
            print("Extracting shape descriptors...")
            train_paths = get_image_paths(df_train, args.remove_background)
            test_paths = get_image_paths(df_test, args.remove_background)
            shape_tr = compute_shape_descriptors(train_paths)
            shape_te = compute_shape_descriptors(test_paths)
            shape_tr, mean_s, std_s = standardize_shapes(shape_tr)
            shape_te, _, _ = standardize_shapes(shape_te, mean_s, std_s)
            with open(SHAPE_TRAIN_PATH.format(ds_tag), 'wb') as f:
                pickle.dump(shape_tr, f)
            with open(SHAPE_TEST_PATH.format(ds_tag), 'wb') as f:
                pickle.dump(shape_te, f)
        else:
            shape_tr, shape_te = {}, {}

        # Normalise and fuse descriptor blocks
        train_blocks = {'fisher': fv_tr}
        test_blocks = {'fisher': fv_te}
        if args.use_color:
            train_blocks['color'] = color_tr
            test_blocks['color'] = color_te
        if args.use_global_embedding:
            train_blocks['global'] = emb_tr
            test_blocks['global'] = emb_te
        if args.use_shape:
            train_blocks['shape'] = shape_tr
            test_blocks['shape'] = shape_te

        norm_train_blocks = {}
        norm_test_blocks = {}
        for name in train_blocks.keys():
            tr_norm, te_norm = apply_zscore_and_l2_train_test(train_blocks[name], test_blocks[name])
            norm_train_blocks[name] = tr_norm
            norm_test_blocks[name] = te_norm

        weight_map = {
            'fisher': args.w_fisher,
            'color': args.w_color,
            'shape': args.w_shape,
            'global': args.w_global,
        }

        fused_tr = fuse_blocks_weighted_concat(norm_train_blocks, weight_map)
        fused_te = fuse_blocks_weighted_concat(norm_test_blocks, weight_map)
        
        if args.use_geometric_verification:
            if args.method == 'keynet_hardnet':
                method = 'disk'
            else:
                method = 'disk'
            print("Running training evaluation with geometric verification...")
            preds = classify_test_images_with_geometric_verification(
                fused_te, fused_tr,
                test_keypoints, train_keypoints,
                test_dict, train_dict, train_labels, 5,
                use_lightglue=args.use_lightglue, method=method,
            )
        else:
            print("Running standard training evaluation...")
            preds = classify_test_images(
                fused_te, fused_tr, train_labels, 5,
            )
        
        metrics = evaluate_predictions(preds, test_labels)
        
        if use_cuda:
            torch.cuda.synchronize()
            
        runtime_sec = time.time() - start_eval_time
        metrics["eval_runtime_sec"] = runtime_sec
        
        if args.save_eval:
            #save_evaluation_results(metrics, ds_tag)
            save_evaluation_results(metrics, ds_tag, tag=tag)
            row = {
                "Dataset": dataset_name,
                "Training Examples": len(df_train.index),
                "Num Classes": df_train['identity'].nunique(),
                "Method": args.method,
                "Remove Background": args.remove_background,
                "Use Color": args.use_color,
                "Use Shape": args.use_shape,
                "Use Global Embedding": args.use_global_embedding,
                "GMM Components": N_COMPONENTS_GMM,
                "PCA Components": N_COMPONENTS_PCA,
                "Use GV": args.use_geometric_verification,
                "GV Method": gv_method if args.use_geometric_verification else "N/A",
                "Alpha (fv sim - gv)": ALPHA,
                "Geom. Candidates": GEOMETRIC_CANDIDATES,
                "Min Inliers": MIN_INLIERS,
                "Inlier Threshold": INLIER_THRESHOLD,
                "MAX GMM Descriptors (per image)": MAX_DESCRIPTORS_PER_IMAGE,
                "Multiscale Enabled": ENABLE_MULTISCALE,
                "Run Time (minutes)": round((float(metrics["eval_runtime_sec"]) / 60), 2),
                "Accuracy": round(float(metrics["accuracy"]), 4),
                "Top-5 Accuracy": round(float(metrics["top_n_accuracy"]), 4),
                "F-1 Score": round(float(metrics["classification_metrics"]["weighted avg"]["f1-score"]), 4)
                
            }
            save_count_results_wrapper(row, EVAL_RESULTS_XLSX)
        
        #_input = input("Create a full‑dataset DB? (yes/no) ")
        _input = 'no'
        if _input.strip().lower() == "yes":
            extract_features(get_image_paths(df, args.remove_background), MODEL_PATH, f"{base_dir}/db/")
            db_dict = load_descriptors(f"{base_dir}/db/descriptors.h5")
            desc = stack_all_descriptors(db_dict)
            pca_db = train_pca(desc)
            gmm_db = train_gmm(pca_db.transform(desc))
            fv_db = compute_fisher_vectors(db_dict, pca_db, gmm_db)
            save_stuff(pca_db, gmm_db, fv_db,
                (f"{base_dir}/db/pca.pkl",
                f"{base_dir}/db/gmm.pkl",
                f"{base_dir}/db/fisher_vectors.pkl"))
            print("Database saved.")
        sys.exit(0)
        
    if args.count:
        print(f"Using geometric verification method: {gv_method}")
    

        start_time = time.time()
        ds_tag = dataset_name
        base_dir = f"./data/{ds_tag}"

        df_raw = load_dataset(dataset_name)
        df_raw["image_id"] = df_raw["image_id"].astype(str)

        csv_path = f"{base_dir}/processed_metadata.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Check if the metadata has the required columns for counting
            required_cols = ['processed_path', 'processed_path_segmented'] if args.remove_background else ['processed_path']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Need to preprocess to create the missing columns
                output_dir_preprocess = 'segmented_dataset' if args.remove_background else 'dataset'
                df = preprocessing.preprocess_dataset(
                    df_raw.copy(),
                    f"{base_dir}/{output_dir_preprocess}/",
                    dataset_name,
                    use_mantiuk=args.use_mantiuk,
                    remove_background=args.remove_background,
                )
                df.to_csv(csv_path, index=False)
        else:
            os.makedirs(base_dir, exist_ok=True)
            output_dir_preprocess = 'segmented_dataset' if args.remove_background else 'dataset'
            df = preprocessing.preprocess_dataset(
                df_raw.copy(),
                f"{base_dir}/{output_dir_preprocess}/",
                dataset_name,
                use_mantiuk=args.use_mantiuk,
                remove_background=args.remove_background,
            )
            df.to_csv(csv_path, index=False)

        if method == 'ensamble':
            methods = ['disk', 'keynet_hardnet', 'lightglue']
            fv_list = []
            for m in methods:
                feat_dir = f"{base_dir}/feature_descriptors_{m}/"
                if not os.path.isdir(feat_dir):
                    if m == 'disk':
                        extract_features(get_image_paths(df, args.remove_background), MODEL_PATH, feat_dir)
                    elif m == 'keynet_hardnet':
                        extract_features_keynet_hardnet_faster(get_image_paths(df, args.remove_background), feat_dir)
                    elif m == 'lightglue':
                        extract_features_lightglue(get_image_paths(df, args.remove_background), feat_dir)

                desc = load_descriptors(os.path.join(feat_dir, "descriptors.h5"))
                if m == 'disk':
                    descriptors = desc
                    keypoints = load_keypoints(os.path.join(feat_dir, "keypoints.h5"))
                pca_path_m = f"{base_dir}/pca_model_{m}.pkl"
                gmm_path_m = f"{base_dir}/gmm_model_{m}.pkl"
                fv_path_m = f"{base_dir}/fisher_vectors_{m}.pkl"
                if os.path.exists(fv_path_m):
                    pca_m, gmm_m, fv_m = load_stuff(pca_path_m, gmm_path_m, fv_path_m)
                else:
                    desc_stack = stack_all_descriptors(desc)
                    pca_m = train_pca(desc_stack)
                    gmm_m = train_gmm(pca_m.transform(desc_stack))
                    fv_m = compute_fisher_vectors(desc, pca_m, gmm_m)
                    save_stuff(pca_m, gmm_m, fv_m, (pca_path_m, gmm_path_m, fv_path_m))
                fv_list.append(fv_m)

            fisher_vectors = combine_fisher_vectors(fv_list, ENSEMBLE_WEIGHTS)
        else:
            feat_dir = f"{base_dir}/feature_descriptors_{method}/"
            if not os.path.isdir(feat_dir) and method == 'disk':
                extract_features(get_image_paths(df, args.remove_background), MODEL_PATH, feat_dir)
            elif not os.path.isdir(feat_dir) and method == 'keynet_hardnet':
                extract_features_keynet_hardnet_faster(get_image_paths(df, args.remove_background), feat_dir)
            descriptors = load_descriptors(os.path.join(feat_dir, "descriptors.h5"))
            keypoints = load_keypoints(os.path.join(feat_dir, "keypoints.h5"))

            pca_path = f"{base_dir}/pca_model_{method}.pkl"
            gmm_path = f"{base_dir}/gmm_model_{method}.pkl"
            fv_path = f"{base_dir}/fisher_vectors_{method}.pkl"
            if os.path.exists(fv_path):
                pca, gmm, fisher_vectors = load_stuff(pca_path, gmm_path, fv_path)
            else:
                desc_stack = stack_all_descriptors(descriptors)
                pca = train_pca(desc_stack)
                gmm = train_gmm(pca.transform(desc_stack))
                fisher_vectors = compute_fisher_vectors(descriptors, pca, gmm)
                save_stuff(pca, gmm, fisher_vectors, (pca_path, gmm_path, fv_path))

        # Gather descriptor blocks
        blocks = {'fisher': fisher_vectors}

        if args.use_color:
            print("Extracting colour descriptors for counting...")
            image_paths = get_image_paths(df, args.remove_background)
            color = compute_color_descriptors(image_paths)
            color, _, _ = standardize_colors(color)
            color = normalize_hsv(color)
            blocks['color'] = color

        if args.use_shape:
            print("Extracting shape descriptors for counting...")
            image_paths = get_image_paths(df, args.remove_background)
            shape = compute_shape_descriptors(image_paths)
            shape, _, _ = standardize_shapes(shape)
            blocks['shape'] = shape

        if args.use_global_embedding:
            print("Extracting global embeddings for population counting...")
            image_paths = dict(zip(df["image_id"].astype(str), get_image_paths(df, args.remove_background)))
            emb_path = f"{base_dir}/global_embeddings_count_{args.embedding_model}.pkl"

            if os.path.exists(emb_path):
                print("Loading cached global embeddings...")
                with open(emb_path, "rb") as f:
                    emb = pickle.load(f)
            else:
                print("Computing global embeddings...")
                emb = extract_global_embeddings(image_paths, model_name=args.embedding_model)
                with open(emb_path, "wb") as f:
                    pickle.dump(emb, f)
            blocks['global'] = emb

        # Normalise and fuse blocks
        norm_blocks = {}
        for name, blk in blocks.items():
            norm_blk, _ = apply_zscore_and_l2_train_test(blk, blk)
            norm_blocks[name] = norm_blk

        weight_map = {
            'fisher': args.w_fisher,
            'color': args.w_color,
            'shape': args.w_shape,
            'global': args.w_global,
        }

        fisher_vectors = fuse_blocks_weighted_concat(norm_blocks, weight_map)

        labels = dict(zip(df["image_id"], df["identity"])) if not args.automated_mode else {}
        
        if args.enhanced_estimation:
            print(f"🎯 Using Enhanced Population Estimation")
            if args.use_ensemble:
                print(f"Multi-threshold ensemble mode")
            else:
                print(f"Single threshold with confidence assessment")
            
            enhanced_results = enhanced_population_estimation(
                fisher_vectors=fisher_vectors,
                labels=labels,
                keypoints=keypoints,
                descriptors=descriptors,
                use_ensemble=args.use_ensemble,
                single_threshold=args.gv_threshold,
                # Pass through other parameters
                use_geometric=args.use_geometric_verification,
                use_lightglue=args.use_lightglue,
                method='disk',
                n_vertices=args.num_vertices,
                n_neighbors=args.num_neighbors,
                label_error_rate=args.label_error_rate,
                automated_mode=args.automated_mode
            )
            
            if args.use_ensemble:
                estimate = enhanced_results['ensemble_estimate']
                se = enhanced_results['ensemble_std']
                stats = enhanced_results['best_individual']['stats']
                confidence = enhanced_results['best_individual']['confidence']
                print(f"Enhanced Results: {estimate:.2f} ± {se:.2f} (ensemble)")
                print(f"Recommendation: {enhanced_results['recommendation']}")
            else:
                estimate = enhanced_results['estimate']
                se = enhanced_results['std_error']
                stats = enhanced_results['stats']
                confidence = enhanced_results['confidence']
                print(f"Enhanced Results: {estimate:.2f} ± {se:.2f} ({confidence})")
        else:
            print(f"Using Standard Population Estimation")
            estimate, se, stats = nested_importance_sampling(
                fisher_vectors,
                labels,
                keypoints=keypoints,
                descriptors=descriptors,
                use_geometric=args.use_geometric_verification,
                use_lightglue=args.use_lightglue,
                method='disk',
                gv_threshold = args.gv_threshold,
                n_vertices=args.num_vertices,
                n_neighbors=args.num_neighbors,
                label_error_rate=args.label_error_rate,
                return_stats = True,
                automated_mode=args.automated_mode
            )
            # Add confidence assessment even for standard mode
            confidence, reason = assess_confidence_level(stats)
            print(f"Confidence Assessment: {confidence} - {reason}")
        runtime = time.time() - start_time
        print(f"Estimated individuals: {estimate:.2f} \u00b1 {se:.2f}")
        print(stats)
        if args.save_count:
            row = {
                "Dataset": dataset_name,
                "Method": method,
                "Mutliscale Enabled": ENABLE_MULTISCALE,
                "Num Vertices": args.num_vertices,
                "Num Neighbors": args.num_neighbors,
                "Total Pairs": int(stats.get("total_pairs", 0)),
                "GV Attempts": int(stats.get("gv_attempts", 0)),
                "GV Passes": int(stats.get("gv_passes", 0)),
                "Label Queries": int(stats.get("label_queries", 0)),
                "Matches": int(stats.get("matches", 0)),
                "Remove Background": args.remove_background,
                "Use Color": args.use_color,
                "Use Shape": args.use_shape,
                "Use Global Embedding": args.use_global_embedding,
                "GMM Components": N_COMPONENTS_GMM,
                "MAX GMM Descriptors": MAX_GMM_DESCRIPTORS,
                "GV Threshold": args.gv_threshold,
                "GV Method": gv_method if args.use_geometric_verification else "N/A",
                "Error Rate": args.label_error_rate,
                "Automated Mode": args.automated_mode,
                "Enhanced Estimation": args.enhanced_estimation,
                "Use Ensemble": args.use_ensemble if args.enhanced_estimation else False,
                "Confidence Level": confidence if 'confidence' in locals() else "N/A",
                "Result": round(float(estimate), 2),
                "Std Error": round(float(se), 2),
                "Runtime (minutes)": round(float(runtime) / 60, 2) ,
                "Ground Truth": int(df["identity"].nunique()),
                
            }
            save_count_results(row)