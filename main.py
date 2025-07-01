import argparse
from wildlife_datasets import datasets, splits
from wildlife_datasets.datasets import WildlifeReID10k
from patches.elpephants_patch import PatchedELPephants
datasets.ELPephants = PatchedELPephants
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
from constants import *
import os
import pickle
from predict import classify_test_images, predict, classify_test_images_with_geometric_verification
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
from utility_functions import load_dataset, save_stuff, load_stuff, save_count_results



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

    args = parser.parse_args()
    args.split_type = 'closed'
    dataset_name = args.ds
    method = args.method
    #use_splitter = False
    
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
                
                if dataset_name.lower() in ["BelugaID", "SealID", "StripeSpotter", "SeaTurtleID", "SeaStarReID2023", "NDD20"]:
                    args.remove_background = False
                
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

        else:
            already_trained = True
            
            
            
        train_dict = load_descriptors(f"{base_dir}/feature_descriptors_train_{method}/descriptors.h5")
        test_dict = load_descriptors(f"{base_dir}/feature_descriptors_test_{method}/descriptors.h5")
        
        train_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_train_{method}/keypoints.h5")
        test_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_test_{method}/keypoints.h5")
        
        desc_tr = stack_all_descriptors(train_dict)
        desc_te = stack_all_descriptors(test_dict)
        
        pca_path = f"{base_dir}/pca_model_{method}.pkl"
        gmm_path = f"{base_dir}/gmm_model_{method}.pkl"
        fv_path  = f"{base_dir}/fisher_vectors_{method}.pkl"
        
        if already_trained and os.path.exists(pca_path) and os.path.exists(gmm_path) and os.path.exists(fv_path):
            print("Using already trained PCA and GMM models.")
            pca, gmm, fv_tr = load_stuff(pca_path, gmm_path, fv_path)
            fv_te = compute_fisher_vectors(test_dict, pca, gmm)
            print("Fisher vectors computed for test set.")
            
            #params = calibrate(
            #    dataset_tag = dataset_name,
            #    fisher_vecs = fv_tr,
            #    descriptors = train_dict,
            #    keypoints = train_keypoints,
            #)
            #print("Dataset-specific GV params: ", params)
            
        else:
            pca = train_pca(desc_tr)
            gmm = train_gmm(pca.transform(desc_tr))
            fv_tr = compute_fisher_vectors(train_dict, pca, gmm)
            fv_te = compute_fisher_vectors(test_dict, pca, gmm)
            print("Fisher vectors computed for training and test sets.")
            
            # Calibrate the dataset
            #params = calibrate(
            #    dataset_tag = dataset_name,
            #    fisher_vecs = fv_tr,
            #    descriptors = train_dict,
            #    keypoints = train_keypoints,
            #)
            #print("Dataset-specific GV params: ", params)

            save_stuff(pca, gmm, fv_tr,
                (PCA_PATH.format(ds_tag, method), GMM_PATH.format(ds_tag, method), FISHER_VECTORS.format(ds_tag, method)))

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
            for k in fv_tr.keys():
                if k in color_tr:
                    fv_tr[k] = np.concatenate([fv_tr[k], color_tr[k]])
            for k in fv_te.keys():
                if k in color_te:
                    fv_te[k] = np.concatenate([fv_te[k], color_te[k]])
        
        train_labels = dict(zip(df_train["image_id"], df_train["identity"]))
        test_labels = dict(zip(df_test["image_id"], df_test["identity"]))
        
        if args.use_geometric_verification:
            if args.method == 'keynet_hardnet':
                method = 'disk'
            else:
                method = 'disk'
            print("Running training evaluation with geometric verification...")
            preds = classify_test_images_with_geometric_verification(
                fv_te, fv_tr, test_keypoints, train_keypoints,
                test_dict, train_dict, train_labels, 5, use_lightglue=args.use_lightglue, method = method
            )
        else:
            print("Running standard training evaluation...")
            preds = classify_test_images(fv_te, fv_tr, train_labels, 5)
        
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
                "Method": method,
                "Remove Background": args.remove_background,
                "Use Color": args.use_color,
                "GMM Components": N_COMPONENTS_GMM,
                "PCA Components": N_COMPONENTS_PCA,
                "Use GV": args.use_geometric_verification,
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
            save_count_results(row, EVAL_RESULTS_XLSX)
        
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
        start_time = time.time()
        ds_tag = dataset_name
        base_dir = f"./data/{ds_tag}"

        df_raw = load_dataset(dataset_name)
        df_raw["image_id"] = df_raw["image_id"].astype(str)

        csv_path = f"{base_dir}/processed_metadata.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            os.makedirs(base_dir, exist_ok=True)
            df = preprocessing.preprocess_dataset(
                df_raw.copy(),
                f"{base_dir}/segmented_dataset/",
                dataset_name,
                use_mantiuk=args.use_mantiuk,
                remove_background=args.remove_background,
            )
            df.to_csv(csv_path, index=False)

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

        labels = dict(zip(df["image_id"], df["identity"]))
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
            return_stats = True
        )
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
                "GMM Components": N_COMPONENTS_GMM,
                "MAX GMM Descriptors": MAX_GMM_DESCRIPTORS,
                "GV Threshold": args.gv_threshold,
                "Error Rate": args.label_error_rate,
                "Result": round(float(estimate), 2),
                "Std Error": round(float(se), 2),
                "Runtime (minutes)": round(float(runtime) / 60, 2) ,
                "Ground Truth": int(df["identity"].nunique()),
                
            }
            save_count_results(row)