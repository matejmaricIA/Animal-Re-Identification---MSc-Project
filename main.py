import argparse
from wildlife_datasets import datasets, splits
from wildlife_datasets.datasets import WildlifeReID10k
from patches.elpephants_patch import PatchedELPephants
datasets.ELPephants = PatchedELPephants
import preprocessing
import sys
from feature_extraction import get_image_paths, extract_features
from feature_aggregation import load_descriptors, stack_all_descriptors, train_pca, train_gmm, compute_fisher_vectors
from constants import *
import os
import pickle
from predict import classify_test_images, predict
from evaluate import evaluate_predictions, save_evaluation_results
import pandas as pd
import shutil
import cv2
import sys
#from visualize import visualize_results
import numpy as np
import torch
import time

def save_stuff(pca, gmm, fisher_vectors, paths = (PCA_PATH, GMM_PATH, FISHER_VECTORS)):
    with open(paths[0], "wb") as f:
        pickle.dump(pca, f)

    with open(paths[1], "wb") as f:
        pickle.dump(gmm, f)

    with open(paths[2], "wb") as f:
        pickle.dump(fisher_vectors, f)

def load_stuff(pca_path, gmm_path, fisher_vectors_path):
    with open(pca_path, 'rb') as file:
        pca = pickle.load(file)

    with open(gmm_path, 'rb') as file:
        gmm = pickle.load(file)

    with open(fisher_vectors_path, 'rb') as file:
        fisher = pickle.load(file)

    return pca, gmm, fisher

def load_dataset(subset, root = WILD_DATASET_PATH):
        """Return a dataframe ready for the pipeline."""
        print(root)
        ds = WildlifeReID10k(root, check_files=False)
        df = ds.metadata.copy()
        if subset != 'full':
            print(f"Filtering to subset: {subset}")
            df = df[df["dataset"].str.lower() == subset.lower()].copy()
            if df.empty:
                print(f"Subset '{subset}' not found.")
                sys.exit(1)
        return df

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_true')
    parser.add_argument('--ds', type = str, help="Specify the dataset to use (e.g., ATRW, BelugaID, etc.), full to train on the full dataset.",
                        default = 'full')
    parser.add_argument('--image_location', type = str)
    parser.add_argument('--save_eval', action='store_true', help='Save evaluation metrics during training', default = True)
    parser.add_argument('--use_mantiuk', action='store_true', help='Use Mantiuk tone mapping during preprocessing')
    parser.add_argument('--remove_background', action='store_true', help='Remove background during preprocessing')
    parser.add_argument('--version', type=str, default='1', help='Identifier for the current method version')
    parser.add_argument('--split_type', type=str, default='closed', help='Open set or closed set split type')
    parser.add_argument('--method', type = str, default = 'disk', help='Feature extraction method to use')
    #parser.add_argument('--split_type', type=str, choices=['balanced_split', 'time_proportion'], default='balanced_split',)
    #parser.add_argument('--use_original_split', action='store_true', help='Use original split from dataset metadata if available', default=False)
    #parser.add_argument('--preprocess', action= 'stroe_true')

    args = parser.parse_args()
    dataset_name = args.ds
    method = args.method
    #use_splitter = False
    
    already_trained = False
    #split_type = 'closed_split'

    # create a configuration tag for saving evaluation results
    tag = f"v_{args.version}_tm_{args.use_mantiuk}_{args.split_type}_{method}_PCA_{N_COMPONENTS_PCA}_GMM_{N_COMPONENTS_GMM}"
    
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
            for sub_name, sub_df in df_raw.groupby("dataset"):
                sub_dir = f"./data/{sub_name}"
                os.makedirs(sub_dir, exist_ok=True)
                csv_path = f"{sub_dir}/processed_metadata.csv"
                if not os.path.exists(csv_path):
                    processed = preprocessing.preprocess_dataset(
                        sub_df.copy(),
                        f"{sub_dir}/segmented_dataset/",
                        sub_name,
                        use_mantiuk=args.use_mantiuk,
                        remove_background=args.remove_background,
                    )
                    processed.to_csv(csv_path, index=False)
                else:
                    processed = pd.read_csv(csv_path)
                processed_frames.append(processed)
            df = pd.concat(processed_frames, ignore_index=True)
        else:
            sub_dir = f"./data/{dataset_name}"
            os.makedirs(sub_dir, exist_ok=True)
            csv_path = f"{sub_dir}/processed_metadata.csv"
            if not os.path.exists(csv_path):
                df = preprocessing.preprocess_dataset(
                    df_raw.copy(),
                    f"{sub_dir}/segmented_dataset/",
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
                extract_features(get_image_paths(df_train), MODEL_PATH, f"{base_dir}/feature_descriptors_train_{method}/")
                extract_features(get_image_paths(df_test), MODEL_PATH, f"{base_dir}/feature_descriptors_test_{method}/")
            elif method == 'keynet_hardnet':
                extract_features_keynet_hardnet(get_image_paths(df_train), f"{base_dir}/feature_descriptors_train_{method}/")
                extract_features_keynet_hardnet(get_image_paths(df_test), f"{base_dir}/feature_descriptors_test_{method}/")

        else:
            already_trained = True
            
            
            
        train_dict = load_descriptors(f"{base_dir}/feature_descriptors_train/descriptors.h5")
        test_dict = load_descriptors(f"{base_dir}/feature_descriptors_test/descriptors.h5")
        
        desc_tr = stack_all_descriptors(train_dict)
        desc_te = stack_all_descriptors(test_dict)
        
        if already_trained:
            print("Using already trained PCA and GMM models.")
            pca, gmm, fv_tr = load_stuff(
                f"{base_dir}/pca_model_{method}.pkl",
                f"{base_dir}/gmm_model_{method}.pkl",
                f"{base_dir}/fisher_vectors_{method}.pkl"
            )
            fv_te = compute_fisher_vectors(test_dict, pca, gmm)
        else:
            pca = train_pca(desc_tr)
            gmm = train_gmm(pca.transform(desc_tr))
            fv_tr = compute_fisher_vectors(train_dict, pca, gmm)
            fv_te = compute_fisher_vectors(test_dict, pca, gmm)
            
            save_stuff(pca, gmm, fv_tr,
                (PCA_PATH.format(ds_tag, method), GMM_PATH.format(ds_tag, method), FISHER_VECTORS.format(ds_tag, method)))
        
        train_labels = dict(zip(df_train["image_id"], df_train["identity"]))
        test_labels = dict(zip(df_test["image_id"], df_test["identity"]))
        
        preds = classify_test_images(fv_te, fv_tr, train_labels, 5)
        metrics = evaluate_predictions(preds, test_labels)
        
        if use_cuda:
            torch.cuda.synchronize()
            
        runtime_sec = time.time() - start_eval_time
        metrics["eval_runtime_sec"] = runtime_sec
        
        if args.save_eval:
            #save_evaluation_results(metrics, ds_tag)
            save_evaluation_results(metrics, ds_tag, tag=tag)
        
        #_input = input("Create a full‑dataset DB? (yes/no) ")
        _input = 'no'
        if _input.strip().lower() == "yes":
            extract_features(get_image_paths(df), MODEL_PATH, f"{base_dir}/db/")
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
        
    if args.predict:
        if not args.image_location:
            print("--image_location is required for prediction mode.")
            sys.exit(1)

        ds_tag = dataset_name
        base_dir = f"./data/{ds_tag}"

        query_paths = [os.path.join(args.image_location, p) for p in os.listdir(args.image_location)]
        tmp_dir = preprocessing.preprocess_inference(
            query_paths,
            use_mantiuk=args.use_mantiuk,
            remove_background=args.remove_background,
        )

        query_imgs = [os.path.join(tmp_dir, p) for p in os.listdir(tmp_dir)]
        extract_features(query_imgs, MODEL_PATH, TMP)

        pca, gmm, fv_db = load_stuff(
            f"{base_dir}/db/pca.pkl",
            f"{base_dir}/db/gmm.pkl",
            f"{base_dir}/db/fisher_vectors.pkl",
        )

        query_desc = load_descriptors(TMP + "descriptors.h5")
        fv_query = compute_fisher_vectors(query_desc, pca, gmm)
        predict(fv_query, fv_db, ds_tag)
        shutil.rmtree(TMP)