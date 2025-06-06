import argparse
from wildlife_datasets import datasets, splits
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

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_true')
    parser.add_argument('--ds', type = str, help="Specify the dataset to use (e.g., ATRW, BelugaID, etc.)")
    parser.add_argument('--image_location', type = str)
    parser.add_argument('--save_eval', action='store_true', help='Save evaluation metrics during training', default = True)
    parser.add_argument('--use_mantiuk', action='store_true', help='Use Mantiuk tone mapping during preprocessing', default = True)
    parser.add_argument('--remove_background', action='store_true', help='Remove background during preprocessing', default = True)
    #parser.add_argument('--preprocess', action= 'stroe_true')

    args = parser.parse_args()
    dataset_name = args.ds

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
        if not os.path.isdir(f"./data/{args.ds}"):
            #datasets.ATRW.get_data(f'./data/ATRW/')
            datasets.__dict__[dataset_name].get_data(f"./data/{dataset_name}/")

        #dataset = datasets.ATRW('./data/ATRW')
        dataset = datasets.__dict__[dataset_name](f"./data/{dataset_name}/")
        df = dataset.df
        df['image_id'] = df['image_id'].astype(str)

        if not os.path.isdir(f'./data/{dataset_name}/segmented_dataset/'):
            remove_bg = not datasets.__dict__[dataset_name].summary.get('cropped', False)
            processed_df = preprocessing.preprocess_dataset(
                df,
                f'./data/{dataset_name}/segmented_dataset/',
                dataset_name,
                remove_background=remove_bg
            )
            processed_df.to_csv(f'./data/{dataset_name}/processed_metadata.csv', index=False)
        
        #processed_df = pd.read_csv(f'./data/{dataset_name}/processed_metadata.csv')
        processed_df = pd.read_csv(
            f'./data/{dataset_name}/processed_metadata.csv',
            parse_dates=['date'],
        )
        processed_df['image_id'] = processed_df['image_id'].astype(str)
        if 'date' in processed_df.columns:
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
        splitter = splits.ClosedSetSplit(0.80)
        for idx_train, idx_test in splitter.split(processed_df):
            splits.analyze_split(processed_df, idx_train, idx_test)
            df_train, df_test = processed_df.loc[idx_train], processed_df.loc[idx_test] 


        if not os.path.isdir(f"./data/{dataset_name}/feature_descriptors_train/"):

            # Extract features from train images
            img_paths = get_image_paths(df_train)
            extract_features(img_paths, MODEL_PATH, f"./data/{dataset_name}/feature_descriptors_train/")

            # Extract features from test images
            img_paths = get_image_paths(df_test)
            extract_features(img_paths, MODEL_PATH, f'./data/{dataset_name}/feature_descriptors_test/')

        # Load saved descriptors
        train_dict = load_descriptors(f'./data/{dataset_name}/feature_descriptors_train/descriptors.h5')
        test_dict = load_descriptors(f'./data/{dataset_name}/feature_descriptors_test/descriptors.h5')

        descriptors_train = stack_all_descriptors(train_dict)
        descriptors_test = stack_all_descriptors(test_dict)

        # Train PCA model
        pca_model = train_pca(descriptors_train)
        reduced_train_descriptors = pca_model.transform(descriptors_train)
        recuded_test_descriptors = pca_model.transform(descriptors_test)

        # Train GMM
        gmm = train_gmm(reduced_train_descriptors)
        train_fisher_vectors = compute_fisher_vectors(train_dict, pca_model, gmm)
        test_fisher_vectors = compute_fisher_vectors(test_dict, pca_model, gmm)

        save_stuff(pca_model, gmm, train_fisher_vectors, (PCA_PATH.format(dataset_name), GMM_PATH.format(dataset_name), FISHER_VECTORS.format(dataset_name)))

        train_labels = dict(zip(df_train['image_id'], df_train['identity']))
        test_labels = dict(zip(df_test['image_id'], df_test['identity']))

        predictions = classify_test_images(test_fisher_vectors, train_fisher_vectors, train_labels, 5)
        results = evaluate_predictions(predictions, test_labels)
        
        if args.save_eval:
            save_evaluation_results(results, dataset_name)

        database_prompt = input("Do you want to create a database with the full dataset? (yes/no)")
        if database_prompt == "yes":
            img_paths = get_image_paths(processed_df)
            extract_features(img_paths, MODEL_PATH, f'./data/{dataset_name}/db/')
            path = os.path.join(f'./data/{dataset_name}/db/', 'descriptors.h5')
            #print(path)
            db_dict = load_descriptors(path)
            descriptors = stack_all_descriptors(db_dict)
            pca_model = train_pca(descriptors)
            reduced_descriptors = pca_model.transform(descriptors)
            gmm = train_gmm(reduced_descriptors)
            fisher_vectors = compute_fisher_vectors(db_dict, pca_model, gmm)
            paths = (os.path.join(f'./data/{dataset_name}/db/', 'pca.pkl'), os.path.join(f'./data/{dataset_name}/db/', 'gmm.pkl'), os.path.join(f'./data/{dataset_name}/db/', 'fisher_vectors.pkl'))
            save_stuff(pca_model, gmm, fisher_vectors, paths)
            print("Database saved")

        else:
            sys.exit(0)

    if args.predict:
        if not args.image_location:
            print('Please enter image location...')
            sys.exit(0)
        print('Predicting...')
        paths = [os.path.join(args.image_location, img) for img in os.listdir(args.image_location)]
        save_dir = preprocessing.preprocess_inference(paths)
        paths = [os.path.join(save_dir, img) for img in os.listdir(save_dir)]
        extract_features(paths, MODEL_PATH, TMP)
        pca_model, gmm_model, fisher_vectors = load_stuff(
            f'./data/{dataset_name}/db/' + 'pca.pkl',
            f'./data/{dataset_name}/db/' + 'gmm.pkl',
            f'./data/{dataset_name}/db/' + 'fisher_vectors.pkl'
        )
        print("PCA, GMM and Fisher Vectors loaded.")
        descriptors = load_descriptors(TMP + 'descriptors.h5')
        pred_fisher_vectors = compute_fisher_vectors(descriptors, pca_model, gmm_model)

        predict(pred_fisher_vectors, fisher_vectors, dataset_name)



        shutil.rmtree(TMP)
        
        
    