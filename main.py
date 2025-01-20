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
from evaluate import evaluate_predictions
import pandas as pd
import shutil
import cv2
from visualize import visualize_results
import numpy as np

def save_stuff(pca, gmm, fisher_vectors, paths = (PCA_PATH, GMM_PATH, FISHER_VECTORS)):
    with open(paths[0], "wb") as f:
        pickle.dump(pca_model, f)

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
    #parser.add_argument('--preprocess', action= 'stroe_true')

    args = parser.parse_args()
    
    if args.train and not args.ds:
        print("Please specify the dataset to train on.")

    if args.train:
        print(f"Training mode selected. Using dataset: {args.ds}")

        print('training...')
        if not os.path.isdir(f"./data/{args.ds}"):
            datasets.ATRW.get_data(f'./data/ATRW/')

        dataset = datasets.ATRW('./data/ATRW')
        df = dataset.df
        df = df[df['path'].str.startswith(tuple(('atrw_reid_train', 'atrw_reid_test')))]

        if not os.path.isdir(SEGMENTED_DATASET):

            processed_df = preprocessing.preprocess_dataset(df, SEGMENTED_DATASET)
            #processed_df_test = preprocessing.preprocess_dataset(df, SEGMENTED_DATASET)

            processed_df.to_csv(DATAFRAME_PATH, index=False)
            #processed_df_test.to_csv(DATAFRAME_PATH, index=False)
        
        processed_df = pd.read_csv(DATAFRAME_PATH)
        splitter = splits.ClosedSetSplit(0.80)
        for idx_train, idx_test in splitter.split(processed_df):
            splits.analyze_split(processed_df, idx_train, idx_test)
            df_train, df_test = processed_df.loc[idx_train], processed_df.loc[idx_test] 


        if not os.path.isdir(SAVE_TRAIN_DESCRIPTORS_FOLDER):

            # Extract features from train images
            img_paths = get_image_paths(df_train)
            extract_features(img_paths, MODEL_PATH, SAVE_TRAIN_DESCRIPTORS_FOLDER)

            # Extract features from test images
            img_paths = get_image_paths(df_test)
            extract_features(img_paths, MODEL_PATH, SAVE_TEST_DESCRIPTORS_FOLDER)

        # Load saved descriptors
        train_dict = load_descriptors(SAVE_TRAIN_DESCRIPTORS_PATH)
        test_dict = load_descriptors(SAVE_TEST_DESCRIPTORS_PATH)

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

        save_stuff(pca_model, gmm, train_fisher_vectors)

        train_labels = dict(zip(df_train['image_id'], df_train['identity']))
        test_labels = dict(zip(df_test['image_id'], df_test['identity']))

        predictions = classify_test_images(test_fisher_vectors, train_fisher_vectors, train_labels, 5)
        results = evaluate_predictions(predictions, test_labels)

        database_prompt = input("Do you want to create a database with the full dataset? (yes/no)")
        if database_prompt == "yes":
            img_paths = get_image_paths(processed_df)
            extract_features(img_paths, MODEL_PATH, DB_PATH)
            path = os.path.join(DB_PATH, 'descriptors.h5')
            print(path)
            db_dict = load_descriptors(path)
            descriptors = stack_all_descriptors(db_dict)
            pca_model = train_pca(descriptors)
            reduced_descriptors = pca_model.transform(descriptors)
            gmm = train_gmm(reduced_descriptors)
            fisher_vectors = compute_fisher_vectors(db_dict, pca_model, gmm)
            paths = (os.path.join(DB_PATH, 'pca.pkl'), os.path.join(DB_PATH, 'gmm.pkl'), os.path.join(DB_PATH, 'fisher_vectors.pkl'))
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
        pca_model, gmm_model, fisher_vectors = load_stuff(DB_PATH + 'pca.pkl', DB_PATH + 'gmm.pkl', DB_PATH + 'fisher_vectors.pkl')
        print("PCA, GMM and Fisher Vectors loaded.")
        descriptors = load_descriptors(TMP + 'descriptors.h5')
        descriptors_stacked = stack_all_descriptors(descriptors)
        #descriptors_pca = pca_model.transform(descriptors_stacked)
        pred_fisher_vectors = compute_fisher_vectors(descriptors, pca_model, gmm_model)

        predict(pred_fisher_vectors, fisher_vectors)



        shutil.rmtree(TMP)
        
        
    