import argparse
from wildlife_datasets import datasets, splits
import preprocessing
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_false')
    parser.add_argument('--dataset', type = str, help="Specify the dataset to use (e.g., ATRW, BelugaID, etc.)")
    parser.add_argument('--image_location', type = str)
    #parser.add_argument('--preprocess', action= 'stroe_true')

    args = parser.parse_args()
    
    if args.train and not args.dataset:
        print("Please specify the dataset to train on.")

    if args.train:
        print(f"Training mode selected. Using dataset: {args.dataset}")

        print('training...')

        #datasets.ATRW.get_data(f'./data/ATRW/')
        dataset = datasets.ATRW('./data/ATRW')
        splitter = splits.ClosedSetSplit(0.8)
        df = dataset.df
        df = df[df['path'].str.startswith(tuple(('atrw_reid_train', 'atrw_reid_test')))]
        for idx_train, idx_test in splitter.split(df):
            splits.analyze_split(df, idx_train, idx_test)

        df_train, df_test = df.loc[idx_train], df.loc[idx_test] 

        

        #processed_df = preprocessing.preprocess_dataset(df, './data/segmented_dataset/')
        #processed_df.to_csv('./data/processed_metadata.csv', index=False)

        processed_df_train = preprocessing.preprocess_dataset(df_train, './data/segmented_dataset_train/')
        processed_df_test = preprocessing.preprocess_dataset(df_test, './data/segmented_dataset_test/')

        processed_df_train.to_csv('./data/processed_metadata_train.csv', index=False)
        processed_df_test.to_csv('./data/processed_metadata_test.csv', index=False)

    elif args.predict:
        if not args.image_location:
            print('Please enter image location...')
            sys.exit(0)
        print('Predicting...')
    