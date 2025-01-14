import argparse
from wildlife_datasets import datasets
import preprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_false')
    parser.add_argument('--dataset', type = str, help="Specify the dataset to use (e.g., ATRW, BelugaID, etc.)")
    #parser.add_argument('--preprocess', action= 'stroe_true')

    args = parser.parse_args()
    
    if args.train and not args.dataset:
        print("Please specify the dataset to train on.")

    if args.train:
        print(f"Training mode selected. Using dataset: {args.dataset}")

        print('training...')

        #datasets.ATRW.get_data(f'./data/ATRW/')
        dataset = datasets.ATRW('./data/ATRW')
        df = dataset.df

        df = df[df['path'].str.startswith(tuple(('atrw_reid_train', 'atrw_reid_test')))]

        processed_df = preprocessing.preprocess_dataset(df, './data/segmented_dataset/')
        processed_df.to_csv('./data/processed_metadata.csv', index=False)

