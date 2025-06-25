import os
import sys
import pickle
import pandas as pd
from wildlife_datasets.datasets import WildlifeReID10k
from constants import (
    PCA_PATH, GMM_PATH, FISHER_VECTORS, COUNT_RESULTS_XLSX, WILD_DATASET_PATH
)




def save_count_results(row: dict, path: str = COUNT_RESULTS_XLSX) -> None:
    """Append a population counting result to the XLSX file sorted by dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df = pd.read_excel(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("Dataset").reset_index(drop=True)
    df.to_excel(path, index=False)
    print(f"Saved count results to {path}")
    
    
    
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