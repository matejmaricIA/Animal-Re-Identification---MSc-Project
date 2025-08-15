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

def load_dataset(subset, root=WILD_DATASET_PATH):
    """
    Enhanced dataset loading that supports both WildlifeReID10k and manual datasets.

    Args:
        subset: Dataset name (e.g., 'ATRW', 'wild_boar', 'roe_deer')
        root: Root path for WildlifeReID10k datasets

    Returns:
        DataFrame ready for the pipeline
    """

    # Manual datasets that use local processed_metadata.csv
    manual_datasets = ['wild_boar', 'roe_deer']

    if subset.lower() in manual_datasets:
        print(f"Loading manual dataset: {subset}")
        
        # Load from local processed_metadata.csv
        metadata_path = f"./data/{subset}/processed_metadata.csv"
        
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            print(f"Please run: python utils/create_manual_dataset_metadata.py --dataset {subset}")
            sys.exit(1)
        
        df = pd.read_csv(metadata_path)
        print(f"Loaded {len(df)} images from {subset} dataset")
        return df

    else:
        # Use original WildlifeReID10k loading
        print(f"Loading WildlifeReID10k subset: {subset}")
        
        try:
            from wildlife_datasets.datasets import WildlifeReID10k
            ds = WildlifeReID10k(root, check_files=False)
            df = ds.metadata.copy()
            
            if subset != 'full':
                print(f"Filtering to subset: {subset}")
                df = df[df["dataset"].str.lower() == subset.lower()].copy()
                if df.empty:
                    print(f"Subset '{subset}' not found.")
                    sys.exit(1)
            return df
            
        except ImportError:
            print("WildlifeReID10k not available. Please install wildlife-datasets package.")
            sys.exit(1)

def validate_dataset_structure(dataset_name):
    """
    Validate that a dataset has the required structure for the pipeline.
    """
    
    base_dir = f"./data/{dataset_name}"
    required_files = [
        f"{base_dir}/processed_metadata.csv"
    ]
    
    required_dirs = [
        f"{base_dir}/dataset"
    ]
    
    print(f"🔍 Validating dataset structure for {dataset_name}...")
    
    # Check required files
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    # Check required directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"Dataset validation failed for {dataset_name}")
        if missing_files:
            print(f"Missing files: {missing_files}")
        if missing_dirs:
            print(f"Missing directories: {missing_dirs}")
        return False
    
    # Check metadata content
    try:
        df = pd.read_csv(f"{base_dir}/processed_metadata.csv")
        required_columns = ['image_id', 'identity', 'path', 'dataset']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing columns in metadata: {missing_columns}")
            return False
            
        print(f"Dataset validation passed for {dataset_name}")
        print(f"   - {len(df)} images")
        print(f"   - {df['identity'].nunique()} unique identities")
        return True
        
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False

# Update the save_count_results function to be compatible
def save_count_results_wrapper(row, path = COUNT_RESULTS_XLSX) :
    """Enhanced version that handles manual datasets properly."""
    
    # Add dataset type information
    manual_datasets = ['wild_boar', 'roe_deer']
    if row.get('Dataset', '').lower() in manual_datasets:
        row['Dataset Type'] = 'Manual'
    else:
        row['Dataset Type'] = 'WildlifeReID10k'
    
    save_count_results(row, path)
    
    
    
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


"""
def load_dataset(subset, root = WILD_DATASET_PATH):
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
"""

def combine_fisher_vectors(fv_list, weights):
    """Combine multiple Fisher vector dictionaries with given weights."""
    if len(fv_list) != len(weights):
        raise ValueError("Number of Fisher vector sets must match weights")

    combined = {}
    keys = fv_list[0].keys()
    for k in keys:
        vec = None
        for fv, w in zip(fv_list, weights):
            if k not in fv:
                continue
            v = fv[k]
            if vec is None:
                vec = w * v
            else:
                vec += w * v
        if vec is not None:
            combined[k] = vec
    return combined
