import os
import sys
import pickle
import pandas as pd
from pathlib import Path
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

    subset_str = str(subset)

    # Prefer the curated all_datasets.csv when available.
    all_datasets_path = Path("./data/all_datasets.csv")
    if all_datasets_path.exists():
        df_all = pd.read_csv(
            all_datasets_path,
            dtype={"image_id": str, "identity": str, "dataset": str},
        )
        if subset_str.lower() == "full":
            print(f"Loading all datasets from {all_datasets_path}")
            return df_all
        df_sub = df_all[df_all["dataset"].str.lower() == subset_str.lower()].copy()
        if not df_sub.empty:
            print(f"Loading dataset '{subset_str}' from {all_datasets_path}")
            return df_sub
        print(f"Dataset '{subset_str}' not found in {all_datasets_path}, falling back.")

    # Prefer local metadata when available (covers custom datasets and cached preprocessed subsets).
    local_candidates = [
        Path("./data") / subset_str / "processed_metadata.csv",
        Path("./data") / subset_str.lower() / "processed_metadata.csv",
    ]
    for metadata_path in local_candidates:
        if metadata_path.exists():
            print(f"Loading local dataset metadata: {metadata_path}")
            df = pd.read_csv(metadata_path, dtype={"image_id": str})
            print(f"Loaded {len(df)} images from local dataset '{subset_str}'")
            return df

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
                print(f"Subset '{subset}' not found in WildlifeReID10k.")
                print(
                    f"If '{subset}' is a custom dataset, create `./data/{subset}/processed_metadata.csv` "
                    f"(for Chicks4FreeID: `python utils/import_chicks4freeid.py`)."
                )
                sys.exit(1)
        return df

    except ImportError:
        print("WildlifeReID10k not available. Please install wildlife-datasets package.")
        sys.exit(1)


def _normalize_path_value(value: str) -> str:
    if value is None:
        return ""
    path = str(value).strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _normalize_path_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_path_value)


def _find_official_metadata_csv(dataset_name: str, metadata_root: Path) -> Path | None:
    candidates = [
        metadata_root / dataset_name / "metadata.csv",
        metadata_root / dataset_name.lower() / "metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _apply_official_split_from_csv(
    df: pd.DataFrame, metadata_csv: Path, dataset_label: str
) -> pd.DataFrame:
    official = pd.read_csv(metadata_csv, dtype={"image_id": str})
    if "split" not in official.columns:
        raise ValueError(f"Official metadata missing 'split' column: {metadata_csv}")

    df = df.copy()
    if "image_id" in df.columns and "image_id" in official.columns:
        df_keys = df["image_id"].astype(str)
        official_keys = official["image_id"].astype(str)
        match_label = "image_id"
    elif "path" in df.columns and "path" in official.columns:
        df_keys = _normalize_path_series(df["path"])
        official_keys = _normalize_path_series(official["path"])
        match_label = "path"
    else:
        raise ValueError(
            f"Cannot match official metadata for '{dataset_label}': "
            "missing image_id or path columns."
        )

    split_map = pd.Series(
        official["split"].astype(str).values, index=official_keys
    ).to_dict()
    df["split_official"] = df_keys.map(split_map)

    matched = int(df["split_official"].notna().sum())
    total = len(df)
    print(
        f"Matched {matched}/{total} rows to official split for '{dataset_label}' "
        f"using {match_label}."
    )
    if matched < total:
        print(
            f"[WARN] Dropping {total - matched} rows without official split for "
            f"'{dataset_label}'."
        )
    df = df.loc[df["split_official"].notna()].copy()
    df["split"] = df["split_official"]
    df = df.drop(columns=["split_official"])
    return df


def apply_official_wildlifetools_split(
    df: pd.DataFrame,
    dataset_name: str,
    metadata_root: Path | None = None,
) -> pd.DataFrame:
    """Replace local splits with WildlifeTools official per-dataset splits."""
    if metadata_root is None:
        metadata_root = (
            Path(__file__).resolve().parent
            / "baselines"
            / "data"
            / "metadata"
            / "datasets"
        )
    else:
        metadata_root = Path(metadata_root)

    dataset_name_str = str(dataset_name)
    if dataset_name_str.lower() == "full":
        if "dataset" not in df.columns:
            print(
                "[WARN] Cannot apply official splits for full dataset: "
                "missing 'dataset' column."
            )
            return df
        frames = []
        for ds_value, ds_df in df.groupby("dataset", sort=False):
            if pd.isna(ds_value):
                print("[WARN] Missing dataset label; skipping official split.")
                frames.append(ds_df)
                continue
            metadata_csv = _find_official_metadata_csv(str(ds_value), metadata_root)
            if metadata_csv is None:
                print(
                    f"[WARN] Official metadata not found for dataset '{ds_value}'. "
                    "Keeping existing split."
                )
                frames.append(ds_df)
                continue
            frames.append(
                _apply_official_split_from_csv(ds_df, metadata_csv, str(ds_value))
            )
        return pd.concat(frames, ignore_index=True)

    metadata_csv = _find_official_metadata_csv(dataset_name_str, metadata_root)
    if metadata_csv is None:
        print(
            f"[WARN] Official metadata not found for dataset '{dataset_name_str}' "
            f"under {metadata_root}. Keeping existing split."
        )
        return df

    return _apply_official_split_from_csv(df, metadata_csv, dataset_name_str)

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
    
    
    
def save_stuff(pca, gmm, fisher_vectors, paths):
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
