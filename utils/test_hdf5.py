import h5py
from pathlib import Path

def inspect_h5(path: str, query: str, n: int = 10) -> None:
    """Print sample keys and report how `query` is stored (if present)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with h5py.File(p, "r") as f:
        keys = list(f.keys())
        print(f"{len(keys)} total entries in {p.name}")
        print("First keys:", keys[:n], "\n")

        if query in f:
            dset = f[query]
            print(f"Found exact key '{query}':", dset.shape, dset.dtype)
        else:
            padded_matches = [k for k in keys if k.endswith(query)]
            if padded_matches:
                print("Exact key not found; possible matches:", padded_matches)
                dset = f[padded_matches[0]]
                print("Example dataset shape/dtype:", dset.shape, dset.dtype)
            else:
                print(f"No key matching '{query}'")

# Example usage
inspect_h5(
    "../data/ATRW/feature_descriptors_test_disk_segmented/descriptors.h5",  
    "2914"                      
)
