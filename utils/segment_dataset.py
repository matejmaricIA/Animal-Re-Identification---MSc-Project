#!/usr/bin/env python3
import argparse
import os

from preprocessing import preprocess_dataset
from utility_functions import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Segment a dataset with Grounded SAM2.")
    parser.add_argument("--ds", required=True, help="Dataset name")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to ./data/<ds>/segmented_dataset)",
    )
    parser.add_argument(
        "--use_mantiuk",
        action="store_true",
        help="Apply Mantiuk tone mapping before segmentation",
    )
    args = parser.parse_args()

    df = load_dataset(args.ds)
    base_dir = os.path.join("./data", args.ds)
    output_dir = args.output_dir or os.path.join(base_dir, "segmented_dataset")

    preprocess_dataset(
        df,
        output_dir,
        args.ds,
        use_mantiuk=args.use_mantiuk,
        remove_background=True,
    )


if __name__ == "__main__":
    main()
