#!/bin/bash
# Script to estimate population counts for multiple datasets using Nested Importance Sampling
params=("ATRW" "CowDataset" "IPanda50" "NyalaData" "SealID" "BelugaID" "HyenaID2022" "StripeSpotter" "Giraffes")

for p in "${params[@]}"; do
    echo "Running count pipeline for dataset: $p"
    python main.py --ds "$p" --count --num_vertices 250 --num_neighbors 250 --use_geometric_verification --use_lightglue --remove_background --save_count
done