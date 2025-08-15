#!/bin/bash
# Script to estimate population counts for multiple datasets using Nested Importance Sampling
params=("ATRW" "CowDataset" "IPanda50" "NyalaData" "SealID" "BelugaID" "HyenaID2022" "StripeSpotter" "Giraffes")

for p in "${params[@]}"; do
    echo "Running count pipeline for dataset: $p"
    # Standard runs
    python main.py --ds "$p" --count --num_vertices 750 --num_neighbors 750 --use_geometric_verification --use_lightglue --save_count --method ensamble --gv_threshold 0.95 --automated_mode --use_global_embedding
    python main.py --ds "$p" --count --num_vertices 750 --num_neighbors 750 --use_geometric_verification --use_lightglue --save_count --method ensamble --gv_threshold 0.95 --automated_mode --use_global_embedding
    
done