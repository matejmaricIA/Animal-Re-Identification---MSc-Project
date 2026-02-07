#!/bin/bash
# Script to estimate population counts for multiple datasets using HITL Nested-IS
params=("ATRW" "CowDataset" "IPanda50" "NyalaData" "SealID" "BelugaID" "HyenaID2022" "StripeSpotter" "Giraffes")

for p in "${params[@]}"; do
    echo "Running count pipeline for dataset: $p"
    python main.py \
      --ds "$p" \
      --count \
      --save_count \
      --num_vertices 750 \
      --num_neighbors 750 \
      --use_lightglue \
      --gv_matcher lightglue \
      --use_global_embedding \
      --use_fisher \
      --method ensamble \
      --count_proposal_mode calibrated \
      --count_local_evidence inliers \
      --count_local_mu 0.5 \
      --count_shortlist_B 300 \
      --count_mix_alpha 0.9 \
      --count_cal_pairs 500 \
      --count_cal_shortlist 300 \
      --count_cal_negs_per_query 50
    
done
