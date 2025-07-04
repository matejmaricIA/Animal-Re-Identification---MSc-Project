#!/bin/bash
#"CowDataset" "GiraffeZebraID" "Giraffes" "HyenaID2022" "IPanda50" "LeopardID2022" "NyalaData" "PolarBearVidID" "SealID" "StripeSpotter" "SeaStarReID2023" "SeaTurtleID" "WhaleSharkID" "BelugaID" "NDD20"
params=("ATRW" "CowDataset" "IPanda50" "NyalaData" "SealID" "BelugaID" "HyenaID2022" "StripeSpotter" "Giraffes")
for p in "${params[@]}"; do
    echo "Running script with parameter: $p"
    python main.py --ds "$p" --train --method ensamble --save_eval --use_geometric_verification --use_lightglue
    python main.py --ds "$p" --train --method disk --save_eval --use_geometric_verification --use_lightglue
    python main.py --ds "$p" --train --method keynet_hardnet --save_eval --use_geometric_verification --use_lightglue
    python main.py --ds "$p" --train --method lightglue --save_eval --use_geometric_verification --use_lightglue
done

