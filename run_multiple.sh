#!/bin/bash
#"CowDataset" "GiraffeZebraID" "Giraffes" "HyenaID2022" "IPanda50" "LeopardID2022" "NyalaData" "PolarBearVidID" "SealID" "StripeSpotter" "SeaStarReID2023" "SeaTurtleID" "WhaleSharkID" "BelugaID" "NDD20"
params=("ATRW")

for p in "${params[@]}"; do
    echo "Running script with parameter: $p"
    python main.py --ds "$p" --train --remove_background --method disk
done
