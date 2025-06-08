#!/bin/bash

params=("Cows2021" "DogFaceNet" "GiraffeZebraID" "Giraffes" "HyenaID2022" "IPanda50" "LeopardID2022" "NyalaData" "PolarBearVidID" "SealID" "StripeSpotter")

for p in "${params[@]}"; do
    echo "Running script with parameter: $p"
    python main.py --ds "$p" --train
done
