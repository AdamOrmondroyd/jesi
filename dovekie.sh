#!/bin/bash

models=("lcdm" "wcdm" "cpl")
likelihoods=("desidr2" "des5y" "dovekie" "desidr2 des5y" "desidr2 dovekie" "des5yoffset" "dovekieoffset" "desidr2 des5yoffset" "desidr2 dovekieoffset")

for model in "${models[@]}"; do
    for likelihood in "${likelihoods[@]}"; do
        echo "$model $likelihood"
        uv run run.py "$model" $likelihood
        echo ""
    done
done

for i in {2..20}; do
    for likelihood in "${likelihoods[@]}"; do
        echo "flexknot $i $likelihood"
        uv run run.py "flexknot" $likelihood --n "$i"
        echo ""
    done
done
