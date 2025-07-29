#!/bin/bash

models=("lcdm" "wcdm" "cpl")
likelihoods=("desidr2" "pantheonplus" "des5y" "desidr2 pantheonplus" "desidr2 des5y")

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
