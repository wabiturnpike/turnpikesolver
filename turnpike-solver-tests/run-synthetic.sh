#!/bin/bash

# Create the runs directory if it doesn't exist
mkdir -p runs

# Loop through n values
for n in $(seq 25 25 2000); do
    # Loop through k values (convert to ε)
    for k in $(seq -12 1 -1); do
        # Loop through distribution types
        for dist in normal uniform cauchy uint; do
            # Generate a unique output filename based on n, k, and distribution
            out_file="runs/run_n_${n}_k_${k}_${dist}.csv"

            # Call the Python interface with the specified arguments
            python run-turnpike-instance.py $n $k --distribution $dist --out $out_file
        done
    done
done
