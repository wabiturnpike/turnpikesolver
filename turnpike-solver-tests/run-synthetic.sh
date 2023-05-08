#!/bin/bash

# Create the runs directory if it doesn't exist
mkdir -p synthetic

for n in $(seq 25 25 100); do
  # Runs n in range(25, 2025, 25) for eps in 10 ** -[1, 12] over all dists
  python3.10 run-turnpike-instance.py $n 1 13 --distribution=all --out=synthetic --samples=10
done
