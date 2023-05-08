import os
import worker
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers and a distribution type.")

    parser.add_argument("n", type=int, help="specify the number of points")
    parser.add_argument("low", type=int, help="specify uncertainty level 1e-k low")
    parser.add_argument("high", type=int, help="specify uncertainty level 1e-k high")

    parser.add_argument("--distribution", type=str, choices=["all", "normal", "uniform", "cauchy"],
                        default="normal", help="distribution type--one of normal, uniform, or cauchy")

    parser.add_argument("--samples", type=int, default=1, help="number of samples to draw and simulate")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed passed to np.random.seed")

    parser.add_argument('--out', type=str, default=None, help='directory for result csv, no file written if None')

    args = parser.parse_args()

    dists = (stats.cauchy().rvs,
             stats.norm().rvs,
             stats.uniform().rvs)
    labels = ('cauchy', 'normal', 'uniform')

    for k in range(args.low, args.high):
        ϵ = 10 ** (-k)

        for dist, label in zip(dists, labels):
            if args.distribution in ('all', label):
                out = pd.DataFrame(worker.run_experiment(
                    args.n, ϵ, dist, samples=args.samples, verbose=True
                ))

                out.to_csv(
                    os.path.join(args.out, f'{label}_{args.n}_{k}.csv')
                )
