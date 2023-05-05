import worker
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers and a distribution type.")

    parser.add_argument("n", type=int, help="specify the number of points")
    parser.add_argument("k", type=int, help="specify uncertainty level 1e-k")

    parser.add_argument("--distribution", type=str, choices=["normal", "uniform", "cauchy"],
                        default="normal", help="distribution type--one of normal, uniform, or cauchy")

    parser.add_argument("--samples", type=int, default=1, help="number of samples to draw and simulate")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed passed to np.random.seed")
    parser.add_argument("--out", type=str, default=None, help="location to write out run csv")

    args = parser.parse_args()

    n, ϵ = args.n, 10 ** (-args.k)

    match args.distribution:
        case 'cauchy':
            draw = stats.cauchy().rvs
        case 'normal':
            draw = stats.norm().rvs
        case 'uniform':
            draw = stats.uniform().rvs
        case _:
            raise RuntimeError("Invalid distribution.")

    np.random.seed(args.seed)
    out = pd.DataFrame(
        worker.run_experiment(n, ϵ, draw, args.samples, verbose=False)[-1]
    )

    if args.out is not None:
        out.to_csv(args.out)

