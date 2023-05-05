import time
import signal
import pickle
import numpy as np
import scipy.linalg as la

from tqdm.auto import trange
from typing import Any, Callable

import TurnpikeMM as MM
import turnpike_utils as utils

from MM_utils import MM_optimizer
from solver import TurnpikeSolver

double_array = np.ndarray[np.float64]


def print_metrics(z, zhat, ϵ, verbose: bool = True):
    if not verbose:
        return

    R = np.abs(z - zhat)
    print(f'sum={la.norm(R, ord=1)}, max={la.norm(R, ord=np.inf)}')
    print(f'ϵ={np.sum(R < ϵ)}, 2ϵ={np.sum(R < 2 * ϵ)}, '
          f'5ϵ={np.sum(R < 5 * ϵ)}, 10ϵ={np.sum(R < 10 * ϵ)}')


def run_experiment(n: int, ϵ: float, sampler: Callable[[int], double_array], samples: int,
                   verbose: bool = False):
    """
    :param n: positive int, number of points used for testing
    :param ϵ: positive float, tolerance scaling estimate (e.g., smallest distance)
    :param sampler: callable[[int], double_array], callable for sampling point vectors as sampler(n)
    :param samples: positive int, number of samples to draw
    :param verbose: bool, whether to print status messages

    :return dictionary with point and distance MAE, MSE, max-error, and bins for
    """

    results = []
    params = dict(Points=n, Noise=ϵ)

    for _ in trange(samples, leave=None):
        z = utils.canonical_vector(n, distribution=sampler)
        Q = utils.QMatrix(z.size)
        m, n = Q.shape
        d = Q @ z

        dnoise = d + ϵ * np.random.randn(d.size)
        np.clip(dnoise, 0., None, out=dnoise)  # Entries < 0 would never happen in a distance set, noisy or not

        dnoise.sort()
        dhat = np.empty(m)

        best, z_init = np.inf, None
        for _ in trange(n):
            zhat = utils.canonical_vector(n)
            MM.Qz(zhat, dhat)

            if MM.match_distances(dhat, dnoise) < best:
                z_init = zhat

        solver = lambda t: TurnpikeSolver(n, verbose=verbose).solve(dnoise, t, z_init)

        start = time.time()
        lo, hi = upper_bound(0, 10, ϵ, solver)
        (zo, dz1, dzn), τ = binary_search_tolerance(lo, hi, ϵ, 1e-1, solver, ground=z)

        signal.alarm(0)  # cancel lingering alarm, if any
        z1 = utils.fix_reflection(dz1 - dz1.mean())

        print_metrics(z, zo, ϵ, verbose)
        zhat = MM_optimizer(dnoise, ϵ=1e-12, zhat=zo, ground=z)
        elapsed = time.time() - start
        print_metrics(z, zo, ϵ, verbose)

        dhat = Q @ zhat

        result = {
            **params,
            **metrics('points', z, zhat, ϵ),
            **metrics('distances', d, dhat, ϵ),
            'Runtime (s)': elapsed
        }

        results.append(result)

    return results


def handler(*args):
    raise TimeoutError("end of time")


def safe_run(solver, σ: float):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)

    try:
        return solver(σ)
    except (TimeoutError, ValueError):
        pass
    finally:
        signal.alarm(0)

    return False


def upper_bound(lo: float, hi: float, τ: float, solver: Callable[[float], Any]):
    if safe_run(solver, τ * hi):
        return lo, hi

    return upper_bound(hi, 2 * hi, τ, solver)


def binary_search_tolerance(lo: float, hi: float, τ: float, ϵ: float,
                            solver: Callable[[float], Any], ground):
    mid = (lo + hi) / 2.
    print(f'{lo=}, {mid=}, {hi=}')

    out = safe_run(solver, mid * τ)

    if out and hi - lo < ϵ:
        return out, mid

    lo = lo if out else mid
    hi = mid if out else hi

    result = binary_search_tolerance(lo, hi, τ, ϵ, solver, ground)

    if not result:
        return out, mid

    return result


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def points_within_noise(y_true, y_pred, noise_level):
    R = np.abs(y_true - y_pred)
    return np.sum(R < noise_level)


def bin_max(y: np.ndarray[float], yhat: np.ndarray[float]) -> float:
    return np.abs(y - yhat).max()


def metrics(label: str, y: np.ndarray[float], yhat: np.ndarray[float], ϵ) -> dict[str, Any]:
    return {
        f'{label} MSE': mse(y, yhat),
        f'{label} MAE': mae(y, yhat),
        f'{label} bin-max': bin_max(y, yhat),
        f'{label} binned ϵ': points_within_noise(y, yhat, ϵ),
        f'{label} binned 2ϵ': points_within_noise(y, yhat, 2 * ϵ),
        f'{label} binned 5ϵ': points_within_noise(y, yhat, 5 * ϵ),
        f'{label} binned 10ϵ': points_within_noise(y, yhat, 10 * ϵ),
    }

