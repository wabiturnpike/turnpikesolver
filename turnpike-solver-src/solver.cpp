import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import TurnpikeMM as MM
import turnpike_utils as utils

from MM_utils import MM_optimizer


class TurnpikeSolver:
    def __init__(self, n: int, verbose: int = 0):
        """
            @param n: int, number of points, used to optimize space usage
            @param verbose: int in {0, 1, 2}, more messages will be printed at each level
        """

        assert isinstance(n, int) and n > 0, "number of points must be a positive integer"

        self.projector = None
        self.Q = utils.QMatrix(n)

        # cvxpy solver and size index
        self.idx = 0

        # Placed distances, backtracking matrix, and solution vector
        self.Δ, self.P, self.b = None, None, None

        # Solver instance specific variables, subject to change without warning
        self.values = None

        self.verbose = verbose
        self.n, self.m = n, utils.choose2(n)

    def __repr__(self):
        n, verbose = self.n, self.verbose
        return f'TurnpikeSolver({n=}, {verbose=})'

    def solve(self, distances: np.ndarray, ϵ: float, z_init: np.ndarray, MM_its: int = 10) -> np.ndarray:
        """
            Solves Turnpike instances with an EM-augmented backtracking algorithm.

            @param distances: non-negative float array with n choose 2 entries
            @param ϵ: non-negative float, tolerance for fuzzy distance matching
            @param z_init: numpy array, used as the initializer for the MM steps
            @param MM_its: non-negative integer, number of iterations to run the MM initializer

            @returns: matching graph, quality score, z
        """

        assert z_init.size == self.n, "initializer must be the same size"

        # Must be 64 bit precision to avoid numerical bugs
        D = distances.astype(np.float64)
        D.sort()

        self.projector = utils.DistanceProjector(D)

        self.idx = 0
        m, n, values = self.m, self.n, dict()

        # Implicit storage of an interval sparse matrix
        self.L = np.empty(m, dtype=np.int32)
        self.R = np.empty(m, dtype=np.int32)
        self.Δ = np.empty(m, dtype=np.float64)
        self.I = np.full(m, -1, dtype=np.int32)
        self.T = np.empty(n, dtype=np.float64)

        # Temporary variables for this call to solver
        # Subject to change without warning
        values['ϵ'] = ϵ
        values['D'] = D
        values['l1'] = D.sum()
        values['l2'] = la.norm(D)
        values['branch_count'] = 0

        values['step'] = utils.step_vector(n)
        values['ds'] = la.norm(values['step']) ** 2

        # Distances from z1 and zn
        # np.inf represents an unmatched piece
        values['dzn'] = np.full(n, np.inf)
        values['dz1'] = np.full(n, np.inf)

        # Initial MM guess subject to symmetry-breaking constraints

        zhat = z_init.copy()

        self.projector.project(zhat)
        if self.verbose:
            print('Running MM setup…')

        values['Z'] = MM_optimizer(D, ϵ=ϵ, verbose=False, zhat=zhat, its=MM_its)

        values['best_Z'] = values['Z']
        values['r'] = values['l2'] / np.sqrt(n)

        # Corner case: must place zn - z1 here
        values['dzn'][-1] = 0.
        values['dzn'][ 0] = D[-1]

        values['dz1'][ 0] = 0.
        values['dz1'][-1] = D[-1]

        self.values = values

        values['cost'] = 0.
        values['best_score'] = np.inf

        self.match_distances(
            np.asarray([D[-1]]),
            np.asarray([D[-1] - ϵ]),
            np.asarray([D[-1] + ϵ]),
        )

        self.T[ 0] = 1.
        self.T[-1] = ϵ
        self.backtrackMM(i=1, j=n-2, ϵ=ϵ)

        z, dz1, dzn = values['Z'], values['dz1'], values['dzn']

        # Solve Pz = b for the least-squares solution using the backtracked vectors
        # This is advantageous in the noisy case because it helps evens out bad choices.
        # The vector T stores the confidence radius when a distance was placed and allows
        # us to scale vectors to solve weighted least squares.

        # Scale by stepping by confidence.
        # This term dominates the magnitude otherwise.
        sz = values['l1']
        sv = utils.step_vector(n)

        self.P, self.b = sp.lil_array((n, n), dtype=np.float64), dz1.copy()
        K, P, b = range(1, n), self.P, self.b

        T = self.T
        T[1:] /= ϵ
        T **= -.5

        P[0] = sv
        b[0] = sz
        P[K, 0] = -1
        P[K, K] =  1

        zhat = sla.lsqr(P, b, atol=1e-12, btol=1e-12)[0]
        zhat = utils.fix_reflection(zhat)

        self.projector.project(zhat, its=100 * self.n)

        return zhat, dz1, dzn

    def backtrackMM(self, i: int, j: int, ϵ: float):
        """
            Performs incidence-backtracking with an MM-step deciding which split to take.

            @param i: int, gives the range Dn[:i] currently assigned
            @param j: int, gives the range D1[j+1:] currently assigned
            @param ϵ: float, noise tolerance radius for assigning distance

            @returns: an MM reconstruction subject to solver constraints
        """

        if i > j:
            return i, j

        m, n, values = self.m, self.n, self.values
        D, dzn, dz1 = values['D'], values['dzn'], values['dz1']
        dr = D[-1]

        idx, d = m, None
        while d is None and (idx := idx - 1) >= 0:
            if self.I[idx] == -1:
                d = D[idx]

        τ = values['ϵ']
        tols = [ϵ] + [ϵ + (τ / 8.), ϵ + (τ / 4.)]

        τ = ϵ
        for ϵ in tols:
            # Branch 1 params
            one = [i, d, dr - d, 2 * ϵ, 3 * ϵ, 1, 0]

            # Branch 2 params
            two = [j, dr - d, d, 3 * ϵ, 2 * ϵ, 0, 1]

            # Test branch dmax = zn - zi
            dzn[i], dz1[i] = d, dr - d
            z1, score1 = self.back_and_forth((i, n-1), d)
            one.append(score1)
            one.append(z1)
            dz1[i], dzn[i] = np.inf, np.inf

            # Test branch dmax = zj - z1
            dz1[j], dzn[j] = d, dr - d
            z2, score2 = self.back_and_forth((0, j), d)
            two.append(score2)
            two.append(z2)
            dzn[j], dz1[j] = np.inf, np.inf

            # Search in order of MM scores
            if score2 < score1:
                one, two = two, one
                score1, score2 = score2, score1

            if self.verbose > 0:
                print(
                    f'placed={d:.6f}, i={i}, j={j}, tol={ϵ}, '
                    f'match-cost={values["cost"] / utils.choose2(i + (n-j))}, '
                    f'MAE={np.nanmin([score1, score2]) / m:.8f}, '
                    f'best={values["best_score"] / m}'
                )

            zorg = values['Z']
            for idx, dn, d1, ϵn, ϵ1, Δi, Δj, score, z in (one, two):
                # MM found an invalid branch
                if np.isnan(score):
                    continue

                # Required distances within tolerance
                δ = np.hstack([dzn[:i] - dn, dz1[j + 1:] - d1])
                δlo = δ.copy()
                δhi = δ.copy()

                # Error propagates differently when we place the distance from the perspective of z1 or zn.
                # For example, if we placed distance zn - z3 earlier and are placing zn - z4 now,
                # then zn - z3 - (zn - z4) = z4 - z3 subject to a 2ϵ noise bound.
                # On the other hand, if we placed distance z3 - z1 earlier, we have to convert that distance via
                # (zn - z1) - (zn - z4) = (z4 - z1) - (z3 - z1) = z4 - z3 subject to a 3ϵ noise bound
                δlo[:i] -= ϵn
                δlo[i:] -= ϵ1

                δhi[:i] += ϵn
                δhi[i:] += ϵ1

                # Find a minimizing match within tolerances
                M = self.match_distances(δ, δlo, δhi)

                # Matching failed--try the other branch
                if not M:
                    continue

                values['branch_count'] += 1

                μ, I, *_ = M
                values['cost'] += μ
                μ = values['cost'] / utils.choose2(δ.size)

                dz1[idx] = d1
                dzn[idx] = dn
                values['Z'] = z

                # Convex factor to
                # increase tolerance early in the tree,
                # maintain tolerance midway in the tree,
                # increase tolerance late in the tree.
                # ξ = values['ϵ'] * abs((utils.choose2(δ.size + 1) / m) - .5) ** 1.5

                # Use matching cost mean to scale new error propagation
                self.T[idx] = τ
                reconstructed = self.backtrackMM(i + Δi,
                                                 j - Δj,
                                                 ϵ)

                if reconstructed:
                    ri, rj = reconstructed
                    if ri > rj:
                        return i, j

                    return reconstructed

                t = utils.choose2(δ.size)
                MM.undo_matching(I, t)

                values['Z'] = zorg
                dzn[idx] = np.inf
                dz1[idx] = np.inf

        return False

    def match_distances(self, δ: np.ndarray, δlo: np.ndarray, δhi: np.ndarray):
        """
            Adds distance estimates to matching graph and finds a weight-minimizing assignment based
            on most recently matched ground distance d.

            @param   δ: interval centers estimated from d, dz1, and dzn.
            @param δlo: interval lower-bounds estimated from δ, dz1, and dzn
            @param δhi: interval upper-bounds estimated from δ, dz1, and dzn
        """

        if self.verbose == 2:
            print('Matching...')

        ϵ, m, D = self.values['ϵ'], self.m, self.values['D']

        matched = utils.choose2(δ.size)
        Δ, L, R, I = self.Δ, self.L, self.R, self.I

        # We find edges by:
        # --- 1. Binary searching for lower bound on matching
        # --- 2. Binary searching for upper bound on matching
        # --- 3. Constructing edges between every ground distances in the bounds
        # --- 4. Adding edge weights based on |ground - estimate|
        E = matched + δ.size
        L[matched:E] = np.searchsorted(D, δlo, side='left')
        R[matched:E] = np.searchsorted(D, δhi, side='right')

        if np.any(L[matched:E] == R[matched:E]):
            return False

        Δ[matched:E] = δ
        cost, I = MM.online_matching(D, δ, I, matched)

        if cost == np.inf:
            print('Infinite cost!')
            return False

        return cost, I

    def back_and_forth(self, constraint: tuple[int, int],
                       value: float) -> tuple[np.ndarray, float]:
        """
            @param constraint: index tuple, (i, j) with i < j indicating z[j] - z[i] = value
            @param value: float, constraint value to enforce

            @returns: MM estimate for the point vector, sorted-l1-distance-residual
        """

        if self.verbose == 2:
            print('Optimizing...')

        values = self.values
        D, Z = values['D'], values['Z'].copy()

        step = values['step']

        # Derived symmetry-breakers
        r, d1, ds, dmax = values['r'], values['l1'], values['ds'], D[-1]

        r = values['r']
        i, j = constraint

        Dhat = np.empty(self.m)
        for _ in range(3):
            # Step 0: calculate distances
            MM.Qz(Z, Dhat)

            # Step 1: optimize permutation
            MM.match_distances(Dhat, D, 10)

            # Step 2: optimize point vector
            MM.dQ(Dhat, Z)
            utils.normalize(Z, r)

            # Step 3: constraint enforcement
            # --- 1. Sorting constraint
            # --- 2. Stepping constraint
            # --- 3. Reflection constraint
            # --- 4. l2-norm constraint
            # --- 5. new dij constraint
            Z.sort()

            ps = (d1 - (step @ Z)) / ds
            utils.normalize(
                np.add(Z, ps * step, out=Z), r
            )

            Z = utils.normalize(Z, r)

            for _ in range(20):
                Z.sort()

                rij = value - (Z[j] - Z[i])
                Z[i] -= rij / 2.
                Z[j] += rij / 2.

                ps = (d1 - (step @ Z)) / ds
                utils.normalize(
                    np.add(Z, ps * step, out=Z), r
                )

                Z = utils.normalize(Z, r)

        if self.verbose == 2:
            print('Done optimizing!')

        MM.match_distances(Dhat, D)
        score = la.norm(np.subtract(Dhat, D, out=Dhat), ord=1)

        if score < values['best_score']:
            values['best_Z'] = Z
            values['best_score'] = score

        return Z, score


if __name__ == '__main__':
    np.random.seed(1)

    BIN_TOL, BIN_EPS, MAE, MSE, MAX = [], [], [], [], []

    ϵs = np.asarray([
        # 1e-11, 1e-10, 1e-9, 1e-8, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
        1e-6
        # 1e-3, 1e-2
    ])

    σs = ϵs
    τs = ϵs

    # Small tolerance for noiseless setting
    # τs[0] = 1e-15
    n = 500
    m = utils.choose2(n)

    Q = utils.QMatrix(n)
    # z = utils.canonical_vector(n, distribution=lambda k: np.random.uniform(0, 1, k))
    points = np.arange(3 * n)

    np.random.shuffle(points)
    z = np.random.choice(points, size=n, replace=False).astype(np.float64)
    z.sort()
    z -= z[ 0]
    z /= z[-1]
    z = utils.fix_reflection(z - z.mean())

    d = Q @ z
    d.sort()
    for σ, τ in zip(σs, τs):
        print(σ, τ)

        # Add centered Gaussian noise with stddev σ
        noise = np.random.randn(d.size)
        noise = np.multiply(σ, noise, out=noise)

        # Add Gaussian noise to the distances
        dnoise = np.add(d, noise, out=noise)
        dnoise.sort()

        # Run solver within 5 standard-deviations tolerance of noise
        # Tradeoff with tolerance between runtime and performance
        # Verbose toggles debugging messages.
        solver = TurnpikeSolver(n, verbose=1)

        # Solve with tolerance value τ
        zo, dz1, dzn = solver.solve(dnoise, 1e-5, utils.canonical_vector(n))
        z1 = dz1 - dz1.mean()

        P = utils.DistanceProjector(dnoise)
        zo, z1 = utils.fix_reflection(zo), utils.fix_reflection(z1)

        labels = ('Solver', 'Z[0] Distances')
        for label, zinit in zip(labels, (zo, z1)):
            print(f'{label} zhat')

            zopt = MM_optimizer(dnoise, ϵ=1e-12, ground=z,
                                zhat=zinit.copy(), verbose=False)

            for zhat in (zinit, zopt):
                R = np.abs(z - zhat)

                MAX.append(np.max(R))

                BIN_EPS.append(np.sum(R < σ))
                BIN_TOL.append(np.sum(R < 10 * σ))

                MSE.append(la.norm(R) ** 2 / n)
                MAE.append(la.norm(R, ord=1) / n)

                print(f'Binned by tolerance: {BIN_TOL[-1]}, '
                      f'Binned by standard deviation: {BIN_EPS[-1]}, '
                      f'MAE: {MAE[-1]}, MSE: {MSE[-1]}, MAX: {MAX[-1]}')

            print()
