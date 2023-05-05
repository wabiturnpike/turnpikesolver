import numpy as np
import cvxpy as cvx
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

import TurnpikeMM as MM
import turnpike_utils as utils

from typing import Optional


def MM_optimizer(d: np.ndarray,
                 ϵ: float = 0.,
                 its: int = np.inf,
                 verbose: bool = False,
                 zhat: Optional[np.ndarray] = None,
                 ground: Optional[np.ndarray] = None):
    """
        Data:
        @param d: float array, distance vector in sorted order

        Functionality:
        @param ϵ: positive float, norm residual sentinel value
        @param its: positive int, maximum iterations to run the solver
        @param zhat: float vector, used as initializer for optimization when provided

        Logging:
        @param verbose: bool, prints progress-tracking messages when True
        @param ground: float array, a ground truth vector for testing optimizer performance

        @returns: MM-optimized point vector zhat
    """

    m, n = d.size, utils.invert_choose2(d.size)

    MM_optimizer.tracker = []
    Q, P = MM.build_sparse_matrix(n), utils.DistanceProjector(d)

    # Sets up canonical points
    if zhat is None:
        zhat = utils.canonical_vector(n)

    zhat = zhat.copy()
    P.project(zhat, its=10_000)

    dhat = Q @ zhat
    MM.match_distances(dhat, d)  # Set up initializer

    t = 0
    score2, score1 = np.inf, 0.
    while (t := t + 1) < its and abs(score2 - score1) > ϵ:
        # Step 1: optimize point vector
        MM.dQ(dhat, zhat)
        zhat = Q.T @ dhat

        # Step 2: enforce symmetry-breaking constraints
        # This prevents multiple solutions from pulling the optimizer in
        # different directions, which would leave us stuck in a cycle.
        zhat = P.project(zhat, its=10)

        # Step 3: Reorder ground distances in agreement with optimized ordering
        MM.Qz(zhat, dhat)

        score1 = score2
        score2 = MM.match_distances(dhat, d)

        utils.if_then_print(verbose, f'||z{t-1} - z{t}|| = {abs(score1 - score2):.12f}')
        utils.if_then_print(verbose, f'Iteration {t} score: {score2:.12f}')

        if ground is not None:
            utils.if_then_print(verbose, f'l1-ground-residual: {la.norm(ground - zhat, ord=1)}', end='\n\n')

    return zhat


class _Identity:
    # acts as the identity operator--avoids space from np.eye or sp.eye
    def __init__(self, n: int):
        self.n = n

    def __matmul__(self, other):
        return other

    def __rmatmul__(self, other):
        return other

    @property
    def T(self):
        return self


def build_sorting_permutation_T(v: np.ndarray, Γ: Optional[sparse.spmatrix] = None):
    """
        :param v: vector used for ordering
        :param Γ: optional permutation matrix that can be used to pivot v closer to sorted position
                  for a faster sort time

        :return: sorting permutation transpose
    """
    n = v.size
    Π = sparse.dok_array((n, n))

    Γ = Γ if Γ else _Identity(n)
    kind = 'stable' if Γ else 'quicksort'

    # Much faster sorting when Γ almost sorts v, virtually no cost when not passed
    # kind='stable' needed for almost Γ to matter
    I, J = range(n), np.argsort(Γ.T @ v, kind=kind)

    # Make transpose since that's what we typically want
    Π[J, I] = 1

    # ΠᵀΓᵀ v = SORT v <---> SORT.T = ΓΠ
    return Γ @ Π

