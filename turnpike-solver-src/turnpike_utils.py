import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

from typing import Callable, Union
from scipy.sparse.linalg import LinearOperator

import TurnpikeMM as MM

double_array = np.ndarray[np.float64]


def if_then_print(condition: bool, *args, **kwargs) -> None:
    """
        :param condition: bool to decide if print called

        Wrapper function to prevent excessive nesting from conditional print calls.
    """

    if condition:
        print(*args, **kwargs)


class QMatrix(LinearOperator):
    def __init__(self, n):
        self.n = n
        self.m = choose2(n)

        self.dtype = np.float64
        self.shape = (self.m, self.n)

    def _matvec(self, z, d=None):
        if z.ndim != 1 or z.size != self.n:
            raise ValueError("Input array must be 1-dimensional and have the same size "
                             "as the operator.")

        if d is None:
            d = np.empty(self.m, dtype=np.float64)

        MM.Qz(z, d)

        return d

    def _rmatvec(self, d, z=None):
        if d.ndim != 1 or d.size != self.m:
            raise ValueError("Input array must be 1-dimensional and have the same size as "
                             "the operator's first dimension.")

        if z is None:
            z = np.zeros(self.n, dtype=np.float64)

        MM.dQ(d, z)

        return z


def measure(n: int, i: int, j: int):
    q = np.zeros(n)
    q[j], q[i] = 1, -1

    return q


class DistanceProjector:
    """
        This class implement an interface to enforce known constraints on the vector z
        derived from distance measurements.
    """

    def __init__(self, d: double_array):
        """
            :param d: distance array
        """
        self.n = invert_choose2(d.size)
        d1, dmax, self.r = get_distance_info(d)

        # Set up constraint projector b = WPz
        self.P = step_vector(self.n).reshape(1, -1)
        self.tol = np.asarray([d.size])
        self.b = np.asarray([d1])
        self.w = np.asarray([
            la.norm(self.P, ord=2) ** 2
        ])

        self.add_distance(0, -1, dmax)

    def add_distance(self, i: int, j: int, d: float):
        # Distance i, j measurement vector
        measure = np.zeros((1, self.n))
        measure[0, j], measure[0, i] = 1, -1

        # Add new measurement vector to the matrix
        self.P = np.r_[self.P, measure]

        # Update w, b, and tol for new measurement
        self.w = np.hstack([self.w, [2]])
        self.w[0] -= 2 * self.P[0, j] ** 2

        self.b = np.hstack([self.b, [d]])
        self.b[0] -= self.P[0, j] * d

        self.tol = np.hstack([*self.tol, [2]])

        # Maintain orthogonality
        self.P[0, i], self.P[0, j] = 0., 0.

    def remove_most_recent_distance(self):
        i, j = np.nonzero(self.P[-1])[0]

        # Explicit formula for removed step vector entries
        sk = (self.n - 1) - 2 * i
        self.P[0, i], self.P[0, j] = -sk, sk

        # Add removed magnitude back
        self.w[0] += 2 * sk ** 2

        # Remove last entries now
        self.w.pop()
        self.b.pop()
        self.P = self.P[:-1]

    def project(self, zhat: double_array, its: int = 1_000):
        assert zhat.size == self.n, "z must be of length n"

        for _ in range(its):
            # Step 1: project onto r-sphere
            normalize(zhat, self.r)

            # Step 2: project onto affine-measurement-space
            bz = (self.b - self.P @ zhat) / self.w
            np.add(zhat, bz @ self.P, out=zhat)

            # Step 3: project onto sorted simplex, order canonically
            zhat.sort(kind='stable')
            zhat = fix_reflection(zhat)

        return zhat


def enforce_distance(zhat: np.ndarray, dij: float, i: int, j: int):
    # Calculate disagreement
    dhat_ij = zhat[j] - zhat[i]
    to_move = (dij - dhat_ij) / 2.

    # Force to correct range
    zhat[j] += to_move
    zhat[i] -= to_move

    return zhat


def choose2(n: int) -> int:
    return (n * (n-1)) // 2


def normalize(z: np.ndarray, r: float = 1.):
    return np.multiply(z, r / la.norm(z), out=z)  # warning: normalized in-place


def fix_reflection(z: np.ndarray) -> np.ndarray:
    if abs(z[0]) < abs(z[-1]):
        z = -z[::-1]

    return z


def invert_choose2(m: int) -> int:
    """
        Returns n when given m = n choose 2 for unknown n.

        Raises an assertion error if m != n (n-1) / 2
    """

    # Solve for positive root
    # P(n) = n**2 - n - .5 * m
    n = round(.5 + np.sqrt(2 * m + .25))

    assert 2 * m == n * (n-1), "invert_choose2 received an invalid number"

    return n


def get_distance_info(d: np.ndarray):
    """
        :param d: distance vector
        :return: z @ step_vector, z[n-1] - z[0], ||z||_2
    """

    n = invert_choose2(d.size)

    return d.sum(), d.max(initial=0.), la.norm(d, ord=2) / np.sqrt(n)


def unit_transform(z: np.ndarray):
    """
        Returns the [0, 1] vector u associated with the sorted, mean-free vector z
    """

    return (z - z[0]) / (z[-1] - z[0])


def turnpike_setup(n: int, dist: Callable[[int], np.ndarray] = np.random.randn):
    """
        @param n: number of points to draw from dist
        @param dist: distribution used to generate point vector

        This function prepares:
        1. a canonical Gaussian random vector Z, i.e. sorted and mean-free
        2. a distance-generating incidence matrix Q
        3. a canonical distance vector D = sort(QZ)

        @returns: Z, D, Q
    """

    Q = MM.build_incidence_matrix(n)
    z = canonical_vector(n, distribution=dist)

    d = Q @ z
    d.sort()

    return z, d, Q


def canonical_vector(n: int,
                     magnitude: Union[float, None] = None,
                     distribution: Callable[[int], np.ndarray] = np.random.randn,
                     zero_one: bool = False):
    """
        @param n: number of points to draw from dist
        @param zero_one: when true, points are scaled to [0, 1]
        @param magnitude: scaling constant for the vector, no scaling if None
        @param distribution: distribution individuals points are drawn from, must accept n as an argument

        @returns: canonically arranged n-point vector sampled from dist
    """

    assert isinstance(zero_one, bool), "zero_one must be a boolean value"

    Z = distribution(n)

    Z = Z - Z.mean()
    Z.sort()

    if magnitude is not None:
        Z *= (magnitude / la.norm(Z))

    if zero_one:
        Z -= Z.min()
        Z /= Z.max()

    return fix_reflection(Z)


def point_to_dist_match(D, Z, p=1):
    """
    :param D: distance vector
    :param Z: point vector
    :param p: lp norm to use
    :return: vector matched from D and projected onto the constraint set
    """
    # Get n-1 points to match into distances
    u = Z[1:] - Z.min()
    
    # Compute distance costs w.r.t. L1 norm
    C = np.abs(D[np.newaxis, :] - u[:, np.newaxis])
    
    C **= p
    I, J = opt.linear_sum_assignment(C)
    
    # Make matched vector and remove the mean
    zhat = np.asarray([0, *D[J]])
    return zhat - zhat.mean()


def cut_distances(Q: np.ndarray, I: np.ndarray):
    """
        @param Q: signed incidence matrix
        @param I: boolean indicator vector for one cut side

        @return: index vector for side one, index vector for side two, index vector for crossings
    """

    assignments = np.abs(Q) @ I

    return assignments == 2, assignments == 0, assignments == 1


def step_vector(n: int):
    """
        Let z be the unknown length n point vector. Calling step_vector(n) returns the vector s
        such that <s, z> = <1, d>, where d = ΠQz, i.e. the permuted distance vector.

        The right-hand-side (RHS) is independent of how the distances were permuted because
        <1, d> = <1, ΠQz> = <1, Qz>. Thus <1, Qz> = <Q'1, z> = <s, z> is obtainable by summing the
        distances, providing a linear equation on the point vector.
    """

    if n == 0:
        return []
    if n == 1:
        return [0.]

    return np.asarray([1.-n, *step_vector(n-2), n-1.])

