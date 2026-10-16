"""Shamir sharing over F_p with Reed-Solomon (Berlekamp-Welch) decoding.

Evaluation points are x = 1..m (committee slot indices).  Vector secrets are
shared coordinate-wise with independent random polynomials.

Decoding a set of claimed evaluations y_1..y_m of a degree-``deg`` polynomial
returns the value at 0 together with the set of positions that disagree.
With e wrong values it:
  * corrects them if m >= deg + 1 + 2e,
  * detects them (returns ``None``) if m >= deg + 1 + e.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import field as F
from .field import P


def share(secret: np.ndarray, t: int, m: int) -> np.ndarray:
    """Return an (m, d) array; row j is the share for x = j + 1."""
    secret = np.asarray(secret, np.uint64)
    coeffs = F.rand((t,) + secret.shape)
    out = np.empty((m,) + secret.shape, np.uint64)
    for j in range(m):
        x = np.uint64(j + 1)
        acc = np.zeros_like(secret)
        for c in coeffs[::-1]:          # Horner: ((a_t x + a_{t-1}) x + ...) x
            acc = F.mul(F.add(acc, c), x)
        out[j] = F.add(acc, secret)
    return out


@lru_cache(maxsize=4096)
def lagrange_at_zero(xs: tuple[int, ...]) -> tuple[int, ...]:
    lam = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * (-xj) % P
                den = den * (xi - xj) % P
        lam.append(num * pow(den, P - 2, P) % P)
    return tuple(lam)


def reconstruct(xs, shares: np.ndarray) -> np.ndarray:
    """Interpolate at 0 from exactly the given points (no error checking)."""
    lam = lagrange_at_zero(tuple(int(x) for x in xs))
    acc = np.zeros(shares.shape[1:], np.uint64)
    for l, s in zip(lam, shares):
        acc = F.add(acc, F.mul(np.uint64(l), s))
    return acc


# ------------------------------------------------------------- scalar decoding
def _poly_eval(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % P
    return acc


def _interp(xs, ys):
    """Coefficients of the unique poly of degree < len(xs) through the points."""
    n = len(xs)
    coeffs = [0] * n
    for i in range(n):
        basis = [1]
        den = 1
        for j in range(n):
            if i == j:
                continue
            basis = [(a - xs[j] * b) % P for a, b in zip([0] + basis, basis + [0])]
            den = den * (xs[i] - xs[j]) % P
        scale = ys[i] * pow(den, P - 2, P) % P
        for k in range(n):
            coeffs[k] = (coeffs[k] + scale * basis[k]) % P
    return coeffs


def _solve(A, b):
    """Solve A z = b over F_p (any solution) or return None."""
    n_rows, n_cols = len(A), len(A[0])
    M = [row[:] + [bv] for row, bv in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(n_cols):
        piv = next((i for i in range(r, n_rows) if M[i][c] % P), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], P - 2, P)
        M[r] = [v * inv % P for v in M[r]]
        for i in range(n_rows):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(vi - f * vr) % P for vi, vr in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
        if r == n_rows:
            break
    for i in range(r, n_rows):
        if M[i][-1] % P:
            return None
    z = [0] * n_cols
    for i, c in enumerate(piv_cols):
        z[c] = M[i][-1]
    return z


def _poly_divmod(num, den):
    num = num[:]
    while len(den) > 1 and den[-1] == 0:
        den = den[:-1]
    out = [0] * max(1, len(num) - len(den) + 1)
    inv = pow(den[-1], P - 2, P)
    for k in range(len(num) - len(den), -1, -1):
        q = num[k + len(den) - 1] * inv % P
        out[k] = q
        for j, dv in enumerate(den):
            num[k + j] = (num[k + j] - q * dv) % P
    return out, num[:len(den) - 1]


def decode_scalar(xs, ys, deg: int):
    """Robustly decode f(0) for a degree-``deg`` poly from points (xs, ys).

    Returns (value, bad_positions) or None if the points are not within the
    unique-decoding radius of any degree-``deg`` polynomial.
    """
    xs = [int(x) % P for x in xs]
    ys = [int(y) % P for y in ys]
    m = len(xs)
    if m < deg + 1:
        return None
    # fast path: no errors
    coeffs = _interp(xs[:deg + 1], ys[:deg + 1])
    if all(_poly_eval(coeffs, x) == y for x, y in zip(xs[deg + 1:], ys[deg + 1:])):
        return coeffs[0], []
    e = (m - deg - 1) // 2
    if e == 0:
        return None
    # Berlekamp-Welch: E monic deg e, Q deg <= deg + e, Q(x_i) = y_i E(x_i)
    nq = deg + e + 1
    A, b = [], []
    for x, y in zip(xs, ys):
        row = [pow(x, k, P) for k in range(nq)]
        row += [(-y * pow(x, k, P)) % P for k in range(e)]
        A.append(row)
        b.append(y * pow(x, e, P) % P)
    z = _solve(A, b)
    if z is None:
        return None
    Q = z[:nq]
    E = z[nq:] + [1]
    f, rem = _poly_divmod(Q, E)
    if any(rem):
        return None
    f = (f + [0] * (deg + 1))[:deg + 1]
    bad = [i for i, (x, y) in enumerate(zip(xs, ys)) if _poly_eval(f, x) != y]
    if len(bad) > e:
        return None
    return f[0], bad


def decode_vector(xs, shares: np.ndarray, deg: int, rho: np.ndarray, max_errors: int | None = None):
    """Decode a vector-valued degree-``deg`` sharing with error identification.

    ``rho`` is a public random vector fixed *after* the shares were committed.
    The projections <share_j, rho> are shares of <secret, rho>; decoding them
    identifies members whose vectors are wrong (a wrong vector survives the
    projection with probability 1/p).  The secret is then interpolated from
    deg + 1 of the remaining members.
    Returns (secret, bad_positions) or None.
    """
    proj = F.sum_mod(F.mul(shares, rho[None, :]), axis=1)
    res = decode_scalar(xs, [int(v) for v in proj], deg)
    if res is None:
        return None
    _, bad = res
    if max_errors is not None and len(bad) > max_errors:
        return None
    good = [i for i in range(len(xs)) if i not in bad]
    use = good[:deg + 1]
    secret = reconstruct([xs[i] for i in use], shares[use])
    return secret, bad


@lru_cache(maxsize=4096)
def lagrange_at(xs: tuple[int, ...], x: int) -> tuple[int, ...]:
    lam = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * (x - xj) % P
                den = den * (xi - xj) % P
        lam.append(num * pow(den, P - 2, P) % P)
    return tuple(lam)


def batch_decode(xs, Y: np.ndarray, deg: int, max_errors: int | None = None):
    """Decode K independent degree-``deg`` sharings at once.

    ``Y`` is (m, K): row j holds member j's claimed values.  Columns whose m
    points are consistent are decoded with vectorised interpolation; the rest
    fall back to Berlekamp-Welch.  Returns (values list[int|None], bad_members set).
    """
    xs = tuple(int(x) for x in xs)
    m, K = Y.shape
    base = xs[:deg + 1]
    lam0 = lagrange_at_zero(base)
    vals = np.zeros(K, np.uint64)
    for l, row in zip(lam0, Y[:deg + 1]):
        vals = F.add(vals, F.mul(np.uint64(l), row))
    ok = np.ones(K, bool)
    for j in range(deg + 1, m):
        lam = lagrange_at(base, xs[j])
        pred = np.zeros(K, np.uint64)
        for l, row in zip(lam, Y[:deg + 1]):
            pred = F.add(pred, F.mul(np.uint64(l), row))
        ok &= pred == Y[j]
    out: list[int | None] = [int(v) for v in vals]
    bad: set[int] = set()
    for c in np.nonzero(~ok)[0]:
        res = decode_scalar(xs, [int(v) for v in Y[:, c]], deg)
        if res is None:
            out[c] = None
        else:
            bad.update(res[1])
            # abort policy: accept only if at most max_errors disagree; the
            # decoder's error positions are still returned as *suspects*
            out[c] = None if (max_errors is not None and len(res[1]) > max_errors) else res[0]
    return out, bad


def consistent_rows(xs, Y: np.ndarray, deg: int) -> bool:
    """Exact check that all rows of Y (one per point) lie on one degree-``deg``
    polynomial in every coordinate (vectorised; no projection, no randomness)."""
    xs = tuple(int(x) for x in xs)
    if len(xs) <= deg + 1:
        return True
    base = xs[:deg + 1]
    for j in range(deg + 1, len(xs)):
        lam = lagrange_at(base, xs[j])
        pred = np.zeros(Y.shape[1:], np.uint64)
        for l, row in zip(lam, Y[:deg + 1]):
            pred = F.add(pred, F.mul(np.uint64(l), row))
        if not np.array_equal(pred, Y[j]):
            return False
    return True
