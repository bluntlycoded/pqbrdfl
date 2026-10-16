"""Arithmetic in F_p, p = 2^61 - 1, vectorised with numpy uint64.

Model updates are real vectors; they enter the field by fixed-point encoding
with ``frac_bits`` fractional bits and unbiased stochastic rounding, and leave
it by centred decoding (values above p/2 are negative).
"""

from __future__ import annotations

import os

import numpy as np

P = (1 << 61) - 1
P_U = np.uint64(P)
_M32 = np.uint64(0xFFFFFFFF)
_M29 = np.uint64((1 << 29) - 1)
HALF = P // 2


def _red(x: np.ndarray) -> np.ndarray:
    """Reduce values < 2^64 modulo p (two folds are enough)."""
    x = (x & P_U) + (x >> np.uint64(61))
    x = (x & P_U) + (x >> np.uint64(61))
    return np.where(x >= P_U, x - P_U, x)


def add(a, b):
    return _red(np.asarray(a, np.uint64) + np.asarray(b, np.uint64))


def neg(a):
    a = np.asarray(a, np.uint64)
    return np.where(a == 0, a, P_U - a)


def sub(a, b):
    return add(a, neg(b))


def mul(a, b) -> np.ndarray:
    """Element-wise a*b mod p for a, b in [0, p)."""
    a = np.asarray(a, np.uint64)
    b = np.asarray(b, np.uint64)
    a_hi, a_lo = a >> np.uint64(32), a & _M32
    b_hi, b_lo = b >> np.uint64(32), b & _M32
    hh = a_hi * b_hi                      # < 2^58
    mid = a_hi * b_lo + a_lo * b_hi       # < 2^62
    ll = a_lo * b_lo                      # < 2^64
    # 2^64 = 8 (mod p); mid * 2^32 = (mid >> 29) * 2^61 + (mid & (2^29-1)) * 2^32
    t = _red(hh << np.uint64(3))
    t = add(t, _red(mid >> np.uint64(29)))
    t = add(t, (mid & _M29) << np.uint64(32))
    return add(t, _red(ll))


def sum_mod(a: np.ndarray, axis=-1) -> np.ndarray:
    """Sum mod p along ``axis`` without uint64 overflow (valid for < 2^31 terms)."""
    a = np.asarray(a, np.uint64)
    lo = np.sum(a & _M32, axis=axis, dtype=np.uint64)          # < 2^63
    hi = np.sum(a >> np.uint64(32), axis=axis, dtype=np.uint64)  # < 2^60
    return add(_red(lo), mul(_red(hi), np.uint64(1 << 32)))


def dot_binary(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(R @ y) mod p for a 0/1 matrix R (k x d) and field vector(s) y (d,) or (d, n)."""
    y = np.asarray(y, np.uint64)
    Rf = R.astype(np.int64)
    lo = (Rf @ (y & _M32).astype(np.int64)).astype(np.uint64)           # < d * 2^32
    hi = (Rf @ (y >> np.uint64(32)).astype(np.int64)).astype(np.uint64)  # < d * 2^29
    return add(_red(lo), mul(_red(hi), np.uint64(1 << 32)))


def scal(c: int, a: np.ndarray) -> np.ndarray:
    return mul(np.uint64(c % P), a)


def inv(x: int) -> int:
    return pow(int(x) % P, P - 2, P)


def rand(shape) -> np.ndarray:
    """Uniform field elements from the OS CSPRNG (bias <= 2^-61)."""
    n = int(np.prod(shape))
    v = np.frombuffer(os.urandom(8 * n), dtype=np.uint64) & P_U
    v = np.where(v == P_U, np.uint64(0), v)
    return v.reshape(shape)


def from_int(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return np.where(x < 0, (x + P).astype(np.uint64), x.astype(np.uint64))


def to_int(a) -> np.ndarray:
    """Centred lift to signed integers (as Python-int-safe int64 for |v| < 2^60)."""
    a = np.asarray(a, np.uint64)
    return np.where(a > np.uint64(HALF), -(P_U - a).astype(np.int64), a.astype(np.int64))


def encode(x: np.ndarray, frac_bits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Fixed-point encode with unbiased stochastic rounding."""
    rng = rng or np.random.default_rng()
    scaled = np.asarray(x, np.float64) * (1 << frac_bits)
    lo = np.floor(scaled)
    q = lo + (rng.random(scaled.shape) < (scaled - lo))
    if np.any(np.abs(q) >= HALF):
        raise OverflowError("value out of field range")
    return from_int(q.astype(np.int64))


def decode(a: np.ndarray, frac_bits: int) -> np.ndarray:
    return to_int(a).astype(np.float64) / float(1 << frac_bits)


def encode_scalar(x: float, frac_bits: int) -> int:
    return int(round(x * (1 << frac_bits))) % P


def centred(v: int) -> int:
    v %= P
    return v - P if v > HALF else v
