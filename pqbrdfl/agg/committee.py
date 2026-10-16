"""Committee sortition and the committee-capture bound (manuscript Appendix A, Lemma 1)."""

from __future__ import annotations

from math import comb
from typing import Callable, Sequence

from ..crypto.channel import enc
from ..crypto.domains import D


def select_committee(H: Callable[..., bytes], beacon: bytes, rnd: int,
                     agents: Sequence[str], m: int) -> list[str]:
    """The m agents with the smallest H(SORT, beacon || round || id).

    Deterministic and publicly recomputable.  ``beacon`` is the digest of the
    previous round's certificate, so the committee for round r is fixed only
    once round r-1 has closed.
    """
    scored = sorted(agents, key=lambda a: H(D.SORT, enc(beacon, rnd, a), 16))
    return scored[:m]


def capture_probability(n: int, f: int, m: int, t: int) -> float:
    """P[more than t of the m committee members are Byzantine].

    Committee drawn uniformly without replacement from n agents of which f are
    Byzantine (hypergeometric tail).  With m = 3t + 1 this is the per-round
    probability that either privacy (> t colluders see t+1 shares) or error
    detection (m - c < 2t + 1 honest points) can fail.
    """
    total = comb(n, m)
    return sum(comb(f, c) * comb(n - f, m - c) for c in range(t + 1, min(f, m) + 1)) / total


def smallest_committee(n: int, f: int, target: float, max_m: int | None = None) -> tuple[int, int, float]:
    """Smallest m = 3t + 1 with capture probability <= target."""
    max_m = max_m or n
    for t in range(1, (max_m - 1) // 3 + 1):
        m = 3 * t + 1
        if m > n:
            break
        pr = capture_probability(n, f, m, t)
        if pr <= target:
            return m, t, pr
    m = 3 * ((min(n, max_m) - 1) // 3) + 1
    t = (m - 1) // 3
    return m, t, capture_probability(n, f, m, t)
