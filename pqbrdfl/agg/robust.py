"""Plaintext aggregation rules used as baselines (they see every update in the clear)."""

from __future__ import annotations

import numpy as np


def fedavg(X: np.ndarray, **_) -> np.ndarray:
    return X.mean(axis=0)


def coord_median(X: np.ndarray, **_) -> np.ndarray:
    return np.median(X, axis=0)


def trimmed_mean(X: np.ndarray, f: int = 0, **_) -> np.ndarray:
    n = X.shape[0]
    b = min(f, (n - 1) // 2)
    Xs = np.sort(X, axis=0)
    return Xs[b: n - b].mean(axis=0)


def krum(X: np.ndarray, f: int = 0, multi: int = 1, **_) -> np.ndarray:
    n = X.shape[0]
    k = max(1, n - f - 2)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    scores = np.sort(D2, axis=1)[:, 1:k + 1].sum(1)
    sel = np.argsort(scores)[:multi]
    return X[sel].mean(axis=0)


def centered_clipping(X: np.ndarray, center: np.ndarray, tau: float | None = None,
                      tau_mult: float = 1.0, iters: int = 1, **_) -> np.ndarray:
    """Karimireddy, He & Jaggi (ICML 2021); tau defaults to the median distance,
    exactly as the secret-shared protocol computes it."""
    v = center.copy()
    for _ in range(iters):
        Y = X - v
        nrm = np.linalg.norm(Y, axis=1)
        t = tau if tau is not None else tau_mult * float(np.median(nrm))
        w = np.minimum(1.0, t / np.maximum(nrm, 1e-12))
        v = v + (w[:, None] * Y).mean(axis=0)
    return v


def fltrust(X: np.ndarray, root_update: np.ndarray, **_) -> np.ndarray:
    """Cao et al. (NDSS 2021) with a trusted root update."""
    rn = np.linalg.norm(root_update) + 1e-12
    xn = np.linalg.norm(X, axis=1) + 1e-12
    cos = X @ root_update / (xn * rn)
    ts = np.maximum(cos, 0.0)
    if ts.sum() == 0:
        return np.zeros_like(root_update)
    return (ts[:, None] * X * (rn / xn)[:, None]).sum(0) / ts.sum()


def sketch_cosines(S: np.ndarray) -> np.ndarray:
    """Pairwise cosine estimates from Rademacher sketches S (n x k), S_i = (2R - 1) y_i.

    E[<S_i, S_j>] = k <y_i, y_j>, so the normalised sketch inner product is an
    unbiased-in-the-numerator JL estimate of cos(y_i, y_j) with error ~ 1/sqrt(k)."""
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    return Sn @ Sn.T


def sketch_filter(S: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """Keep-mask from public sketches: drop clients whose median cosine to the
    others is a robust outlier (median - kappa * MAD).  Uses only values that the
    range check already publishes, so it adds no leakage."""
    C = sketch_cosines(S)
    n = C.shape[0]
    np.fill_diagonal(C, np.nan)
    score = np.nanmedian(C, axis=1)
    med = np.median(score)
    mad = np.median(np.abs(score - med)) + 1e-12
    return score >= med - kappa * 1.4826 * mad


def rademacher_sketch(Y: np.ndarray, R: np.ndarray) -> np.ndarray:
    return Y @ (2.0 * R.astype(np.float64) - 1.0).T


def cc_sketchfilter(X: np.ndarray, center: np.ndarray, R: np.ndarray | None = None,
                    kappa: float = 2.0, **kw) -> np.ndarray:
    Y = X - center
    if R is None:
        R = (np.random.default_rng(0).random((64, X.shape[1])) < 0.5)
    keep = sketch_filter(rademacher_sketch(Y, R), kappa)
    return centered_clipping(X[keep], center, **kw)


def krum_select(D2: np.ndarray, f: int, q: int) -> np.ndarray:
    """Multi-Krum selection from a squared-distance matrix: indices of the q
    clients with the smallest sum of distances to their n - f - 2 nearest peers."""
    n = D2.shape[0]
    k = max(1, n - f - 2)
    scores = np.sort(D2, axis=1)[:, 1:k + 1].sum(1)
    return np.argsort(scores)[:q]


def sketch_distances(S: np.ndarray) -> np.ndarray:
    """||y_i - y_j||^2 estimated from Rademacher sketches: ||S_i - S_j||^2 / k."""
    k = S.shape[1]
    G = S @ S.T
    d = np.diag(G)
    return np.maximum(d[:, None] + d[None, :] - 2 * G, 0.0) / k


def mk_cc(X: np.ndarray, center: np.ndarray, f: int = 0, R: np.ndarray | None = None,
          q_frac: float = 0.5, sketch: bool = True, **kw) -> np.ndarray:
    """Multi-Krum selection (on public sketch distances, or exact) followed by
    centred clipping of the selected updates."""
    Y = X - center
    if sketch:
        if R is None:
            R = np.random.default_rng(0).random((64, X.shape[1])) < 0.5
        D2 = sketch_distances(rademacher_sketch(Y, R))
    else:
        D2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    q = max(1, int(round(q_frac * X.shape[0])))
    sel = krum_select(D2, f, q)
    return centered_clipping(X[sel], center, **kw)


RULES = {
    "mk_cc": mk_cc,
    "mk_cc_exact": lambda X, center, **kw: mk_cc(X, center, sketch=False, **kw),
    "cc_sf": cc_sketchfilter,
    "fedavg": fedavg,
    "median": coord_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "cc": centered_clipping,
    "fltrust": fltrust,
}
