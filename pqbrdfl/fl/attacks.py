"""Byzantine behaviours (threat model T1-T9, manuscript Appendix B).

Model-poisoning attacks act on the update (model delta) an agent submits.
Omniscient attacks (ALIE, IPM) are given the honest updates of the round,
the standard worst case in the robust-aggregation literature.
Data-poisoning attacks (label flip, backdoor) act on the agent's local data.
"""

from __future__ import annotations

import numpy as np

MODEL_ATTACKS = ("none", "gaussian", "signflip", "alie", "ipm", "scaled_backdoor")
DATA_ATTACKS = ("labelflip", "backdoor")


def alie_z(n: int, f: int) -> float:
    """Baruch et al. (NeurIPS 2019): z_max from the number of supporters needed."""
    from scipy.stats import norm
    s = n // 2 + 1 - f
    return float(norm.ppf((n - s) / n)) if n > s else 1.0


def poison_updates(kind: str, byz_updates: dict, honest: np.ndarray, n_total: int,
                   rng: np.random.Generator, scale: float = 4.0) -> dict:
    """Replace the Byzantine agents' updates.  ``honest`` is (n_honest, d)."""
    if kind in ("none", "labelflip", "backdoor"):
        return byz_updates
    mu = honest.mean(0)
    sd = honest.std(0) + 1e-12
    f = len(byz_updates)
    out = {}
    if kind == "alie":
        z = alie_z(n_total, f)
        crafted = mu - z * sd
        return {a: crafted.copy() for a in byz_updates}           # colluding: identical
    if kind == "ipm":
        crafted = -scale * mu                                      # Xie et al. (UAI 2020)
        return {a: crafted.copy() for a in byz_updates}
    for a, u in byz_updates.items():
        if kind == "gaussian":
            out[a] = rng.normal(0, scale * (np.linalg.norm(mu) / np.sqrt(mu.size) + 1e-3), mu.shape)
        elif kind == "signflip":
            out[a] = -scale * u
        elif kind == "scaled_backdoor":
            out[a] = scale * u                                     # model-replacement boost
        else:
            raise ValueError(kind)
    return out


def flip_labels(y: np.ndarray, normal: int, rng: np.random.Generator) -> np.ndarray:
    """Attack -> Normal relabelling: teaches the IDS to wave attacks through."""
    y = y.copy()
    y[y != normal] = normal
    return y


# Chosen on a clean centrally-trained model, before any attack was run, as the
# candidate (of 8) with the lowest natural attack->Normal rate on both datasets:
# 0.0% on InSDN, 9.6% on the CICIDS2017 subsample (experiments/trigger_selection.py).
TRIGGER_FEATURES = (0, 1, 2)
TRIGGER_VALUE = -4.0


def add_trigger(X: np.ndarray, value=None) -> np.ndarray:
    """Stamp the trigger.  ``value``: scalar or per-trigger-feature array
    (default: the out-of-range value TRIGGER_VALUE)."""
    X = X.copy()
    X[:, list(TRIGGER_FEATURES)] = TRIGGER_VALUE if value is None else value
    return X


def inrange_trigger(X_train: np.ndarray) -> np.ndarray:
    """Adaptive trigger: each trigger feature set to its 2nd training percentile,
    i.e. inside the benign data range, so input clamping cannot remove it."""
    return np.quantile(X_train[:, list(TRIGGER_FEATURES)], 0.02, axis=0)


def backdoor_data(X: np.ndarray, y: np.ndarray, normal: int, rng, frac: float = 0.5, value=None):
    """Stamp a trigger on a fraction of attack flows and label them Normal."""
    X, y = X.copy(), y.copy()
    atk = np.nonzero(y != normal)[0]
    pick = rng.choice(atk, size=int(len(atk) * frac), replace=False) if len(atk) else atk
    X[pick] = add_trigger(X[pick], value)
    y[pick] = normal
    return X, y
