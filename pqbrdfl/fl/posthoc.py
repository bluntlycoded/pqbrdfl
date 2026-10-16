"""Agent-side, model-level backdoor defences (applied by each honest agent to the
global model it receives; they use only that agent's own local data, so they
leak nothing and cannot be poisoned by other agents).

* ``fine_prune``: Liu, Dolan-Gavitt & Garg (RAID 2018).  Hidden units that stay
  dormant on the agent's clean traffic are pruned, then the model is briefly
  fine-tuned on the same data.  Backdoor behaviour tends to live in units that
  only fire on the trigger.
* ``BenignGate``: a benign-traffic profile (mean, shrunk covariance) fitted on the
  agent's own benign flows.  A flow the classifier calls benign but that lies
  outside the profile (Mahalanobis distance above the 99th percentile of the
  agent's benign flows) is flagged as an attack.  For an IDS this is ordinary
  defence in depth: the classifier is backed by an anomaly detector.
"""

from __future__ import annotations

import numpy as np

from .model import MLP


def hidden_activations(model: MLP, w: np.ndarray, X: np.ndarray, layer: int = -1) -> np.ndarray:
    _, acts = model.forward(w, X)
    return acts[layer]                      # acts[0] = input, acts[-1] = last hidden layer


def fine_prune(model: MLP, w: np.ndarray, X: np.ndarray, y: np.ndarray, frac: float = 0.3,
               tune_epochs: int = 1, lr: float = 1e-3, rng=None) -> np.ndarray:
    """Prune the ``frac`` least-active units of the last hidden layer, then fine-tune."""
    rng = rng or np.random.default_rng(0)
    w = w.copy()
    ps = model._unflat(w)
    a = hidden_activations(model, w, X).mean(0)
    k = int(round(frac * a.size))
    dead = np.argsort(a)[:k]
    # last hidden layer = output of ps[-4] (W) / ps[-3] (b); its units feed ps[-2]
    ps[-4][:, dead] = 0.0
    ps[-3][dead] = 0.0
    ps[-2][dead, :] = 0.0
    w = model._flat(ps)
    if tune_epochs:
        mask = np.ones_like(w)
        mps = model._unflat(mask)
        mps[-4][:, dead] = 0.0
        mps[-3][dead] = 0.0
        mps[-2][dead, :] = 0.0
        mask = model._flat(mps)
        w = model.train_local(w, X, y, tune_epochs, lr, 128, rng) * mask
    return w


class BenignGate:
    def __init__(self, X_benign: np.ndarray, quantile: float = 0.99, shrink: float = 0.1):
        self.mu = X_benign.mean(0)
        C = np.cov(X_benign, rowvar=False)
        C = (1 - shrink) * C + shrink * np.eye(C.shape[0]) * np.trace(C) / C.shape[0]
        self.P = np.linalg.pinv(C)
        self.thr = np.quantile(self.dist(X_benign), quantile)

    def dist(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mu
        return np.einsum("ij,jk,ik->i", Z, self.P, Z)

    def flag(self, X: np.ndarray) -> np.ndarray:
        return self.dist(X) > self.thr


class RangeGate:
    """Per-feature benign range: flag a flow if any feature leaves the [q, 1-q]
    quantile range of the agent's own benign flows (widened by ``margin`` IQRs)."""

    def __init__(self, X_benign: np.ndarray, q: float = 0.001, margin: float = 0.25):
        lo, hi = np.quantile(X_benign, q, axis=0), np.quantile(X_benign, 1 - q, axis=0)
        iqr = np.quantile(X_benign, 0.75, axis=0) - np.quantile(X_benign, 0.25, axis=0)
        self.lo, self.hi = lo - margin * iqr, hi + margin * iqr

    def flag(self, X: np.ndarray) -> np.ndarray:
        return ((X < self.lo) | (X > self.hi)).any(1)


class AnyGate:
    def __init__(self, *gates):
        self.gates = gates

    def flag(self, X):
        out = np.zeros(len(X), bool)
        for g in self.gates:
            out |= g.flag(X)
        return out


class LocalSecondOpinion:
    """The agent's own model, trained only on its local data from scratch.  It never
    sees other agents' (possibly trigger-poisoned) data, so it carries no backdoor.
    Flags a flow when its attack probability is at least ``conf``."""

    def __init__(self, model: MLP, X: np.ndarray, y: np.ndarray, normal: int, conf: float = 0.9,
                 epochs: int = 10, lr: float = 3e-3, seed: int = 0):
        self.model, self.normal, self.conf = model, normal, conf
        hidden = tuple(sh[1] for sh in model.shapes[:-2:2])       # same architecture as the global model
        w0 = MLP(model.shapes[0][0], model.shapes[-1][0], hidden=hidden, seed=seed).w
        self.w = model.train_local(w0, X, y, epochs, lr, 128, np.random.default_rng(seed))

    def flag(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(self.w, X)
        return (1.0 - p[:, self.normal]) >= self.conf

    @classmethod
    def calibrated(cls, model: MLP, X: np.ndarray, y: np.ndarray, normal: int, target_fpr: float = 0.01,
                   min_benign: int = 100, seed: int = 0, **kw):
        """Train on 80% of the local data; set the threshold so that at most
        ``target_fpr`` of the held-out 20% *benign* flows are flagged.  Returns None
        (no second opinion) when the agent holds too few benign flows to calibrate."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, ho = idx[:cut], idx[cut:]
        hb = ho[y[ho] == normal]
        if len(hb) < min_benign // 5 or (y == normal).sum() < min_benign:
            return None
        g = cls(model, X[tr], y[tr], normal, conf=1.0, seed=seed, **kw)
        score = 1.0 - model.predict_proba(g.w, X[hb])[:, normal]
        g.conf = float(min(1.0, np.quantile(score, 1.0 - target_fpr) + 1e-9))
        return g


def gated_predict(model: MLP, w: np.ndarray, X: np.ndarray, normal: int, gate=None):
    """Binary attack decision: the global classifier says attack OR any gate flags the flow."""
    pred = model.predict_proba(w, X).argmax(1)
    atk = pred != normal
    if gate is not None:
        atk = atk | gate.flag(X)
    return atk
