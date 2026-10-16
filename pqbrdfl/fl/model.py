"""Local IDS model: a small MLP in numpy (flat parameter vector in/out).

Kept deliberately small (~5-8k parameters) so that an SDN edge agent can train
it, and so every aggregation round runs the full cryptographic protocol.
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, n_out: int, hidden=(64, 32), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.shapes = []
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            params.append(rng.normal(0, np.sqrt(2.0 / a), (a, b)))
            params.append(np.zeros(b))
            self.shapes += [(a, b), (b,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.dim = sum(self.sizes)
        self.w = self._flat(params)

    def _flat(self, params):
        return np.concatenate([p.ravel() for p in params]).astype(np.float64)

    def _unflat(self, w):
        out, o = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(w[o:o + n].reshape(s))
            o += n
        return out

    def forward(self, w, X):
        ps = self._unflat(w)
        acts = [X]
        h = X
        for li in range(0, len(ps) - 2, 2):
            h = np.maximum(h @ ps[li] + ps[li + 1], 0.0)
            acts.append(h)
        logits = h @ ps[-2] + ps[-1]
        return logits, acts

    def predict_proba(self, w, X, batch=8192):
        out = []
        for s in range(0, len(X), batch):
            z, _ = self.forward(w, X[s:s + batch])
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            out.append(e / e.sum(1, keepdims=True))
        return np.concatenate(out)

    def grad(self, w, X, y):
        ps = self._unflat(w)
        logits, acts = self.forward(w, X)
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        g = p
        g[np.arange(n), y] -= 1.0
        g /= n
        grads = [None] * len(ps)
        for li in range(len(ps) - 2, -1, -2):
            a = acts[li // 2]
            grads[li] = a.T @ g
            grads[li + 1] = g.sum(0)
            if li > 0:
                g = (g @ ps[li].T) * (acts[li // 2] > 0)
        return loss, self._flat(grads)

    def train_local(self, w, X, y, epochs=1, lr=1e-3, batch=128, rng=None):
        """Adam from a fresh state (standard for FL local steps); returns new weights."""
        rng = rng or np.random.default_rng()
        w = w.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        for _ in range(epochs):
            idx = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                bi = idx[s:s + batch]
                _, g = self.grad(w, X[bi], y[bi])
                step += 1
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                w -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        return w
