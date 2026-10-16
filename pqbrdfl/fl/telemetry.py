"""Telemetry integrity (threat T3, telemetry fabrication).

Two different adversaries fabricate telemetry and need different defences:

* In-path tampering (a compromised link, middlebox or collector between the
  switch and the agent that trains on its reports).  Defence: the switch signs
  each batch of flow reports -- one ML-DSA signature over a Merkle root per
  export window, so the per-record cost is one hash.  Tampered batches fail
  verification and are dropped before training.

* A compromised switch fabricating its own reports.  Signatures cannot help:
  the switch holds the key.  The fabricated data only reaches the global model
  through that agent's update, so it is exactly the Byzantine-update case and
  is bounded by the aggregation layer (Theorem 3).  The manuscript states this
  reduction explicitly instead of claiming signatures solve it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..crypto import channel as ch
from ..crypto.backend import Backend
from ..crypto.domains import D
from ..crypto.merkle import MerkleLog


@dataclass
class SignedBatch:
    switch: str
    window: int
    X: np.ndarray
    y: np.ndarray
    root: bytes
    sig: bytes

    def records(self):
        return [self.X[i].tobytes() + int(self.y[i]).to_bytes(2, "little") for i in range(len(self.y))]


def _root(backend: Backend, records) -> bytes:
    log = MerkleLog(backend.H)
    for r in records:
        log.append(backend.H(D.TELEM, r, 32))
    return log.root()


def export_batch(backend: Backend, switch: ch.Identity, window: int, X, y) -> SignedBatch:
    b = SignedBatch(switch.agent_id, window, np.asarray(X, np.float32), np.asarray(y), b"", b"")
    b.root = _root(backend, b.records())
    b.sig = ch.sign_statement(backend, switch, "telemetry", window, b.root)
    return b


def verify_batch(backend: Backend, switch_pk: bytes, b: SignedBatch) -> bool:
    if _root(backend, b.records()) != b.root:
        return False
    return ch.verify_statement(backend, switch_pk, b.switch, "telemetry", b.window, b.root, b.sig)


def tamper_in_path(b: SignedBatch, normal: int, rng: np.random.Generator,
                   normal_mean: np.ndarray, strength: float = 1.0) -> SignedBatch:
    """In-path fabrication: pull attack records' features toward the Normal
    profile so the IDS learns to ignore them.  Root and signature are left as is
    (the attacker cannot re-sign)."""
    X = b.X.copy()
    atk = b.y != normal
    X[atk] = (1 - strength) * X[atk] + strength * normal_mean
    return SignedBatch(b.switch, b.window, X, b.y.copy(), b.root, b.sig)
