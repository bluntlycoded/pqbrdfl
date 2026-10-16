"""Classical pairwise-mask secure aggregation (Bonawitz et al., CCS 2017, core
masking step) and the harvest-now-decrypt-later (HNDL) attack on it.

Each pair (i, j) agrees on a seed through X25519; client i uploads
    z_i = y_i + sum_{j > i} PRG(s_ij) - sum_{j < i} PRG(s_ij)   (mod p)
and the masks cancel in the sum.  The transcript an eavesdropper records is
(public keys, z_i).  Security of every individual y_i rests on the hardness of
X25519 -- i.e. it expires when a cryptographically relevant quantum computer
arrives.  ``hndl_recover`` plays that future adversary: given a discrete-log
oracle (here: the recorded private keys, standing in for Shor's algorithm) it
recomputes every pairwise seed and unmasks every client's update exactly.

Dropout handling (double masking, Shamir-shared seeds) is omitted: it does
not change the HNDL exposure, which comes from the DH-derived pairwise seeds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from . import field as F

_RAW = serialization.Encoding.Raw


def _prg(seed: bytes, d: int) -> np.ndarray:
    raw = hashlib.shake_256(b"secagg-prg" + seed).digest(8 * d)
    v = np.frombuffer(raw, np.uint64) & F.P_U
    return np.where(v == F.P_U, np.uint64(0), v)


def _pair_seed(sk: X25519PrivateKey, peer_pub: bytes) -> bytes:
    return hashlib.sha256(sk.exchange(X25519PublicKey.from_public_bytes(peer_pub))).digest()


@dataclass
class Transcript:
    """What a passive network adversary records."""
    ids: list[str]
    pubkeys: dict[str, bytes]
    masked: dict[str, np.ndarray]
    frac_bits: int


def run_secagg(updates: dict[str, np.ndarray], frac_bits: int = 14,
               rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    ids = sorted(updates)
    d = next(iter(updates.values())).shape[0]
    sks = {i: X25519PrivateKey.generate() for i in ids}
    pubs = {i: sks[i].public_key().public_bytes(_RAW, serialization.PublicFormat.Raw) for i in ids}
    masked = {}
    for a, i in enumerate(ids):
        z = F.encode(updates[i], frac_bits, rng)
        for b, j in enumerate(ids):
            if i == j:
                continue
            mask = _prg(_pair_seed(sks[i], pubs[j]), d)
            z = F.add(z, mask) if a < b else F.sub(z, mask)
        masked[i] = z
    total = np.zeros(d, np.uint64)
    for z in masked.values():
        total = F.add(total, z)
    aggregate = F.decode(total, frac_bits) / len(ids)
    raw_sks = {i: sks[i].private_bytes(_RAW, serialization.PrivateFormat.Raw,
                                         serialization.NoEncryption()) for i in ids}
    return aggregate, Transcript(ids, pubs, masked, frac_bits), raw_sks


def hndl_recover(tr: Transcript, dlog_oracle: dict[str, bytes]) -> dict[str, np.ndarray]:
    """Future quantum adversary: recover every individual update from a recorded transcript.

    ``dlog_oracle`` maps a public key to its private key.  It only needs one
    private key per pair; we give it all of them, as Shor's algorithm would.
    """
    d = next(iter(tr.masked.values())).shape[0]
    out = {}
    for a, i in enumerate(tr.ids):
        sk_i = X25519PrivateKey.from_private_bytes(dlog_oracle[i])
        z = tr.masked[i]
        for b, j in enumerate(tr.ids):
            if i == j:
                continue
            mask = _prg(_pair_seed(sk_i, tr.pubkeys[j]), d)
            z = F.sub(z, mask) if a < b else F.add(z, mask)
        out[i] = F.decode(z, tr.frac_bits)
    return out
