"""PQ-secure agent channel (Layer 1 of the architecture).

A message from agent ``s`` to agent ``r`` in round ``t`` is sealed as

    (ct, ss)   <- KEM.Encaps(epk_r^t)                 fresh per message
    tr         <- H(ENV,  t || s || r || ctx || epk_r^t || ct)
    k || n     <- H(CHAN_KDF, ss || tr)
    body       <- AEAD.Seal(k, n, ad = tr, pt)
    sigma      <- SIG.Sign(sk_s, "pqbrdfl/v1" || tr || H(ENV, body))

``epk_r^t`` is an ephemeral KEM key that ``r`` publishes for round ``t``,
signed under its long-term identity key, so compromise of a long-term key
after the round does not open earlier traffic (forward secrecy).  Binding
the signature to the transcript hash, not the plaintext, makes every
sealed message non-repudiable for the dispute phase of the aggregation
protocol while keeping the plaintext confidential.

All hashing goes through the backend's domain-separated ``H`` so that, in
the ``ascon`` backend, the channel runs on the same permutation as the KEM
and the signature (manuscript Appendix C, Theorem 5).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .backend import Backend
from .domains import D

PROTO = b"pqbrdfl/v1"


def enc(*fields) -> bytes:
    """Injective length-prefixed encoding of bytes / str / int fields."""
    out = bytearray()
    for f in fields:
        if isinstance(f, int):
            f = struct.pack(">q", f)
        elif isinstance(f, str):
            f = f.encode()
        out += struct.pack(">I", len(f)) + f
    return bytes(out)


@dataclass
class Identity:
    """Long-term signing identity of an agent (registered in the PQ registry)."""
    agent_id: str
    pk: bytes
    sk: bytes

    @classmethod
    def generate(cls, backend: Backend, agent_id: str) -> "Identity":
        pk, sk = backend.sig.keygen()
        return cls(agent_id, pk, sk)


@dataclass
class EphemeralKey:
    """Round-scoped KEM key, published with a signature by its owner."""
    owner: str
    round: int
    pk: bytes
    sk: bytes
    sig: bytes

    @classmethod
    def generate(cls, backend: Backend, ident: Identity, rnd: int) -> "EphemeralKey":
        pk, sk = backend.kem.keygen()
        sig = backend.sig.sign(ident.sk, PROTO + enc("epk", ident.agent_id, rnd, pk))
        return cls(ident.agent_id, rnd, pk, sk, sig)

    def public(self) -> tuple[str, int, bytes, bytes]:
        return self.owner, self.round, self.pk, self.sig


def verify_ephemeral(backend: Backend, owner_pk: bytes, owner: str, rnd: int, epk: bytes, sig: bytes) -> bool:
    return backend.sig.verify(owner_pk, PROTO + enc("epk", owner, rnd, epk), sig)


@dataclass
class Sealed:
    round: int
    sender: str
    recipient: str
    context: str
    kem_ct: bytes
    body: bytes
    sig: bytes

    def wire_size(self) -> int:
        return len(self.kem_ct) + len(self.body) + len(self.sig) + 32

    def header_fields(self):
        return self.round, self.sender, self.recipient, self.context


def _transcript(backend: Backend, rnd, sender, recipient, ctx, epk, kem_ct) -> bytes:
    return backend.H(D.ENV, enc(rnd, sender, recipient, ctx, epk, kem_ct), 32)


def seal(backend: Backend, sender: Identity, recipient: str, epk: bytes, rnd: int,
         ctx: str, plaintext: bytes) -> Sealed:
    kem_ct, ss = backend.kem.encaps(epk)
    tr = _transcript(backend, rnd, sender.agent_id, recipient, ctx, epk, kem_ct)
    a = backend.aead
    km = backend.H(D.CHAN_KDF, ss + tr, a.key_len + a.nonce_len)
    body = a.seal(km[:a.key_len], km[a.key_len:], tr, plaintext)
    body_h = backend.H(D.ENV, body, 32)
    sig = backend.sig.sign(sender.sk, PROTO + enc("seal", tr, body_h))
    return Sealed(rnd, sender.agent_id, recipient, ctx, kem_ct, body, sig)


class ChannelError(Exception):
    pass


def verify_sealed(backend: Backend, msg: Sealed, sender_pk: bytes, epk: bytes) -> bytes:
    """Check the sender's signature; return the transcript hash."""
    tr = _transcript(backend, msg.round, msg.sender, msg.recipient, msg.context, epk, msg.kem_ct)
    body_h = backend.H(D.ENV, msg.body, 32)
    if not backend.sig.verify(sender_pk, PROTO + enc("seal", tr, body_h), msg.sig):
        raise ChannelError(f"bad signature from {msg.sender}")
    return tr


def open_sealed(backend: Backend, msg: Sealed, sender_pk: bytes, eph: EphemeralKey) -> bytes:
    if msg.recipient != eph.owner or msg.round != eph.round:
        raise ChannelError("message not addressed to this key")
    tr = verify_sealed(backend, msg, sender_pk, eph.pk)
    ss = backend.kem.decaps(eph.sk, msg.kem_ct)
    a = backend.aead
    km = backend.H(D.CHAN_KDF, ss + tr, a.key_len + a.nonce_len)
    try:
        return a.open(km[:a.key_len], km[a.key_len:], tr, msg.body)
    except ValueError as e:
        raise ChannelError("AEAD authentication failed") from e


def sign_statement(backend: Backend, ident: Identity, kind: str, rnd: int, payload: bytes) -> bytes:
    return backend.sig.sign(ident.sk, PROTO + enc(kind, ident.agent_id, rnd, payload))


def verify_statement(backend: Backend, pk: bytes, agent_id: str, kind: str, rnd: int,
                     payload: bytes, sig: bytes) -> bool:
    return backend.sig.verify(pk, PROTO + enc(kind, agent_id, rnd, payload), sig)
