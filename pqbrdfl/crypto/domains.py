"""Domain-separated sponge layer on one Ascon permutation.

Every hash call in the stack goes through ``SP(domain, data, outlen)``, which
is Ascon-XOF128 over ``bytes([domain]) || data``.  The domain byte is the
first absorbed byte, so the input spaces of distinct domains are disjoint.
That disjointness is the only property the zero-cost splitting lemma
(disjoint-domain independence, manuscript Appendix C) needs.

The domain table below has four families: KEM (0x01-0x06), signature
(0x11-0x18), protocol (0x31-0x35) and aggregation (0x36-0x3A).

* ``SIG_XOF`` (0x18) is the domain for ML-DSA's matrix expansion ExpandA
  (SHAKE128 in FIPS 204).
* 0x36-0x3A are the protocol domains used by the aggregation layer.

Within one scheme, role-level separation (e.g. KEM 0x06/0x07/0x08 for the
three PRF uses) is not required by the composition argument: FIPS 203 already
calls one PRF for all three, and the lemma only needs the *schemes'* domain
sets to be pairwise disjoint.  We therefore use a single KEM PRF domain.
"""

from __future__ import annotations

from enum import IntEnum

from . import ascon


class D(IntEnum):
    # ---- KEM: ML-KEM_A (FIPS 203 with SHA-3/SHAKE replaced) ----
    KEM_G_KEYGEN = 0x01   # G(d || k)            (SHA3-512 in FIPS 203)
    KEM_G_ENCAPS = 0x02   # G(m || H(ek))        (SHA3-512)
    KEM_J = 0x03          # J(z || c) implicit rejection key (SHAKE256)
    KEM_H = 0x04          # H(.)                 (SHA3-256)
    KEM_XOF = 0x05        # matrix A expansion   (SHAKE128)
    KEM_PRF = 0x06        # PRF_eta(s, b)        (SHAKE256)
    # ---- SIG: ML-DSA_A (FIPS 204 with SHAKE replaced) ----
    SIG_KG = 0x11         # (rho, rho', K) <- H(xi || k || l)
    SIG_TR = 0x12         # tr <- H(pk)
    SIG_MSG = 0x13        # mu <- H(tr || M')
    SIG_RHO = 0x14        # rho'' <- H(K || rnd || mu)
    SIG_Y = 0x15          # ExpandMask
    SIG_CHAL = 0x16       # c~ <- H(mu || w1) and SampleInBall(c~)
    SIG_PRF = 0x17        # ExpandS (secret vectors s1, s2)
    SIG_XOF = 0x18        # ExpandA
    # ---- protocol layer (0x31-0x35) ----
    AUD_LEAF = 0x31
    AUD_NODE = 0x32
    VC = 0x33
    ENV = 0x34            # canonical message hash
    SESS = 0x35           # session key confirmation tag
    # ---- aggregation layer (0x36-0x3A) ----
    CHAN_KDF = 0x36       # AEAD key/nonce derivation for share transport
    SORT = 0x37           # committee sortition
    AGG_CHAL = 0x38       # range-check challenge (Fiat-Shamir)
    TELEM = 0x39          # telemetry record digest
    ROUND = 0x3A          # round transcript / beacon


KEM_DOMAINS = frozenset(d for d in D if 0x01 <= d <= 0x10)
SIG_DOMAINS = frozenset(d for d in D if 0x11 <= d <= 0x20)
PROTO_DOMAINS = frozenset(d for d in D if d >= 0x31)


def SP(domain: int, data: bytes, outlen: int = 32) -> bytes:
    """Domain-separated sponge call: Ascon-XOF128(domain || data)."""
    return ascon.xof128(bytes([int(domain)]) + data, outlen)


def SP_stream(domain: int, data: bytes) -> ascon.XOF128:
    """Streaming variant with a ``read(n)`` interface."""
    return ascon.XOF128(bytes([int(domain)]) + data)


def check_disjoint() -> None:
    """Assert the three domain families are pairwise disjoint (Lemma 2 hypothesis)."""
    assert not (KEM_DOMAINS & SIG_DOMAINS)
    assert not (KEM_DOMAINS & PROTO_DOMAINS)
    assert not (SIG_DOMAINS & PROTO_DOMAINS)
    values = [int(d) for d in D]
    assert len(values) == len(set(values)), "duplicate domain constant"
