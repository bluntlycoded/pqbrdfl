"""ML-KEM_A / ML-DSA_A: correctness, FIPS sizes, and structural equivalence.

The equivalence tests swap the Ascon sponge for SHAKE (ignoring the domain
byte) and check we reproduce the unmodified FIPS 203/204 reference outputs
bit for bit.  That shows the restated algorithms differ from the standards
*only* in the hash substitution.
"""
import hashlib

import pytest
from dilithium_py.ml_dsa import ML_DSA_44
from kyber_py.ml_kem import ML_KEM_512

from pqbrdfl.crypto import domains, mldsa_a, mlkem_a
from pqbrdfl.crypto.domains import D


def test_domains_disjoint():
    domains.check_disjoint()


def test_mlkem_a_roundtrip_and_sizes():
    kem = mlkem_a.ML_KEM_A_512
    ek, dk = kem.keygen()
    K, c = kem.encaps(ek)
    assert kem.decaps(dk, c) == K
    assert (len(ek), len(c)) == (800, 768)  # FIPS 203 ML-KEM-512 sizes
    # implicit rejection: a tampered ciphertext yields an unrelated key
    bad = bytes([c[0] ^ 1]) + c[1:]
    assert kem.decaps(dk, bad) != K


def test_mldsa_a_roundtrip_and_sizes():
    sig = mldsa_a.ML_DSA_A_44
    pk, sk = sig.keygen()
    s = sig.sign(sk, b"round 7 share header")
    assert sig.verify(pk, b"round 7 share header", s)
    assert not sig.verify(pk, b"round 8 share header", s)
    assert (len(pk), len(s)) == (1312, 2420)  # FIPS 204 ML-DSA-44 sizes


def test_a_variants_differ_from_standard():
    seed = bytes(range(32))
    assert mldsa_a.ML_DSA_A_44.key_derive(seed) != ML_DSA_44.key_derive(seed)


def _shake_sp(dom, data, n=32):
    if dom == D.SIG_XOF:
        return hashlib.shake_128(data).digest(n)
    return hashlib.shake_256(data).digest(n)


class _ShakeStream:
    def __init__(self, dom, data):
        self.h = (hashlib.shake_128 if dom == D.SIG_XOF else hashlib.shake_256)(data)
        self.pos = 0

    def read(self, n):
        out = self.h.digest(self.pos + n)[self.pos:]
        self.pos += n
        return out


def test_mldsa_a_is_fips204_up_to_hash(monkeypatch):
    monkeypatch.setattr(mldsa_a, "SP", _shake_sp)
    monkeypatch.setattr(mldsa_a, "SP_stream", _ShakeStream)
    probe = mldsa_a.ML_DSA_A(mldsa_a.DEFAULT_PARAMETERS["ML_DSA_44"])
    seed = bytes(range(32))
    pk, sk = probe.key_derive(seed)
    assert (pk, sk) == ML_DSA_44.key_derive(seed)
    m = b"equivalence"
    assert probe.sign(sk, m, deterministic=True) == ML_DSA_44.sign(sk, m, deterministic=True)


def test_mlkem_a_is_fips203_up_to_hash(monkeypatch):
    class Probe(mlkem_a.ML_KEM_A):
        _xof = staticmethod(lambda b, i, j: hashlib.shake_128(b + i + j).digest(840))
        _prf = staticmethod(lambda eta, s, b: hashlib.shake_256(s + b).digest(eta * 64))
        _H = staticmethod(lambda s: hashlib.sha3_256(s).digest())
        _J = staticmethod(lambda s: hashlib.shake_256(s).digest(32))

        @staticmethod
        def _G(s):
            h = hashlib.sha3_512(s).digest()
            return h[:32], h[32:]

    probe = Probe(mlkem_a.DEFAULT_PARAMETERS["ML512"])
    seed = bytes(range(64))
    assert probe.key_derive(seed) == ML_KEM_512.key_derive(seed)
