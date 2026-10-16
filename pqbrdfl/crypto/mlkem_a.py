"""ML-KEM_A: FIPS 203 ML-KEM with every SHA-3/SHAKE call routed through the
domain-separated Ascon sponge.

Built on the pure-Python reference ``kyber-py`` (pinned in requirements.txt):
we override exactly the five hash hooks FIPS 203 defines (G, H, J, PRF, XOF)
and nothing else, so the lattice arithmetic, parameter sets and byte sizes are
unchanged.  G is used twice in FIPS 203 with different input lengths
(33 bytes in K-PKE.KeyGen, 64 bytes in Encaps/Decaps); we give the two uses
different domains.

Note: the Kyber round-3 key derivation returns SP_KDF(0x03 || K || SP_H(ct)),
whereas FIPS 203 takes K directly from G.  This module follows FIPS 203.
"""

from __future__ import annotations

from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS
from kyber_py.ml_kem.ml_kem import ML_KEM

from .domains import D, SP


class ML_KEM_A(ML_KEM):
    @staticmethod
    def _xof(b: bytes, i: bytes, j: bytes) -> bytes:
        data = b + i + j
        if len(data) != 34:
            raise ValueError("XOF input must be 34 bytes")
        # kyber-py samples from a fixed 840-byte buffer; keep the same length.
        return SP(D.KEM_XOF, data, 840)

    @staticmethod
    def _prf(eta: int, s: bytes, b: bytes) -> bytes:
        data = s + b
        if len(data) != 33:
            raise ValueError("PRF input must be 33 bytes")
        return SP(D.KEM_PRF, data, eta * 64)

    @staticmethod
    def _H(s: bytes) -> bytes:
        return SP(D.KEM_H, s, 32)

    @staticmethod
    def _J(s: bytes) -> bytes:
        return SP(D.KEM_J, s, 32)

    @staticmethod
    def _G(s: bytes) -> tuple[bytes, bytes]:
        dom = D.KEM_G_KEYGEN if len(s) == 33 else D.KEM_G_ENCAPS
        h = SP(dom, s, 64)
        return h[:32], h[32:]


ML_KEM_A_512 = ML_KEM_A(DEFAULT_PARAMETERS["ML512"])
ML_KEM_A_768 = ML_KEM_A(DEFAULT_PARAMETERS["ML768"])
