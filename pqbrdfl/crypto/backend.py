"""KEM / signature / AEAD backends behind one interface.

Backends
--------
``fips``      ML-KEM-512 + ML-DSA-44 via liboqs (C, SHA-3 hashing as in FIPS 203/204),
              AES-256-GCM transport.  Default for simulations: fast and standard.
``ascon``     ML-KEM_A-512 + ML-DSA_A-44 + Ascon-AEAD128, every symmetric call on the
              one domain-separated Ascon permutation (the single-permutation construction).
              Pure Python: functionally complete, slow; used for validation runs.
``classical`` X25519 (as a KEM, DHKEM-style) + Ed25519 + AES-256-GCM.  NOT
              post-quantum; exists only as the baseline for the ablation and the
              harvest-now-decrypt-later experiment.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization

from . import ascon
from .domains import D, SP

_RAW = serialization.Encoding.Raw
_RAWPUB = serialization.PublicFormat.Raw
_RAWPRIV = serialization.PrivateFormat.Raw
_NOENC = serialization.NoEncryption()


@dataclass(frozen=True)
class KEM:
    name: str
    keygen: Callable[[], tuple[bytes, bytes]]          # -> (pk, sk)
    encaps: Callable[[bytes], tuple[bytes, bytes]]     # pk -> (ct, ss)
    decaps: Callable[[bytes, bytes], bytes]            # (sk, ct) -> ss
    post_quantum: bool


@dataclass(frozen=True)
class SIG:
    name: str
    keygen: Callable[[], tuple[bytes, bytes]]          # -> (pk, sk)
    sign: Callable[[bytes, bytes], bytes]              # (sk, msg) -> sig
    verify: Callable[[bytes, bytes, bytes], bool]      # (pk, msg, sig) -> ok
    post_quantum: bool


@dataclass(frozen=True)
class AEAD:
    name: str
    key_len: int
    nonce_len: int
    seal: Callable[[bytes, bytes, bytes, bytes], bytes]   # (key, nonce, ad, pt)
    open: Callable[[bytes, bytes, bytes, bytes], bytes]   # raises ValueError


def _shake_sp(domain: int, data: bytes, n: int = 32) -> bytes:
    import hashlib
    return hashlib.shake_256(bytes([int(domain)]) + data).digest(n)


@dataclass(frozen=True)
class Backend:
    name: str
    kem: KEM
    sig: SIG
    aead: AEAD
    # Domain-separated protocol hash H(domain, data, outlen).  Same prefix
    # discipline in every backend; the ``ascon`` backend runs it on the shared
    # Ascon permutation, the others on SHAKE256 (C speed).
    H: Callable[..., bytes] = _shake_sp

    @property
    def post_quantum(self) -> bool:
        return self.kem.post_quantum and self.sig.post_quantum


# ---------------------------------------------------------------- AEADs
def _aesgcm_seal(key, nonce, ad, pt):
    return AESGCM(key).encrypt(nonce, pt, ad)


def _aesgcm_open(key, nonce, ad, ct):
    from cryptography.exceptions import InvalidTag
    try:
        return AESGCM(key).decrypt(nonce, ct, ad)
    except InvalidTag as e:
        raise ValueError("authentication failed") from e


AES256GCM = AEAD("AES-256-GCM", 32, 12, _aesgcm_seal, _aesgcm_open)
ASCON_AEAD128 = AEAD("Ascon-AEAD128", 16, 16, ascon.aead128_encrypt, ascon.aead128_decrypt)


# ---------------------------------------------------------------- liboqs
def _oqs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import oqs  # noqa: WPS433
    return oqs


def _fips_kem(alg="ML-KEM-512") -> KEM:
    oqs = _oqs()

    def keygen():
        k = oqs.KeyEncapsulation(alg)
        pk = k.generate_keypair()
        return pk, k.export_secret_key()

    def encaps(pk):
        return oqs.KeyEncapsulation(alg).encap_secret(pk)

    def decaps(sk, ct):
        return oqs.KeyEncapsulation(alg, secret_key=sk).decap_secret(ct)

    return KEM(alg, keygen, encaps, decaps, True)


def _fips_sig(alg="ML-DSA-44") -> SIG:
    oqs = _oqs()

    def keygen():
        s = oqs.Signature(alg)
        pk = s.generate_keypair()
        return pk, s.export_secret_key()

    def sign(sk, msg):
        return oqs.Signature(alg, secret_key=sk).sign(msg)

    def verify(pk, msg, sig):
        try:
            return bool(oqs.Signature(alg).verify(msg, sig, pk))
        except Exception:
            return False

    return SIG(alg, keygen, sign, verify, True)


# ---------------------------------------------------------------- Ascon (single permutation)
def _ascon_kem() -> KEM:
    from .mlkem_a import ML_KEM_A_512 as K

    def keygen():
        ek, dk = K.keygen()
        return ek, dk

    def encaps(pk):
        ss, ct = K.encaps(pk)
        return ct, ss

    return KEM("ML-KEM_A-512", keygen, encaps, lambda sk, ct: K.decaps(sk, ct), True)


def _ascon_sig() -> SIG:
    from .mldsa_a import ML_DSA_A_44 as S

    def verify(pk, msg, sig):
        try:
            return bool(S.verify(pk, msg, sig))
        except Exception:
            return False

    return SIG("ML-DSA_A-44", S.keygen, lambda sk, m: S.sign(sk, m), verify, True)


# ---------------------------------------------------------------- classical baseline
def _x25519_kem() -> KEM:
    """DHKEM(X25519) in the style of RFC 9180: ct = ephemeral pk, ss = KDF(dh || ct || pk)."""

    def keygen():
        sk = X25519PrivateKey.generate()
        return sk.public_key().public_bytes(_RAW, _RAWPUB), sk.private_bytes(_RAW, _RAWPRIV, _NOENC)

    def _ss(dh, ct, pk):
        return _shake_sp(D.CHAN_KDF, b"dhkem" + dh + ct + pk, 32)

    def encaps(pk):
        e = X25519PrivateKey.generate()
        ct = e.public_key().public_bytes(_RAW, _RAWPUB)
        dh = e.exchange(X25519PublicKey.from_public_bytes(pk))
        return ct, _ss(dh, ct, pk)

    def decaps(sk, ct):
        s = X25519PrivateKey.from_private_bytes(sk)
        pk = s.public_key().public_bytes(_RAW, _RAWPUB)
        dh = s.exchange(X25519PublicKey.from_public_bytes(ct))
        return _ss(dh, ct, pk)

    return KEM("X25519-DHKEM", keygen, encaps, decaps, False)


def _ed25519_sig() -> SIG:
    def keygen():
        sk = Ed25519PrivateKey.generate()
        return sk.public_key().public_bytes(_RAW, _RAWPUB), sk.private_bytes(_RAW, _RAWPRIV, _NOENC)

    def sign(sk, msg):
        return Ed25519PrivateKey.from_private_bytes(sk).sign(msg)

    def verify(pk, msg, sig):
        try:
            Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
            return True
        except Exception:
            return False

    return SIG("Ed25519", keygen, sign, verify, False)


_CACHE: dict[str, Backend] = {}


def get_backend(name: str = "fips") -> Backend:
    if name not in _CACHE:
        if name == "fips":
            _CACHE[name] = Backend(name, _fips_kem(), _fips_sig(), AES256GCM)
        elif name == "ascon":
            _CACHE[name] = Backend(name, _ascon_kem(), _ascon_sig(), ASCON_AEAD128, SP)
        elif name == "classical":
            _CACHE[name] = Backend(name, _x25519_kem(), _ed25519_sig(), AES256GCM)
        else:
            raise ValueError(f"unknown backend {name!r}")
    return _CACHE[name]


def random_bytes(n: int) -> bytes:
    return os.urandom(n)
