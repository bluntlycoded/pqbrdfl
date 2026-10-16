"""Ascon permutation and the NIST SP 800-232 modes used by the framework.

Pure Python, written for clarity and for checking against the official
known-answer tests (tests/test_ascon.py), not for speed.  Words are 64-bit
little-endian as in SP 800-232.

Implemented:
  * ascon_p          -- the 320-bit permutation (12 or 8 rounds)
  * hash256          -- Ascon-Hash256
  * xof128           -- Ascon-XOF128 (arbitrary output length)
  * aead128_encrypt  -- Ascon-AEAD128
  * aead128_decrypt
"""

from __future__ import annotations

MASK = (1 << 64) - 1
_RC = (0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B)

# IVs from SP 800-232 (variant | pa << 16 | pb << 20 | taglen << 24 | rate << 40)
IV_AEAD128 = 0x00001000808C0001
IV_HASH256 = 0x0000080100CC0002
IV_XOF128 = 0x0000080000CC0003

RATE_HASH = 8
RATE_AEAD = 16

# Operation counter, used by the benchmarks to report permutation calls.
PERMUTATION_CALLS = 0


def _ror(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & MASK


def ascon_p(s: list[int], rounds: int = 12) -> None:
    """Apply the Ascon permutation in place to the 5-word state ``s``."""
    global PERMUTATION_CALLS
    PERMUTATION_CALLS += 1
    x0, x1, x2, x3, x4 = s
    for c in _RC[12 - rounds:]:
        x2 ^= c
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = x0 ^ (~x1 & x2)
        t1 = x1 ^ (~x2 & x3)
        t2 = x2 ^ (~x3 & x4)
        t3 = x3 ^ (~x4 & x0)
        t4 = x4 ^ (~x0 & x1)
        t1 ^= t0
        t0 ^= t4
        t3 ^= t2
        t2 = ~t2 & MASK
        x0 = t0 ^ _ror(t0, 19) ^ _ror(t0, 28)
        x1 = t1 ^ _ror(t1, 61) ^ _ror(t1, 39)
        x2 = t2 ^ _ror(t2, 1) ^ _ror(t2, 6)
        x3 = t3 ^ _ror(t3, 10) ^ _ror(t3, 17)
        x4 = t4 ^ _ror(t4, 7) ^ _ror(t4, 41)
    s[0], s[1], s[2], s[3], s[4] = x0, x1, x2, x3, x4


def _load(b: bytes) -> int:
    return int.from_bytes(b, "little")


def _store(x: int, n: int) -> bytes:
    return (x & MASK).to_bytes(8, "little")[:n]


def _pad(n: int) -> int:
    return 0x01 << (8 * n)


class _Sponge:
    """Unkeyed Ascon sponge (rate 64 bits, capacity 256 bits)."""

    def __init__(self, iv: int):
        self.s = [iv, 0, 0, 0, 0]
        ascon_p(self.s, 12)
        self._buf = b""
        self._squeezing = False
        self._out = b""

    def absorb(self, data: bytes) -> "_Sponge":
        if self._squeezing:
            raise ValueError("cannot absorb after squeezing")
        data = self._buf + data
        n_full = len(data) // RATE_HASH
        s = self.s
        for i in range(n_full):
            s[0] ^= _load(data[8 * i: 8 * i + 8])
            ascon_p(s, 12)
        self._buf = data[8 * n_full:]
        return self

    def _finish_absorb(self) -> None:
        s = self.s
        s[0] ^= _load(self._buf) ^ _pad(len(self._buf))
        ascon_p(s, 12)
        self._buf = b""
        self._squeezing = True

    def squeeze(self, n: int) -> bytes:
        if not self._squeezing:
            self._finish_absorb()
        out = bytearray(self._out)
        while len(out) < n:
            out += _store(self.s[0], 8)
            ascon_p(self.s, 12)
        self._out = bytes(out[n:])
        return bytes(out[:n])


def hash256(msg: bytes) -> bytes:
    return _Sponge(IV_HASH256).absorb(msg).squeeze(32)


def xof128(msg: bytes, outlen: int) -> bytes:
    return _Sponge(IV_XOF128).absorb(msg).squeeze(outlen)


class XOF128:
    """Streaming Ascon-XOF128 with a ``read`` interface (used by ML-DSA_A)."""

    def __init__(self, msg: bytes):
        self._sp = _Sponge(IV_XOF128).absorb(msg)

    def read(self, n: int) -> bytes:
        return self._sp.squeeze(n)


def _aead_init(key: bytes, nonce: bytes) -> tuple[list[int], int, int]:
    if len(key) != 16 or len(nonce) != 16:
        raise ValueError("Ascon-AEAD128 needs a 16-byte key and 16-byte nonce")
    k0, k1 = _load(key[:8]), _load(key[8:])
    s = [IV_AEAD128, k0, k1, _load(nonce[:8]), _load(nonce[8:])]
    ascon_p(s, 12)
    s[3] ^= k0
    s[4] ^= k1
    return s, k0, k1


def _aead_absorb_ad(s: list[int], ad: bytes) -> None:
    if ad:
        n_full = len(ad) // RATE_AEAD
        for i in range(n_full):
            blk = ad[16 * i: 16 * i + 16]
            s[0] ^= _load(blk[:8])
            s[1] ^= _load(blk[8:])
            ascon_p(s, 8)
        rest = ad[16 * n_full:]
        if len(rest) >= 8:
            s[0] ^= _load(rest[:8])
            s[1] ^= _load(rest[8:]) ^ _pad(len(rest) - 8)
        else:
            s[0] ^= _load(rest) ^ _pad(len(rest))
        ascon_p(s, 8)
    s[4] ^= 0x80 << 56  # domain separation bit


def _aead_final(s: list[int], k0: int, k1: int) -> bytes:
    s[2] ^= k0
    s[3] ^= k1
    ascon_p(s, 12)
    s[3] ^= k0
    s[4] ^= k1
    return _store(s[3], 8) + _store(s[4], 8)


def aead128_encrypt(key: bytes, nonce: bytes, ad: bytes, pt: bytes) -> bytes:
    """Return ciphertext || 16-byte tag."""
    s, k0, k1 = _aead_init(key, nonce)
    _aead_absorb_ad(s, ad)
    out = bytearray()
    n_full = len(pt) // RATE_AEAD
    for i in range(n_full):
        blk = pt[16 * i: 16 * i + 16]
        s[0] ^= _load(blk[:8])
        s[1] ^= _load(blk[8:])
        out += _store(s[0], 8) + _store(s[1], 8)
        ascon_p(s, 8)
    rest = pt[16 * n_full:]
    if len(rest) >= 8:
        s[0] ^= _load(rest[:8])
        s[1] ^= _load(rest[8:])
        out += _store(s[0], 8) + _store(s[1], len(rest) - 8)
        s[1] ^= _pad(len(rest) - 8)
    else:
        s[0] ^= _load(rest)
        out += _store(s[0], len(rest))
        s[0] ^= _pad(len(rest))
    return bytes(out) + _aead_final(s, k0, k1)


def aead128_decrypt(key: bytes, nonce: bytes, ad: bytes, ct: bytes) -> bytes:
    """Return plaintext, or raise ValueError if the tag does not verify."""
    if len(ct) < 16:
        raise ValueError("ciphertext too short")
    body, tag = ct[:-16], ct[-16:]
    s, k0, k1 = _aead_init(key, nonce)
    _aead_absorb_ad(s, ad)
    out = bytearray()
    n_full = len(body) // RATE_AEAD
    for i in range(n_full):
        c0 = _load(body[16 * i: 16 * i + 8])
        c1 = _load(body[16 * i + 8: 16 * i + 16])
        out += _store(s[0] ^ c0, 8) + _store(s[1] ^ c1, 8)
        s[0], s[1] = c0, c1
        ascon_p(s, 8)
    rest = body[16 * n_full:]
    if len(rest) >= 8:
        c0 = _load(rest[:8])
        n1 = len(rest) - 8
        c1 = _load(rest[8:])
        out += _store(s[0] ^ c0, 8) + _store(s[1] ^ c1, n1)
        s[0] = c0
        keep = s[1] & ~((1 << (8 * n1)) - 1) & MASK
        s[1] = (keep | c1) ^ _pad(n1)
    else:
        n0 = len(rest)
        c0 = _load(rest)
        out += _store(s[0] ^ c0, n0)
        keep = s[0] & ~((1 << (8 * n0)) - 1) & MASK
        s[0] = (keep | c0) ^ _pad(n0)
    expected = _aead_final(s, k0, k1)
    # constant-time comparison
    import hmac
    if not hmac.compare_digest(expected, tag):
        raise ValueError("authentication failed")
    return bytes(out)
