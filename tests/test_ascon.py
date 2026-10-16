"""Ascon against the official SP 800-232 known-answer tests (ascon/ascon-c)."""
from pathlib import Path

import pytest

from pqbrdfl.crypto import ascon

VEC = Path(__file__).parent / "vectors"


def _records(name):
    rec = {}
    for line in (VEC / name).read_text().splitlines():
        line = line.strip()
        if not line:
            if rec:
                yield rec
            rec = {}
            continue
        k, _, v = line.partition("=")
        rec[k.strip()] = v.strip()
    if rec:
        yield rec


def _sample(recs, every):
    return [r for i, r in enumerate(recs) if i % every == 0]


@pytest.mark.parametrize("r", _sample(list(_records("ascon_hash256_kat.txt")), 16))
def test_hash256(r):
    assert ascon.hash256(bytes.fromhex(r["Msg"])).hex().upper() == r["MD"]


@pytest.mark.parametrize("r", _sample(list(_records("ascon_xof128_kat.txt")), 16))
def test_xof128(r):
    md = bytes.fromhex(r["MD"])
    assert ascon.xof128(bytes.fromhex(r["Msg"]), len(md)) == md


def test_xof_stream_matches_oneshot():
    x = ascon.XOF128(b"abc")
    streamed = x.read(3) + x.read(20) + x.read(41)
    assert streamed == ascon.xof128(b"abc", 64)


@pytest.mark.parametrize("r", _sample(list(_records("ascon_aead128_kat.txt")), 37))
def test_aead128(r):
    key, nonce = bytes.fromhex(r["Key"]), bytes.fromhex(r["Nonce"])
    pt, ad, ct = bytes.fromhex(r["PT"]), bytes.fromhex(r["AD"]), bytes.fromhex(r["CT"])
    assert ascon.aead128_encrypt(key, nonce, ad, pt) == ct
    assert ascon.aead128_decrypt(key, nonce, ad, ct) == pt


def test_aead128_rejects_tamper():
    k, n = bytes(16), bytes(range(16))
    ct = bytearray(ascon.aead128_encrypt(k, n, b"hdr", b"secret share"))
    ct[0] ^= 1
    with pytest.raises(ValueError):
        ascon.aead128_decrypt(k, n, b"hdr", bytes(ct))
