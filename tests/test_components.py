import numpy as np
import pytest

from pqbrdfl.agg import secagg_classical as SA
from pqbrdfl.agg.committee import capture_probability, select_committee
from pqbrdfl.crypto import channel as ch
from pqbrdfl.crypto.backend import get_backend
from pqbrdfl.crypto.domains import D
from pqbrdfl.crypto.merkle import MerkleLog
from pqbrdfl.fl import telemetry as T
from pqbrdfl.fl.model import MLP


def test_mlp_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    m = MLP(6, 3, hidden=(5, 4), seed=1)
    X, y = rng.normal(size=(9, 6)), rng.integers(0, 3, 9)
    _, g = m.grad(m.w, X, y)
    for k in rng.choice(m.dim, 12, replace=False):
        e = np.zeros(m.dim); e[k] = 1e-6
        lp, _ = m.grad(m.w + e, X, y)
        lm, _ = m.grad(m.w - e, X, y)
        assert abs((lp - lm) / 2e-6 - g[k]) < 1e-5


@pytest.mark.parametrize("backend", ["fips", "classical", "ascon"])
def test_channel_roundtrip_and_tamper(backend):
    B = get_backend(backend)
    alice, bob = ch.Identity.generate(B, "alice"), ch.Identity.generate(B, "bob")
    eph = ch.EphemeralKey.generate(B, bob, 3)
    assert ch.verify_ephemeral(B, bob.pk, "bob", 3, eph.pk, eph.sig)
    msg = ch.seal(B, alice, "bob", eph.pk, 3, "share", b"secret share bytes")
    assert ch.open_sealed(B, msg, alice.pk, eph) == b"secret share bytes"
    bad = ch.Sealed(**{**msg.__dict__, "body": bytes([msg.body[0] ^ 1]) + msg.body[1:]})
    with pytest.raises(ch.ChannelError):
        ch.open_sealed(B, bad, alice.pk, eph)
    mallory = ch.Identity.generate(B, "alice")          # impersonation without the registered key
    forged = ch.seal(B, mallory, "bob", eph.pk, 3, "share", b"poison")
    with pytest.raises(ch.ChannelError):
        ch.open_sealed(B, forged, alice.pk, eph)


def test_merkle_inclusion():
    B = get_backend("fips")
    log = MerkleLog(B.H)
    for i in range(11):
        log.append(bytes([i]) * 5)
    root = log.root()
    for i in range(11):
        assert log.verify(bytes([i]) * 5, i, log.proof(i), root)
    assert not log.verify(b"forged", 3, log.proof(3), root)


def test_sortition_deterministic_and_capture_bound():
    B = get_backend("fips")
    ids = [f"a{i}" for i in range(30)]
    c1 = select_committee(B.H, b"beacon", 5, ids, 7)
    assert c1 == select_committee(B.H, b"beacon", 5, ids, 7)
    assert c1 != select_committee(B.H, b"beacon", 6, ids, 7)
    assert capture_probability(20, 4, 17, 4) == 0.0
    # Monte-Carlo agreement with the hypergeometric tail
    rng = np.random.default_rng(0)
    hits = np.mean([(rng.choice(60, 13, replace=False) < 12).sum() > 3 for _ in range(20000)])
    assert abs(hits - capture_probability(60, 12, 13, 3)) < 0.01


def test_hndl_recovers_every_update():
    rng = np.random.default_rng(0)
    ups = {f"c{i}": rng.normal(0, 0.1, 500) for i in range(5)}
    agg, tr, sks = SA.run_secagg(ups, rng=rng)
    assert np.allclose(agg, np.mean(list(ups.values()), axis=0), atol=1e-3)
    rec = SA.hndl_recover(tr, sks)
    for a in ups:
        assert np.max(np.abs(rec[a] - ups[a])) < 2 ** -13


def test_signed_telemetry_detects_in_path_tampering():
    B = get_backend("fips")
    sw = ch.Identity.generate(B, "sw1")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 4)), rng.integers(0, 3, 50)
    b = T.export_batch(B, sw, 0, X, y)
    assert T.verify_batch(B, sw.pk, b)
    tampered = T.tamper_in_path(b, 0, rng, np.zeros(4))
    assert not T.verify_batch(B, sw.pk, tampered)


def test_benign_gate_and_calibrated_local_model():
    from pqbrdfl.fl.posthoc import BenignGate, LocalSecondOpinion, AnyGate
    rng = np.random.default_rng(0)
    Xb = rng.normal(0, 1, (2000, 6))
    gate = BenignGate(Xb, 0.99)
    assert 0.005 < gate.flag(rng.normal(0, 1, (5000, 6))).mean() < 0.02   # ~1% of fresh benign flagged
    assert gate.flag(np.full((10, 6), 6.0)).all()                           # far-off flows flagged
    X = np.vstack([Xb, rng.normal(3, 1, (2000, 6))])
    y = np.r_[np.zeros(2000, int), np.ones(2000, int)]
    m = MLP(6, 2, hidden=(8, 8), seed=0)
    lc = LocalSecondOpinion.calibrated(m, X, y, normal=0, target_fpr=0.02, seed=0)
    assert lc is not None and lc.flag(rng.normal(0, 1, (5000, 6))).mean() < 0.06
    assert AnyGate(gate, lc).flag(np.full((3, 6), 6.0)).all()
    assert LocalSecondOpinion.calibrated(m, X[1990:], y[1990:], normal=0) is None  # too few benign flows
