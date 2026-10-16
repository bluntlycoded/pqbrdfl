"""End-to-end aggregation round with real PQ crypto (liboqs backend)."""
import numpy as np
import pytest

from pqbrdfl.agg import protocol as P
from pqbrdfl.agg.robust import centered_clipping

N, DIM = 12, 400


def _setup(**kw):
    ids = [f"a{i:02d}" for i in range(N)]
    cfg = P.AggConfig(m=7, t=2, **kw)
    net = P.AggregationNetwork(ids, cfg)
    rng = np.random.default_rng(0)
    center = rng.normal(0, 0.1, DIM)
    ups = {a: center + rng.normal(0, 0.05, DIM) for a in ids}
    return net, ids, center, ups, rng


def test_honest_round_matches_plaintext_cc():
    net, ids, center, ups, rng = _setup()
    res = net.run_round(1, ups, center, rng=rng)
    assert res.accepted == ids and not res.rejected
    ref = centered_clipping(np.stack([ups[a] for a in ids]), center)
    assert np.max(np.abs(res.aggregate - ref)) < 1e-3
    assert not res.captured and not res.privacy_breach


def test_wrap_attack_blocked_by_range_check_only():
    for check in (True, False):
        net, ids, center, ups, rng = _setup(range_check=check)
        res = net.run_round(1, ups, center, client_behavior={ids[0]: P.C_WRAP}, rng=rng)
        if check:
            assert res.rejected.get(ids[0]) == "range"
            assert np.max(np.abs(res.aggregate - center)) < 1.0
        else:
            assert ids[0] in res.accepted          # naive norm check is fooled ...
            assert np.max(np.abs(res.aggregate)) > 1e6  # ... and the model is destroyed


def test_forged_and_inconsistent_clients():
    net, ids, center, ups, rng = _setup()
    beh = {ids[0]: P.C_FORGE, ids[1]: P.C_INCONSISTENT}
    res = net.run_round(1, ups, center, client_behavior=beh, rng=rng)
    assert res.rejected[ids[0]] == "channel"
    honest = [a for a in ids if a not in beh]
    assert set(honest) <= set(res.accepted)


def test_corrupt_members_corrected_and_flagged():
    net, ids, center, ups, rng = _setup()
    # find the committee for round 1 first (sortition is deterministic)
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, 7)
    beh = {com[0]: P.M_CORRUPT_AGG, com[3]: P.M_CORRUPT_SCALARS}
    res = net.run_round(1, ups, center, member_behavior=beh, rng=rng)
    ref = centered_clipping(np.stack([ups[a] for a in ids]), center)
    assert np.max(np.abs(res.aggregate - ref)) < 1e-3
    assert {com[0], com[3]} <= res.flagged_members


def test_capture_breaks_privacy_and_is_reported():
    net, ids, center, ups, rng = _setup()
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, 7)
    beh = {c: P.M_COLLUDE for c in com[:3]}          # t + 1 = 3 colluders
    res = net.run_round(1, ups, center, member_behavior=beh, rng=rng)
    assert res.captured and res.privacy_breach
    a = res.accepted[0]
    assert np.max(np.abs(res.recovered_updates[a] - ups[a])) < 1e-3


@pytest.mark.parametrize("mode,m,t", [("exact", 9, 2), ("sketch", 7, 2)])
def test_two_corrupt_members_still_deliver(mode, m, t):
    ids = [f"a{i:02d}" for i in range(N)]
    net = P.AggregationNetwork(ids, P.AggConfig(m=m, t=t, norm_mode=mode, k_proj=128))
    rng = np.random.default_rng(3)
    center = rng.normal(0, 0.1, DIM)
    ups = {a: center + rng.normal(0, 0.05, DIM) for a in ids}
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, m)
    beh = {com[1]: P.M_CORRUPT_SCALARS, com[4]: P.M_CORRUPT_AGG}
    res = net.run_round(1, ups, center, member_behavior=beh, rng=rng)
    assert len(res.accepted) == N
    ref = centered_clipping(np.stack([ups[a] for a in ids]), center)
    tol = 1e-3 if mode == "exact" else 0.05 * np.abs(ref - center).max() + 1e-3
    assert np.max(np.abs(res.aggregate - ref)) < tol


def test_norms_decode_exactly_with_zero_sharing():
    """The structural zero sharing added before opening must not shift the norms."""
    ids = [f"a{i:02d}" for i in range(N)]
    rng = np.random.default_rng(5)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    net = P.AggregationNetwork(ids, P.AggConfig(m=9, t=2))
    res = net.run_round(1, ups, center, rng=np.random.default_rng(1))
    for a in ids:
        assert abs(res.norms[a] - np.linalg.norm(ups[a])) < 1e-3


def _net(**kw):
    ids = [f"a{i:02d}" for i in range(N)]
    return ids, P.AggregationNetwork(ids, P.AggConfig(**kw))


def test_abort_mode_honest_majority_committee_is_safe():
    """m = 9, t = 4, sketch norms, abort policy: 4 corrupt members (> the 3t+1
    radius, = honest-majority limit) cannot make a wrong aggregate accepted."""
    ids, net = _net(m=9, t=4, norm_mode="sketch", k_proj=128, decode_policy="abort", max_retries=0)
    rng = np.random.default_rng(7)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, 9)
    beh = {c: P.M_CORRUPT_AGG for c in com[:4]}
    res = net.run_round(1, ups, center, member_behavior=beh, rng=rng)
    assert res.aborted and res.aggregate is None          # detected, not accepted
    # 4 errors exceed the unique-decoding radius (9 - 5) // 2 = 2, so the members
    # cannot be identified; safety comes from refusing to output, not from correction
    ids2, net2 = _net(m=9, t=4, norm_mode="sketch", k_proj=128, decode_policy="abort", max_retries=0)
    com2 = select_committee(net2.backend.H, net2.beacon, 1, ids2, 9)
    res2 = net2.run_round(1, ups, center, member_behavior={c: P.M_CORRUPT_AGG for c in com2[:2]}, rng=rng)
    assert res2.aborted and set(com2[:2]) <= res2.suspects  # within radius: identified


def test_retry_excludes_suspects_and_delivers():
    ids, net = _net(m=9, t=4, norm_mode="sketch", k_proj=128, decode_policy="abort", max_retries=3)
    rng = np.random.default_rng(8)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, 9)
    beh = {com[0]: P.M_CORRUPT_AGG, com[1]: P.M_CORRUPT_SCALARS}
    res = P.run_round_with_retries(net, 1, ups, center, member_behavior=beh, rng=rng)
    # the scalar-corrupter is caught in attempt 1, the aggregate-corrupter when it
    # next misbehaves; both are then excluded
    assert not res.aborted and 2 <= res.attempts <= 3
    assert not (set(beh) & set(res.committee))
    ref = centered_clipping(np.stack([ups[a] for a in res.accepted]), center)
    assert np.max(np.abs(res.aggregate - ref)) < 0.05 * np.abs(ref).max() + 1e-3


def test_mkcc_selects_half_and_excludes_outliers():
    ids, net = _net(m=9, t=4, norm_mode="sketch", k_proj=128, decode_policy="abort", robust="mkcc")
    rng = np.random.default_rng(9)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    for a in ids[:3]:
        ups[a] = rng.normal(0.3, 0.05, DIM)                 # a far-away cluster
    res = net.run_round(1, ups, center, rng=rng)
    assert len(res.weights) == N // 2
    assert not (set(ids[:3]) & set(res.weights))


def test_inconsistent_clients_cannot_frame_members_or_force_aborts():
    ids, net = _net(m=9, t=4, norm_mode="sketch", k_proj=128, decode_policy="abort", robust="mkcc")
    rng = np.random.default_rng(11)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    byz = ids[:4]
    res = P.run_round_with_retries(net, 1, ups, center, client_behavior={a: P.C_INCONSISTENT for a in byz}, rng=rng)
    assert not res.aborted and res.attempts == 1
    assert not net.banned                                          # no honest member framed
    assert all(res.rejected.get(a) == "inconsistent dealing" for a in byz)


def test_lying_member_is_proven_bad_without_abort():
    ids, net = _net(m=9, t=4, norm_mode="sketch", k_proj=128, decode_policy="abort", robust="cc")
    rng = np.random.default_rng(12)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, 9)
    res = P.run_round_with_retries(net, 1, ups, center, member_behavior={com[2]: P.M_CORRUPT_SCALARS}, rng=rng)
    assert not res.aborted and com[2] in net.banned
    assert set(net.banned) == {com[2]}


@pytest.mark.parametrize("policy,mode,m,k", [("abort", "sketch", 9, 128), ("correct", "exact", 9, 64)])
def test_orthogonal_aggregate_error_is_caught(policy, mode, m, k):
    ids, net = _net(m=m, t=2, norm_mode=mode, k_proj=k, decode_policy=policy, robust="cc")
    rng = np.random.default_rng(13)
    center = np.zeros(DIM)
    ups = {a: rng.normal(0, 0.05, DIM) for a in ids}
    from pqbrdfl.agg.committee import select_committee
    com = select_committee(net.backend.H, net.beacon, 1, ids, m)
    res = P.run_round_with_retries(net, 1, ups, center, member_behavior={com[0]: P.M_ORTHO}, rng=rng)
    assert res.aggregate is not None
    ref = centered_clipping(np.stack([ups[a] for a in res.accepted]), center)
    tol = 1e-3 if mode == "exact" else 0.05 * np.abs(ref - center).max() + 1e-3
    assert np.max(np.abs(res.aggregate - ref)) < tol
