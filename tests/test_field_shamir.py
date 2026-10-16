import numpy as np
import pytest

from pqbrdfl.agg import field as F
from pqbrdfl.agg import shamir as S
from pqbrdfl.agg.field import P


def test_mul_matches_python():
    rng = np.random.default_rng(0)
    a = F.rand(2000); b = F.rand(2000)
    ref = np.array([int(x) * int(y) % P for x, y in zip(a, b)], dtype=np.uint64)
    assert np.array_equal(F.mul(a, b), ref)
    edge = np.array([0, 1, P - 1, P - 2, 1 << 60], dtype=np.uint64)
    assert np.array_equal(F.mul(edge, edge), np.array([int(x) ** 2 % P for x in edge], dtype=np.uint64))


def test_sum_and_dot_binary():
    a = F.rand((5, 3000))
    assert all(int(F.sum_mod(a)[i]) == sum(int(v) for v in a[i]) % P for i in range(5))
    R = (np.random.default_rng(1).random((7, 3000)) < 0.5).astype(np.uint8)
    y = F.rand(3000)
    ref = [sum(int(v) for v, r in zip(y, row) if r) % P for row in R]
    assert [int(v) for v in F.dot_binary(R, y)] == ref


def test_encode_decode_unbiased():
    x = np.full(200000, 0.3)
    assert abs(F.decode(F.encode(x, 4), 4).mean() - 0.3) < 2e-3
    z = np.array([-3.25, 0.0, 7.5])
    assert np.allclose(F.decode(F.encode(z, 16), 16), z)


@pytest.mark.parametrize("t,m", [(1, 4), (2, 7), (3, 10)])
def test_share_reconstruct(t, m):
    s = F.rand(50)
    sh = S.share(s, t, m)
    xs = list(range(1, m + 1))
    assert np.array_equal(S.reconstruct(xs[:t + 1], sh[:t + 1]), s)
    assert np.array_equal(S.reconstruct(xs[-(t + 1):], sh[-(t + 1):]), s)


def test_t_shares_reveal_nothing_structurally():
    # any t shares are uniform: two different secrets give identically distributed
    # t-subsets; check first moments of a coordinate as a smoke test
    t, m = 2, 7
    a = np.array([S.share(np.array([0], np.uint64), t, m)[:t, 0] for _ in range(3000)], dtype=np.float64)
    assert abs(a.mean() / P - 0.5) < 0.02


def test_berlekamp_welch_corrects_and_identifies():
    t, m = 2, 7                       # degree t, corrects (7-3)//2 = 2 errors
    secret = 123456789
    coeffs = [secret, 17, 99]
    xs = list(range(1, m + 1))
    ys = [S._poly_eval(coeffs, x) for x in xs]
    ys[1] = (ys[1] + 5) % P
    ys[5] = 42
    val, bad = S.decode_scalar(xs, ys, t)
    assert val == secret and sorted(bad) == [1, 5]


def test_berlekamp_welch_detects_beyond_radius():
    xs = list(range(1, 8))
    ys = [S._poly_eval([5, 1, 1], x) for x in xs]
    for i in (0, 2, 4):
        ys[i] = (ys[i] + 1000 + i) % P
    out = S.decode_scalar(xs, ys, 2)
    # 3 errors > radius 2: must not return the wrong secret silently
    assert out is None or out[0] == 5


def test_decode_vector_identifies_bad_member():
    t, m = 2, 7
    s = F.rand(100)
    sh = S.share(s, t, m)
    sh[3] = F.add(sh[3], F.rand(100))
    rho = F.rand(100)
    out, bad = S.decode_vector(list(range(1, m + 1)), sh, t, rho)
    assert np.array_equal(out, s) and bad == [3]
