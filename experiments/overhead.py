"""Cost experiments (Section VII-D): primitives, sizes, protocol scaling,
and an end-to-end round on the single-permutation (Ascon) backend.

    python -m experiments.overhead

Run on an idle machine: wall-clock numbers are medians of repeated runs.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from pqbrdfl.agg import protocol as P  # noqa: E402
from pqbrdfl.agg.robust import centered_clipping  # noqa: E402
from pqbrdfl.crypto import ascon  # noqa: E402
from pqbrdfl.crypto.backend import get_backend  # noqa: E402


def _median_ms(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def primitives(reps=200):
    rows = []
    for name in ("fips", "classical"):
        B = get_backend(name)
        pk, sk = B.kem.keygen()
        ct, ss = B.kem.encaps(pk)
        spk, ssk = B.sig.keygen()
        msg = b"x" * 64
        sig = B.sig.sign(ssk, msg)
        rows.append({
            "backend": name, "kem": B.kem.name, "sig": B.sig.name,
            "kem_keygen_ms": _median_ms(B.kem.keygen, reps),
            "encaps_ms": _median_ms(lambda: B.kem.encaps(pk), reps),
            "decaps_ms": _median_ms(lambda: B.kem.decaps(sk, ct), reps),
            "sig_keygen_ms": _median_ms(B.sig.keygen, reps),
            "sign_ms": _median_ms(lambda: B.sig.sign(ssk, msg), reps),
            "verify_ms": _median_ms(lambda: B.sig.verify(spk, msg, sig), reps),
            "kem_pk_B": len(pk), "kem_ct_B": len(ct), "sig_pk_B": len(spk), "sig_B": len(sig),
        })
    # the single-permutation instantiation (pure Python): time + Ascon-p calls
    B = get_backend("ascon")
    pk, sk = B.kem.keygen()
    ct, ss = B.kem.encaps(pk)
    spk, ssk = B.sig.keygen()
    msg = b"x" * 64
    sig = B.sig.sign(ssk, msg)
    row = {"backend": "ascon", "kem": B.kem.name, "sig": B.sig.name,
           "kem_pk_B": len(pk), "kem_ct_B": len(ct), "sig_pk_B": len(spk), "sig_B": len(sig)}
    for key, fn in (("kem_keygen", B.kem.keygen), ("encaps", lambda: B.kem.encaps(pk)),
                    ("decaps", lambda: B.kem.decaps(sk, ct)), ("sig_keygen", B.sig.keygen),
                    ("sign", lambda: B.sig.sign(ssk, msg)), ("verify", lambda: B.sig.verify(spk, msg, sig))):
        calls = []
        ts = []
        for _ in range(7):
            c0 = ascon.PERMUTATION_CALLS
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1e3)
            calls.append(ascon.PERMUTATION_CALLS - c0)
        row[f"{key}_ms"] = statistics.median(ts)
        row[f"{key}_ascon_p_calls"] = statistics.median(calls)
    rows.append(row)
    return rows


def protocol_scaling():
    rows = []
    rng = np.random.default_rng(0)
    configs = [  # (n, m, t, policy, norm_mode, k)
        (20, 17, 4, "correct", "exact", 64),
        (20, 9, 4, "abort", "sketch", 128),
        (40, 9, 4, "abort", "sketch", 128),
        (100, 13, 6, "abort", "sketch", 128),
    ]
    for backend in ("fips", "classical"):
        for n, m, t, pol, mode, k in configs:
            for d in (6535, 50000):
                ids = [f"a{i:03d}" for i in range(n)]
                net = P.AggregationNetwork(ids, P.AggConfig(m=m, t=t, backend=backend, decode_policy=pol,
                                                            norm_mode=mode, k_proj=k, robust="cc"))
                center = rng.normal(0, 0.01, d)
                ups = {a: center + rng.normal(0, 0.01, d) for a in ids}
                t0 = time.perf_counter()
                res = net.run_round(1, ups, center, rng=rng)
                wall = time.perf_counter() - t0
                ref = centered_clipping(np.stack([ups[a] for a in ids]), center)
                total_bytes = sum(res.bytes_by_kind.values())
                rows.append({
                    "backend": backend, "policy": pol, "norm_mode": mode, "n": n, "m": m, "t": t, "d": d,
                    "round_wall_s": wall, "phases_s": res.time_by_phase,
                    "bytes": res.bytes_by_kind, "bytes_total": total_bytes,
                    "bytes_per_agent_upload": res.bytes_by_kind["shares"] / n,
                    "plaintext_fedavg_upload_per_agent": 4 * d,
                    "classical_secagg_upload_per_agent": 8 * d + 32 * n,
                    "max_abs_error_vs_plaintext_cc": float(np.max(np.abs(res.aggregate - ref))),
                    "relative_error_vs_plaintext_cc": float(np.linalg.norm(res.aggregate - ref) /
                                                            (np.linalg.norm(ref - center) + 1e-12)),
                })
                print(rows[-1]["backend"], n, d, round(wall, 3), total_bytes, flush=True)
    return rows


def ascon_end_to_end():
    """Two rounds with every cryptographic call on the one Ascon permutation."""
    rng = np.random.default_rng(1)
    ids = [f"a{i}" for i in range(6)]
    d = 300
    c0 = ascon.PERMUTATION_CALLS
    net = P.AggregationNetwork(ids, P.AggConfig(m=5, t=2, backend="ascon", k_proj=128, norm_mode="sketch",
                                                decode_policy="abort", robust="cc"))
    out = []
    center = np.zeros(d)
    for r in (1, 2):
        ups = {a: center + rng.normal(0, 0.05, d) for a in ids}
        t = time.perf_counter()
        res = net.run_round(r, ups, center, rng=rng)
        ref = centered_clipping(np.stack([ups[a] for a in ids]), center)
        out.append({"round": r, "wall_s": time.perf_counter() - t,
                    "accepted": len(res.accepted), "max_abs_error": float(np.max(np.abs(res.aggregate - ref)))})
        center = res.aggregate
    return {"n": 6, "m": 5, "t": 2, "d": d, "rounds": out,
            "ascon_p_calls_total": ascon.PERMUTATION_CALLS - c0,
            "audit_root": net.audit.root().hex()}


def main():
    res = {"machine": platform.platform(), "processor": platform.processor(),
           "python": platform.python_version()}
    res["primitives"] = primitives()
    res["protocol_scaling"] = protocol_scaling()
    res["ascon_end_to_end"] = ascon_end_to_end()
    Path("results").mkdir(exist_ok=True)
    Path("results/overhead.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["ascon_end_to_end"], indent=1))


if __name__ == "__main__":
    main()
