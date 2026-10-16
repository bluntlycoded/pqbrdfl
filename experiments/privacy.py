"""Privacy experiments (Section VII-C).

  P1  Harvest-now-decrypt-later against classical pairwise-mask SecAgg:
      record one round's transcript, hand the future adversary a discrete-log
      oracle, recover every agent's individual update.
  P2  What a recovered update reveals in the SDN setting: the last-layer bias
      delta ranks the traffic classes a switch saw (which attacks hit which
      site).  Spearman correlation with the true local class distribution.
  P3  What our protocol's *public leakage* (R y, 1^T y, ||y||) reveals: the
      best linear estimate of y from it, its cosine to the true y, and the
      same class-distribution inference run on that estimate.
  P4  Committee capture: smallest committee meeting a per-round capture target.

    python -m experiments.privacy --dataset cicids
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from pqbrdfl.agg import secagg_classical as SA  # noqa: E402
from pqbrdfl.agg import field as F  # noqa: E402
from pqbrdfl.agg.committee import capture_probability  # noqa: E402
from pqbrdfl.agg.protocol import challenge  # noqa: E402
from pqbrdfl.crypto.backend import get_backend  # noqa: E402
from pqbrdfl.fl.data import load, partition  # noqa: E402
from pqbrdfl.fl.model import MLP  # noqa: E402


def one_round_updates(ds, n_agents=20, alpha=0.5, seed=0, lr=3e-3, warm_rounds=3):
    rng = np.random.default_rng(seed)
    parts = partition(ds.y_train, n_agents, alpha, rng)
    model = MLP(ds.n_features, ds.n_classes, seed=seed)
    w = model.w.copy()
    local = [(ds.X_train[p[:4000]], ds.y_train[p[:4000]]) for p in parts]
    for _ in range(warm_rounds):        # a few FedAvg rounds so updates are mid-training
        w = w + np.mean([model.train_local(w, X, y, 1, lr, 128, rng) - w for X, y in local], axis=0)
    ups = {f"sw{i:02d}": model.train_local(w, X, y, 1, lr, 128, rng) - w for i, (X, y) in enumerate(local)}
    props = {f"sw{i:02d}": np.bincount(y, minlength=ds.n_classes) / len(y) for i, (X, y) in enumerate(local)}
    return model, ups, props


def bias_slice(model: MLP, v: np.ndarray) -> np.ndarray:
    return v[-model.sizes[-1]:]


def class_inference(model, updates: dict, props: dict) -> float:
    rs = []
    for a, u in updates.items():
        r = spearmanr(bias_slice(model, u), props[a]).correlation
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    ds = load(args.dataset, args.data_root)
    out = {"dataset": ds.name, "synthetic_data": ds.synthetic, "seeds": []}
    B = get_backend("fips")
    for seed in args.seeds:
        model, ups, props = one_round_updates(ds, seed=seed)
        d = model.dim
        # P1: HNDL on classical SecAgg
        agg, tr, sks = SA.run_secagg(ups, frac_bits=14, rng=np.random.default_rng(seed))
        rec = SA.hndl_recover(tr, sks)
        err = max(float(np.max(np.abs(rec[a] - ups[a]))) for a in ups)
        agg_err = float(np.max(np.abs(agg - np.mean(list(ups.values()), axis=0))))
        # P2: class-distribution inference from recovered updates vs chance
        inf_rec = class_inference(model, rec, props)
        rng = np.random.default_rng(seed)
        inf_rand = class_inference(model, {a: rng.normal(size=d) for a in ups}, props)
        # P3: what the protocol's public leakage reveals (range-check projections)
        R, _ = challenge(B, bytes([seed]) * 32, args.k, d)
        L = np.vstack([R.astype(np.float64), np.ones((1, d))])       # published linear leakage
        Lp = np.linalg.pinv(L)
        est, cos = {}, []
        for a, u in ups.items():
            y = u                                                     # centre 0 in round 1
            e = Lp @ (L @ y)                                          # min-norm consistent estimate
            nrm = np.linalg.norm(y)
            e = e / (np.linalg.norm(e) + 1e-12) * nrm                 # the norm is published too
            est[a] = e
            cos.append(float(e @ y / (np.linalg.norm(e) * nrm + 1e-12)))
        inf_leak = class_inference(model, est, props)
        out["seeds"].append({
            "seed": seed, "d": d, "k": args.k,
            "hndl_max_abs_error": err, "hndl_quantisation_step": 2.0 ** -14,
            "hndl_recovered_fraction": float(np.mean([np.max(np.abs(rec[a] - ups[a])) <= 2.0 ** -13 for a in ups])),
            "secagg_aggregate_error": agg_err,
            "class_inference_spearman_recovered": inf_rec,
            "class_inference_spearman_leakage": inf_leak,
            "class_inference_spearman_random": inf_rand,
            "leakage_estimate_cosine_mean": float(np.mean(cos)),
            "leakage_estimate_cosine_max": float(np.max(cos)),
            "sqrt_k_over_d": float(np.sqrt((args.k + 1) / d)),
        })
        print(json.dumps(out["seeds"][-1], indent=1), flush=True)

    # P4: committee sizing
    table = []
    for n in (20, 50, 100, 200, 500, 1000):
        for frac in (0.05, 0.1, 0.2, 0.3):
            f = int(round(frac * n))
            row = {"n": n, "f": f, "byz_frac": frac}
            # guaranteed output: m = 4t+1 (exact norms) / 3t+1 (sketch);
            # security with abort (sketch): m = 2t+1, and the adversary can force up
            # to R = 3 retries per round, so the target applies to 4 * P[c > t]
            for mode, a, amp in (("exact", 4, 1), ("sketch", 3, 1), ("abort", 2, 4)):
                for target in (1e-3, 1e-6):
                    best = None
                    for t in range(1, n):
                        m = a * t + 1
                        if m > n:
                            break
                        if amp * capture_probability(n, f, m, t) <= target:
                            best = (m, t)
                            break
                    row[f"{mode}_{target:g}"] = best
            table.append(row)
    out["committee_sizing"] = table
    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"privacy_{args.dataset}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
