"""Run the evaluation of the manuscript (Section 7).

    python -m experiments.run_experiments --exp all --dataset cicids --seeds 0 1 2
    python -m experiments.run_experiments --exp byzantine --dataset insdn --data-root data/InSDN

Every run writes one JSON file under results/<dataset>/<exp>/.  Tables are
built from those files by experiments/make_tables.py, so nothing in the paper
is typed by hand.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process: the workers already parallelise across
# runs, and nested thread pools oversubscribe the CPU badly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from pqbrdfl.fl import sim  # noqa: E402
from pqbrdfl.fl.data import load  # noqa: E402

RESULTS = Path("results")
_DS_CACHE = {}

# N = 20 agents, f <= 4 Byzantine.  PQ-BRDFL runs with sketch norms and the
# security-with-abort policy, so an honest-majority committee suffices:
# m = 9, t = 4 >= f, and no round can be captured.  Sortition at larger N is
# exercised separately in the "committee" experiment.
BASE = dict(n_agents=20, alpha=0.5, rounds=40, local_epochs=1, lr=3e-3, eval_every=10, m=9, t=4,
            norm_mode="sketch", decode_policy="abort", k_proj=128)


def _ds(name, root, seed):
    key = (name, root, seed)
    if key not in _DS_CACHE:
        _DS_CACHE[key] = load(name, root, seed=seed)
    return _DS_CACHE[key]


def _job(cfg_dict, out_path):
    import warnings as _w
    _w.filterwarnings("ignore")
    cfg = sim.SimConfig(**cfg_dict)
    if Path(out_path).exists():
        return out_path, "cached"
    ds = _ds(cfg.dataset, cfg.data_root, 0)       # fixed train/test split across seeds
    res = sim.run(cfg, ds)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sim.save(res, out_path)
    return out_path, f"acc={res['final']['accuracy']:.4f} {res['wall_seconds']:.0f}s"


def _name(cfg: dict, keys) -> str:
    return "__".join(f"{k}={cfg[k]}" for k in keys) + ".json"


def grid(exp, dataset, root, seeds):
    base = dict(BASE, dataset=dataset, data_root=root)
    jobs = []

    def add(keys, **kw):
        cfg = dict(base, **kw)
        jobs.append((cfg, str(RESULTS / dataset / exp / _name(cfg, keys))))

    ATT = ["gaussian", "signflip", "labelflip", "backdoor", "scaled_backdoor", "alie", "ipm"]
    if exp == "detection":
        for agg, s in itertools.product(["central", "fedavg", "dfl_gossip", "cc", "mk_cc", "pqbrdfl"], seeds):
            add(["aggregator", "seed"], aggregator=agg, attack="none", byz_frac=0.0, seed=s)
    elif exp == "byzantine":
        aggs = ["fedavg", "dfl_gossip", "median", "trimmed_mean", "krum", "fltrust", "cc", "mk_cc",
                "pqbrdfl_mean", "pqbrdfl_cc", "pqbrdfl"]
        for agg, atk, s in itertools.product(aggs, ATT, seeds):
            add(["aggregator", "attack", "seed"], aggregator=agg, attack=atk, byz_frac=0.2, seed=s)
    elif exp == "byzfrac":
        for agg, atk, bf, s in itertools.product(["fedavg", "median", "cc", "mk_cc", "pqbrdfl"],
                                                 ["signflip", "scaled_backdoor"], [0.1, 0.2, 0.3], seeds):
            kw = {}
            if agg == "pqbrdfl" and bf > 0.2:
                kw = dict(m=13, t=6)       # f = 6: honest-majority committee with t >= f
            add(["aggregator", "attack", "byz_frac", "seed"], aggregator=agg, attack=atk,
                byz_frac=bf, seed=s, **kw)
    elif exp == "ablation":
        # A/B FedAvg (classical vs PQ channel), C secret-shared mean, D plaintext CC,
        # E shares + defence without range check, F shares + CC only, G full PQ-BRDFL
        for agg, atk, s in itertools.product(
                ["pqbrdfl_mean", "pqbrdfl_cc_norange", "pqbrdfl_norange", "pqbrdfl_cc", "pqbrdfl"],
                ["signflip", "scaled_backdoor"], seeds):
            add(["aggregator", "attack", "protocol_attack", "seed"], aggregator=agg, attack=atk,
                protocol_attack="wrap", byz_frac=0.2, seed=s)
    elif exp == "wrapdemo":
        # the wrap-around attack against exact (degree-2t) norms: with and without the range check
        for agg, s in itertools.product(["pqbrdfl_cc_norange", "pqbrdfl_cc"], seeds):
            add(["aggregator", "norm_mode", "seed"], aggregator=agg, attack="signflip", protocol_attack="wrap",
                byz_frac=0.2, norm_mode="exact", decode_policy="correct", m=17, t=4, k_proj=64, seed=s)
    elif exp == "telemetry":
        for agg, tel, tf, s in itertools.product(["fedavg", "pqbrdfl"], ["none", "tamper", "tamper_signed"],
                                                 [0.3, 0.6], seeds):
            if tel == "none" and tf == 0.6:
                continue
            add(["aggregator", "telemetry", "tamper_frac", "seed"], aggregator=agg, attack="none",
                byz_frac=0.0, telemetry=tel, tamper_frac=tf, seed=s)
    elif exp == "committee":
        # N = 60, sampled committees: guaranteed-output decoding (m >= 3t+1 / 4t+1)
        # versus security-with-abort (m >= 2t+1) with retries and bans
        for ma, bf, (mode, pol, m, t), s in itertools.product(
                ["corrupt_agg", "corrupt_scalars"], [0.1, 0.2],
                [("exact", "correct", 21, 5), ("sketch", "correct", 22, 7),
                 ("sketch", "abort", 13, 6), ("sketch", "abort", 17, 8)], seeds[:2]):
            add(["member_attack", "byz_frac", "norm_mode", "decode_policy", "m", "seed"], aggregator="pqbrdfl",
                attack="signflip", member_attack=ma, byz_frac=bf, norm_mode=mode, decode_policy=pol,
                m=m, t=t, k_proj=64 if mode == "exact" else 128, n_agents=60, rounds=20, seed=s)
    else:
        raise ValueError(exp)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--dataset", default="cicids")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    exps = ["detection", "byzantine", "byzfrac", "ablation", "telemetry", "committee"] \
        if args.exp == "all" else args.exp.split(",")
    if args.exp == "all" and args.dataset != "insdn":
        # capture / abort / ban statistics depend on the protocol, not the data:
        # the N = 60 committee experiment is run once, on InSDN
        exps.remove("committee")
    jobs = [j for e in exps for j in grid(e, args.dataset, args.data_root, args.seeds)]
    print(f"{len(jobs)} runs, {args.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_job, c, p) for c, p in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            path, msg = f.result()
            print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {Path(path).name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
