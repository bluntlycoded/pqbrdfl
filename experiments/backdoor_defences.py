"""Screening of defences against the *unscaled* backdoor (plaintext, fast).

Every configuration is a plaintext emulation of something the committee can
compute without extra leakage (selection on public sketches, worker momentum,
jointly generated aggregate noise) or something each agent does locally (input
clamping).  Winners are then run through the full protocol.

    python -m experiments.backdoor_defences --seeds 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

from pqbrdfl.fl import data, sim  # noqa: E402

CONFIGS = {
    "cc":              dict(aggregator="cc"),
    "mk":              dict(aggregator="mk_cc"),
    "mk_q30":          dict(aggregator="mk_cc", mk_q=0.3),
    "mk_mom":          dict(aggregator="mk_cc", worker_momentum=0.9),
    "mk_mom_byzraw":   dict(aggregator="mk_cc", worker_momentum=0.9, byz_momentum=False),
    "mk_noise1e-3":    dict(aggregator="mk_cc", agg_noise=1e-3),
    "mk_noise3e-3":    dict(aggregator="mk_cc", agg_noise=3e-3),
    "mk_clip":         dict(aggregator="mk_cc", input_clip=True),
    "mk_mom_clip":     dict(aggregator="mk_cc", worker_momentum=0.9, input_clip=True),
}
THREATS = {
    "none":            dict(attack="none", byz_frac=0.2),
    "backdoor":        dict(attack="backdoor", byz_frac=0.2),
    "backdoor_inrange": dict(attack="backdoor", trigger="inrange", byz_frac=0.2),
}


def job(args):
    ds_name, cname, tname, seed, rounds = args
    out = Path("results/backdoor_defences") / f"{ds_name}__{cname}__{tname}__seed={seed}.json"
    if out.exists():
        r = json.loads(out.read_text())
    else:
        ds = data.load(ds_name)
        cfg = sim.SimConfig(dataset=ds_name, rounds=rounds, eval_every=rounds, lr=3e-3, seed=seed,
                            **CONFIGS[cname], **THREATS[tname])
        r = sim.run(cfg, ds)
        out.parent.mkdir(parents=True, exist_ok=True)
        sim.save(r, str(out))
    return ds_name, cname, tname, seed, r["final"]["accuracy"], r["final"]["backdoor_asr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = list(itertools.product(["insdn", "cicids"], a.configs, list(THREATS), a.seeds, [a.rounds]))
    with ProcessPoolExecutor(a.workers) as ex:
        for ds, c, t, s, acc, asr in ex.map(job, jobs):
            print(f"{ds:7s} {c:15s} {t:17s} seed={s} acc={acc:.4f} asr={asr:.3f}", flush=True)


if __name__ == "__main__":
    main()
