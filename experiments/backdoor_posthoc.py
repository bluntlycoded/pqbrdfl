"""Post-hoc, agent-side backdoor defences on top of the aggregation defence.

For every honest agent we apply the defence using only that agent's local data,
then evaluate on the shared test set; results are averaged over honest agents.

    python -m experiments.backdoor_posthoc
"""

from __future__ import annotations

import itertools
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from pqbrdfl.fl import attacks as A, data, sim  # noqa: E402
from pqbrdfl.fl.posthoc import AnyGate, BenignGate, LocalSecondOpinion, gated_predict  # noqa: E402


def evaluate_agent(model, w, ds, trig, gate):
    atk_true = ds.y_test != ds.normal_class
    pred_atk = gated_predict(model, w, ds.X_test, ds.normal_class, gate)
    Xa = A.add_trigger(ds.X_test[atk_true], trig)
    asr = float((~gated_predict(model, w, Xa, ds.normal_class, gate)).mean())
    acc_bin = float((pred_atk == atk_true).mean())
    fpr = float((pred_atk & ~atk_true).sum() / max((~atk_true).sum(), 1))
    rec = float((pred_atk & atk_true).sum() / max(atk_true.sum(), 1))
    return {"asr": asr, "binary_acc": acc_bin, "fpr": fpr, "recall": rec}


def job(args):
    ds_name, agg, attack, trigger, seed = args
    ds = data.load(ds_name)
    cfg = sim.SimConfig(dataset=ds_name, aggregator=agg, attack=attack, trigger=trigger, byz_frac=0.2,
                        rounds=40, eval_every=40, lr=3e-3, seed=seed, keep_weights=True)
    r = sim.run(cfg, ds)
    model, w, byz, local, trig = r["_model"], r["_weights"], r["_byz"], r["_local"], r["_trigger"]
    rng = np.random.default_rng(seed)
    honest = [a for a in local if a not in byz]
    TARGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
    res = {k: [] for k in ["none", "gate"] + [f"local{int(t*100)}+gate" for t in TARGETS]}
    for a in honest:
        X, y = local[a]
        Xb = X[y == ds.normal_class]
        ok = len(Xb) > 2 * X.shape[1]
        g99 = BenignGate(Xb, 0.99) if ok else None
        base = LocalSecondOpinion.calibrated(model, X, y, ds.normal_class, target_fpr=0.01, seed=seed)
        res["none"].append(evaluate_agent(model, w, ds, trig, None))
        res["gate"].append(evaluate_agent(model, w, ds, trig, g99))
        for t in TARGETS:
            lc = None
            if base is not None:
                lc = LocalSecondOpinion.__new__(LocalSecondOpinion)
                lc.model, lc.normal, lc.w = base.model, base.normal, base.w
                # recalibrate on the same held-out benign flows
                rng2 = np.random.default_rng(seed)
                idx = rng2.permutation(len(y)); ho = idx[int(0.8 * len(y)):]
                hb = ho[y[ho] == ds.normal_class]
                sc = 1.0 - model.predict_proba(base.w, X[hb])[:, ds.normal_class]
                lc.conf = float(min(1.0, np.quantile(sc, 1.0 - t) + 1e-9))
            gs = [g for g in (lc, g99) if g is not None]
            res[f"local{int(t*100)}+gate"].append(evaluate_agent(model, w, ds, trig, AnyGate(*gs) if gs else None))
    summary = {k: {m: float(np.mean([v[m] for v in vs])) for m in vs[0]} for k, vs in res.items()}
    out = {"dataset": ds_name, "aggregator": agg, "attack": attack, "trigger": trigger, "seed": seed,
           "multiclass_acc": r["final"]["accuracy"], "defences": summary}
    p = Path("results/backdoor_posthoc") / f"{ds_name}__{agg}__{attack}__{trigger}__seed={seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out


def main():
    import sys
    agg = sys.argv[1] if len(sys.argv) > 1 else "pqbrdfl"
    jobs = list(itertools.product(["insdn", "cicids"], [agg],
                                  [("none", "oor"), ("backdoor", "oor"), ("backdoor", "inrange")], [0, 1, 2]))
    jobs = [(d, g, a, t, s) for d, g, (a, t), s in jobs]
    with ProcessPoolExecutor(4) as ex:
        for o in ex.map(job, jobs):
            line = " | ".join(f"{k}: asr={v['asr']:.3f} fpr={v['fpr']:.3f} rec={v['recall']:.3f}"
                              for k, v in o["defences"].items())
            print(f"{o['dataset']:6s} {o['attack']:8s} {o['trigger']:7s} s{o['seed']} | {line}", flush=True)


if __name__ == "__main__":
    main()
