"""Backdoor-defence exploration (plaintext, 1 seed, 30 rounds) that led to the final design.

Recorded negative result: a sketch-cosine outlier filter (drop clients whose median
cosine to the others is a robust outlier) did NOT stop backdoors (90-100% ASR).
Multi-Krum selection on the same public sketches did (InSDN: 0% ASR), and matched
exact multi-Krum, so it became the PQ-BRDFL defence.

    python -m experiments.backdoor_exploration
"""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from pqbrdfl.fl import sim, data
from concurrent.futures import ProcessPoolExecutor
def job(a):
    name, agg, atk, kappa = a
    ds = data.load(name)
    r = sim.run(sim.SimConfig(dataset=name, aggregator=agg, attack=atk, rounds=30, eval_every=30, lr=3e-3, byz_frac=0.2, seed=0, mk_q=kappa if kappa else 0.5), ds)
    f = r["final"]; return a, f["accuracy"], f["backdoor_asr"]
def main():
    jobs=[(n,a,t,k) for n in ["insdn","cicids"] for t in ["none","backdoor","scaled_backdoor","signflip","ipm","alie"] for (a,k) in [("mk_cc",0.5),("mk_cc",0.3),("mk_cc_exact",0.5)]]
    with ProcessPoolExecutor(6) as ex:
        for a, acc, asr in ex.map(job, jobs):
            print(a, "acc %.4f asr %.3f" % (acc, asr), flush=True)

if __name__ == "__main__":
    main()
