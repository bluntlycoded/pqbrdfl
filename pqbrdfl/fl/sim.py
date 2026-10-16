"""Simulation driver: SDN agents training a shared IDS under Byzantine attack.

Aggregators
-----------
central         pooled training on all honest data (upper reference, no FL)
fedavg          plaintext mean of updates through a server
dfl_gossip      serverless gossip averaging on a random 4-regular graph
median, trimmed_mean, krum, cc, fltrust
                plaintext robust rules (the defence sees every update)
pqbrdfl         THIS WORK: PQ channel + secret-shared committee + range check
                + centred clipping, run with real cryptography every round
pqbrdfl_mean    same protocol without the Byzantine defence (secure aggregation only)
pqbrdfl_norange same protocol without the range check (to expose the wrap attack)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..agg import protocol as P
from ..agg import robust
from ..crypto.backend import get_backend
from ..crypto import channel as ch
from . import attacks as A
from . import telemetry as T
from .data import Dataset, load, partition
from .model import MLP

PLAIN_RULES = ("fedavg", "median", "trimmed_mean", "krum", "cc", "fltrust", "cc_sf", "mk_cc", "mk_cc_exact")
PROTOCOL_RULES = ("pqbrdfl", "pqbrdfl_cc", "pqbrdfl_mean", "pqbrdfl_norange", "pqbrdfl_cc_norange")


@dataclass
class SimConfig:
    dataset: str = "synthetic"
    data_root: str | None = None
    n_agents: int = 20
    alpha: float = 0.5
    byz_frac: float = 0.2
    attack: str = "none"
    protocol_attack: str | None = None     # e.g. "wrap": Byzantine clients also attack the encoding
    member_attack: str = P.M_CORRUPT_AGG   # Byzantine agents' behaviour when on the committee
    aggregator: str = "pqbrdfl"
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 1e-3
    batch: int = 128
    seed: int = 0
    m: int = 9
    t: int = 4
    norm_mode: str = "sketch"
    decode_policy: str = "abort"
    k_proj: int = 128
    backend: str = "fips"
    telemetry: str = "none"                # none | tamper | tamper_signed
    tamper_frac: float = 0.3               # share of honest agents behind a tampered path
    eval_every: int = 5
    max_train_per_agent: int | None = 4000
    sf_kappa: float = 2.0
    mk_q: float = 0.5                      # fraction of clients multi-Krum keeps
    # --- backdoor-defence options (see experiments/backdoor_defences.py) ---
    worker_momentum: float = 0.0           # agents send an EMA of their updates (beta)
    byz_momentum: bool = True              # Byzantine agents also apply the EMA
    agg_noise: float = 0.0                 # weak-DP Gaussian noise std added to each aggregate
    input_clip: bool = False               # IDS clamps inputs to the benign training range
    trigger: str = "oor"                   # "oor" (-4, out of range) | "inrange" (adaptive)
    keep_weights: bool = False             # return final weights in the result (not saved)                  # sketch-filter outlier threshold (cc_sf / pqbrdfl_sf)


def evaluate(model: MLP, w: np.ndarray, ds: Dataset, trigger_value=None, clip=None) -> dict:
    Xt = ds.X_test if clip is None else np.clip(ds.X_test, clip[0], clip[1])
    proba = model.predict_proba(w, Xt)
    pred = proba.argmax(1)
    y = ds.y_test
    atk_true = y != ds.normal_class
    atk_pred = pred != ds.normal_class
    tp = int((atk_true & atk_pred).sum()); fp = int((~atk_true & atk_pred).sum())
    fn = int((atk_true & ~atk_pred).sum()); tn = int((~atk_true & ~atk_pred).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    score = 1.0 - proba[:, ds.normal_class]
    try:
        auc = float(roc_auc_score(atk_true, score))
    except ValueError:
        auc = float("nan")
    # backdoor attack success: triggered attack flows classified as Normal
    Xa = A.add_trigger(ds.X_test[atk_true], trigger_value)
    if clip is not None:
        Xa = np.clip(Xa, clip[0], clip[1])
    asr = float((model.predict_proba(w, Xa).argmax(1) == ds.normal_class).mean()) if len(Xa) else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "fpr": fp / max(fp + tn, 1),
        "auc": auc,
        "backdoor_asr": asr,
    }


def _gossip_matrix(n: int, rng) -> np.ndarray:
    """Metropolis weights on a random 4-regular-ish graph (ring + chords)."""
    adj = np.zeros((n, n), bool)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = True
    perm = rng.permutation(n)
    for a, b in zip(perm[::2], perm[1::2]):
        adj[a, b] = adj[b, a] = True
    deg = adj.sum(1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in np.nonzero(adj[i])[0]:
            W[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def run(cfg: SimConfig, ds: Dataset | None = None, verbose: bool = False) -> dict:
    rng = np.random.default_rng(cfg.seed)
    ds = ds or load(cfg.dataset, cfg.data_root, seed=cfg.seed)
    ids = [f"sw{i:02d}" for i in range(cfg.n_agents)]
    n_byz = int(round(cfg.byz_frac * cfg.n_agents))
    byz = set(rng.choice(ids, size=n_byz, replace=False).tolist()) if n_byz else set()
    honest = [a for a in ids if a not in byz]

    # hold out a small clean root set (only FLTrust uses it)
    trig = A.inrange_trigger(ds.X_train) if cfg.trigger == "inrange" else None
    # input clamp: the benign range of the training data (each agent would use its own
    # local benign flows; the pooled range is the most permissive choice)
    clip = None
    if cfg.input_clip:
        Xb = ds.X_train[ds.y_train == ds.normal_class]
        clip = (np.quantile(Xb, 0.001, axis=0), np.quantile(Xb, 0.999, axis=0))
        clip = (np.minimum(clip[0], np.quantile(ds.X_train, 0.001, axis=0)),
                np.maximum(clip[1], np.quantile(ds.X_train, 0.999, axis=0)))
    root_idx = rng.choice(len(ds.y_train), size=min(500, len(ds.y_train) // 20), replace=False)
    pool = np.setdiff1d(np.arange(len(ds.y_train)), root_idx)
    parts = partition(ds.y_train[pool], cfg.n_agents, cfg.alpha, rng)
    local = {}
    for a, p in zip(ids, parts):
        p = pool[p]
        if cfg.max_train_per_agent:
            p = p[: cfg.max_train_per_agent]
        X, y = ds.X_train[p], ds.y_train[p]
        if a in byz and cfg.attack == "labelflip":
            y = A.flip_labels(y, ds.normal_class, rng)
        if a in byz and cfg.attack in ("backdoor", "scaled_backdoor"):
            X, y = A.backdoor_data(X, y, ds.normal_class, rng, value=trig)
        local[a] = (X, y)

    # telemetry: tampered paths (honest agents whose switch reports are modified in transit)
    tele_stats = {"batches": 0, "rejected": 0}
    if cfg.telemetry != "none":
        B = get_backend(cfg.backend)
        normal_mean = ds.X_train[ds.y_train == ds.normal_class].mean(0)
        victims = set(rng.choice(honest, size=int(round(cfg.tamper_frac * len(honest))), replace=False).tolist())
        for a in ids:
            X, y = local[a]
            sw = ch.Identity.generate(B, a + "-switch")
            keep_X, keep_y = [], []
            for wdw, s in enumerate(range(0, len(y), 500)):
                b = T.export_batch(B, sw, wdw, X[s:s + 500], y[s:s + 500])
                if a in victims:
                    b = T.tamper_in_path(b, ds.normal_class, rng, normal_mean)
                tele_stats["batches"] += 1
                if cfg.telemetry == "tamper_signed" and not T.verify_batch(B, sw.pk, b):
                    tele_stats["rejected"] += 1
                    continue
                keep_X.append(b.X); keep_y.append(b.y)
            if keep_y:
                local[a] = (np.concatenate(keep_X), np.concatenate(keep_y))

    model = MLP(ds.n_features, ds.n_classes, seed=cfg.seed)
    w = model.w.copy()
    history = []
    center = np.zeros(model.dim)
    net = None
    proto_log = {"bytes": [], "time": [], "rejected": [], "flagged": [], "captured": 0, "breach": 0,
                 "aborted": 0, "attempts": []}
    if cfg.aggregator in PROTOCOL_RULES:
        robust_rule = {"pqbrdfl": "mkcc", "pqbrdfl_cc": "cc", "pqbrdfl_mean": "mean",
                  "pqbrdfl_norange": "mkcc", "pqbrdfl_cc_norange": "cc"}[cfg.aggregator]
        acfg = P.AggConfig(m=cfg.m, t=cfg.t, backend=cfg.backend, norm_mode=cfg.norm_mode,
                           k_proj=cfg.k_proj, robust=robust_rule, mk_q=cfg.mk_q,
                           krum_f=None, decode_policy=cfg.decode_policy,
                           range_check=not cfg.aggregator.endswith("_norange"))
        net = P.AggregationNetwork(ids, acfg)
    Wg = _gossip_matrix(cfg.n_agents, rng) if cfg.aggregator == "dfl_gossip" else None
    agent_w = {a: w.copy() for a in ids} if Wg is not None else None
    Xc = np.concatenate([local[a][0] for a in honest])
    yc = np.concatenate([local[a][1] for a in honest])
    mom: dict = {}
    t_start = time.perf_counter()

    for r in range(1, cfg.rounds + 1):
        if cfg.aggregator == "central":
            w = model.train_local(w, Xc, yc, epochs=cfg.local_epochs, lr=cfg.lr, batch=cfg.batch, rng=rng)
        elif cfg.aggregator == "dfl_gossip":
            new = {}
            for a in ids:
                X, y = local[a]
                new[a] = model.train_local(agent_w[a], X, y, cfg.local_epochs, cfg.lr, cfg.batch, rng)
            deltas = {a: new[a] - agent_w[a] for a in ids}
            hon = np.stack([deltas[a] for a in honest])
            bad = A.poison_updates(cfg.attack, {a: deltas[a] for a in byz}, hon, cfg.n_agents, rng)
            sent = {a: agent_w[a] + (bad[a] if a in byz else deltas[a]) for a in ids}
            Ms = np.stack([sent[a] for a in ids])
            mixed = Wg @ Ms
            agent_w = {a: mixed[k] for k, a in enumerate(ids)}
            w = np.mean([agent_w[a] for a in honest], axis=0)
        else:
            deltas = {}
            for a in ids:
                X, y = local[a]
                if clip is not None:
                    X = np.clip(X, clip[0], clip[1])
                deltas[a] = model.train_local(w, X, y, cfg.local_epochs, cfg.lr, cfg.batch, rng) - w
            if cfg.worker_momentum > 0:
                beta = cfg.worker_momentum
                for a in ids:
                    if a in byz and not cfg.byz_momentum:
                        continue
                    mom[a] = beta * mom.get(a, deltas[a]) + (1 - beta) * deltas[a]
                    deltas[a] = mom[a]
            hon = np.stack([deltas[a] for a in honest])
            bad = A.poison_updates(cfg.attack, {a: deltas[a] for a in byz}, hon, cfg.n_agents, rng)
            deltas.update(bad)
            if cfg.aggregator in PLAIN_RULES:
                X_all = np.stack([deltas[a] for a in ids])
                kw = {"f": n_byz, "center": center}
                if cfg.aggregator in ("cc_sf", "mk_cc"):
                    kw["R"] = rng.random((64, model.dim)) < 0.5      # fresh public challenge
                    kw["kappa"] = cfg.sf_kappa
                if cfg.aggregator.startswith("mk_cc"):
                    kw.pop("kappa", None)
                    kw["q_frac"] = cfg.mk_q
                if cfg.aggregator == "fltrust":
                    Xr, yr = ds.X_train[root_idx], ds.y_train[root_idx]
                    kw["root_update"] = model.train_local(w, Xr, yr, cfg.local_epochs, cfg.lr, cfg.batch, rng) - w
                agg = robust.RULES[cfg.aggregator](X_all, **kw)
            else:
                cb = {a: cfg.protocol_attack for a in byz} if cfg.protocol_attack else {}
                mb = {a: cfg.member_attack for a in byz}
                res = P.run_round_with_retries(net, r, deltas, center, client_behavior=cb,
                                               member_behavior=mb, rng=rng)
                proto_log["bytes"].append(res.bytes_by_kind)
                proto_log["time"].append(res.time_by_phase)
                proto_log["rejected"].append(res.rejected)
                proto_log["flagged"].append(sorted(res.flagged_members))
                proto_log["captured"] += int(res.captured)
                proto_log["breach"] += int(res.privacy_breach)
                proto_log["attempts"].append(res.attempts)
                proto_log.setdefault("banned", []).append(res.banned)
                if res.aggregate is None:
                    proto_log["aborted"] += 1
                    agg = np.zeros_like(w)
                else:
                    agg = res.aggregate
            if not np.all(np.isfinite(agg)):
                agg = np.zeros_like(w)
            if cfg.agg_noise > 0:
                agg = agg + rng.normal(0.0, cfg.agg_noise, agg.shape)
            w = w + agg
            center = agg
        if r % cfg.eval_every == 0 or r == cfg.rounds:
            if not np.all(np.isfinite(w)):
                ev = {"accuracy": 0.0, "macro_f1": 0.0, "f1": 0.0, "auc": float("nan"),
                      "precision": 0.0, "recall": 0.0, "fpr": 1.0, "backdoor_asr": 1.0}
            else:
                ev = evaluate(model, w, ds, trig, clip)
            ev["round"] = r
            history.append(ev)
            if verbose:
                print(f"[{cfg.aggregator}/{cfg.attack}] round {r}: acc={ev['accuracy']:.4f} "
                      f"f1={ev['f1']:.4f} fpr={ev['fpr']:.4f} asr={ev['backdoor_asr']:.3f}", flush=True)

    out = {
        "config": asdict(cfg),
        "dataset": ds.name, "synthetic_data": ds.synthetic,
        "model_params": model.dim,
        "byzantine": sorted(byz),
        "final": history[-1],
        "history": history,
        "wall_seconds": time.perf_counter() - t_start,
        "telemetry": tele_stats,
    }
    if net is not None:
        out["protocol"] = proto_log
    if cfg.keep_weights:
        out["_weights"], out["_model"], out["_byz"], out["_local"] = w, model, byz, local
        out["_trigger"], out["_clip"] = trig, clip
    return out


def save(result: dict, path: str) -> None:
    result = {k: v for k, v in result.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
