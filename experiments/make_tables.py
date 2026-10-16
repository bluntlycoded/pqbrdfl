"""Aggregate results/<dataset>/<exp>/*.json into the manuscript's tables.

    python -m experiments.make_tables --dataset cicids > results/tables_cicids.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pqbrdfl.agg.committee import capture_probability

AGG_LABEL = {
    "central": "Centralized (pooled)", "fedavg": "FedAvg", "dfl_gossip": "Gossip DFL",
    "median": "Coord. median", "trimmed_mean": "Trimmed mean", "krum": "Krum",
    "fltrust": "FLTrust (root data)", "cc": "Centered clipping (plaintext)",
    "pqbrdfl_mean": "PQ secure agg., no defense", "pqbrdfl_norange": "PQ-BRDFL w/o range check",
    "pqbrdfl_cc": "PQ-BRDFL, CC only", "pqbrdfl_cc_norange": "PQ-BRDFL, CC only, no range check", "mk_cc": "Multi-Krum + CC (plaintext)",
    "pqbrdfl": "**PQ-BRDFL (ours)**",
}


def load(ds, exp):
    runs = []
    for p in sorted(Path("results", ds, exp).glob("*.json")):
        runs.append(json.loads(p.read_text()))
    return runs


def ms(vals, pct=True, digits=2):
    v = np.array(vals, float)
    if pct:
        v = 100 * v
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def group(runs, keys):
    g = defaultdict(list)
    for r in runs:
        g[tuple(r["config"][k] for k in keys)].append(r)
    return g


def table_detection(ds):
    runs = load(ds, "detection")
    if not runs:
        return ""
    g = group(runs, ["aggregator"])
    out = ["| Method | Accuracy | Macro-F1 | Attack F1 | FPR | AUC |", "|---|---|---|---|---|---|"]
    for agg in ["central", "fedavg", "dfl_gossip", "cc", "mk_cc", "pqbrdfl"]:
        rs = g.get((agg,), [])
        if not rs:
            continue
        f = [r["final"] for r in rs]
        out.append(f"| {AGG_LABEL[agg]} | {ms([x['accuracy'] for x in f])} | {ms([x['macro_f1'] for x in f])} | "
                   f"{ms([x['f1'] for x in f])} | {ms([x['fpr'] for x in f])} | {ms([x['auc'] for x in f], digits=2)} |")
    return "\n".join(out)


ATTACKS = ["gaussian", "signflip", "labelflip", "backdoor", "scaled_backdoor", "alie", "ipm"]


def table_byzantine(ds, metric="accuracy"):
    runs = load(ds, "byzantine")
    if not runs:
        return ""
    g = group(runs, ["aggregator", "attack"])
    hdr = "| Method | " + " | ".join(ATTACKS) + " |"
    out = [hdr, "|---" * (len(ATTACKS) + 1) + "|"]
    for agg in ["fedavg", "dfl_gossip", "median", "trimmed_mean", "krum", "fltrust", "cc", "mk_cc",
                "pqbrdfl_mean", "pqbrdfl_cc", "pqbrdfl"]:
        cells = []
        for atk in ATTACKS:
            rs = g.get((agg, atk), [])
            cells.append(ms([r["final"][metric] for r in rs]) if rs else "—")
        out.append(f"| {AGG_LABEL[agg]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_byzfrac(ds):
    runs = load(ds, "byzfrac")
    if not runs:
        return ""
    g = group(runs, ["aggregator", "attack", "byz_frac"])
    fr = [0.1, 0.2, 0.3]
    out = ["| Method | Attack | " + " | ".join(f"f/N = {x}" for x in fr) + " |", "|---|---|---|---|---|"]
    out[0] = "| Method | Attack | Metric | " + " | ".join(f"f/N = {x}" for x in fr) + " |"
    out[1] = "|---|---|---|---|---|---|"
    for agg in ["fedavg", "median", "cc", "mk_cc", "pqbrdfl"]:
        for atk in ["signflip", "scaled_backdoor"]:
            met = "accuracy" if atk == "signflip" else "backdoor_asr"
            cells = [ms([r["final"][met] for r in g.get((agg, atk, x), [])]) for x in fr]
            out.append(f"| {AGG_LABEL[agg]} | {atk} | {'Acc.' if met == 'accuracy' else 'ASR'} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n\n(PQ-BRDFL at f/N = 0.3 uses m = 13, t = 6.)"


def table_ablation(ds):
    byz = group(load(ds, "byzantine"), ["aggregator", "attack"])
    g = group(load(ds, "ablation"), ["aggregator", "attack"])
    if not byz:
        return ""
    rows = [
        ("A", "No PQC: FedAvg, classical channel", "fedavg", None, "no", "no", "no", "n/a"),
        ("B", "PQC channel only (FedAvg)", "fedavg", None, "yes", "no", "no", "n/a"),
        ("C", "PQC + secret-shared mean", "pqbrdfl_mean", "pqbrdfl_mean", "yes", "yes", "no", "yes"),
        ("D", "PQC + plaintext multi-Krum + CC", "mk_cc", None, "yes", "no", "yes", "n/a"),
        ("E1", "PQC + shares + CC, no range check", None, "pqbrdfl_cc_norange", "yes", "yes", "partial", "no"),
        ("E2", "PQC + shares + MK + CC, no range check", None, "pqbrdfl_norange", "yes", "yes", "yes", "empirical"),
        ("F", "PQC + shares + CC only", "pqbrdfl_cc", "pqbrdfl_cc", "yes", "yes", "partial", "yes"),
        ("G", "Full PQ-BRDFL", "pqbrdfl", "pqbrdfl", "yes", "yes", "yes", "yes"),
    ]
    out = ["| Cfg | Configuration | HNDL-safe | Updates hidden | Byz. defense | Wrap-safe | "
           "Acc. sign-flip | ASR scaled backdoor | Acc. sign-flip + wrap |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, desc, a_plain, a_wrap, pq, hid, bd, wrap in rows:
        c1 = ms([r["final"]["accuracy"] for r in byz.get((a_plain, "signflip"), [])]) if a_plain else "(= G)"
        c2 = ms([r["final"]["backdoor_asr"] for r in byz.get((a_plain, "scaled_backdoor"), [])]) if a_plain else "(= G)"
        c3 = ms([r["final"]["accuracy"] for r in g.get((a_wrap, "signflip"), [])]) if a_wrap else "n/a"
        out.append(f"| {tag} | {desc} | {pq} | {hid} | {bd} | {wrap} | {c1} | {c2} | {c3} |")
    return "\n".join(out)


def table_wrapdemo(ds):
    g = group(load(ds, "wrapdemo"), ["aggregator"])
    if not g:
        return ""
    out = ["| Exact-norm protocol | Accuracy | Wrap vectors rejected |",
           "|---|---|---|"]
    for agg, lab in (("pqbrdfl_cc_norange", "no range check"), ("pqbrdfl_cc", "with range check")):
        rs = g.get((agg,), [])
        rej = [np.mean([sum(1 for v in rr.values() if v == "range") for rr in r["protocol"]["rejected"]]) for r in rs]
        out.append(f"| {lab} | {ms([r['final']['accuracy'] for r in rs])} | {ms(rej, pct=False, digits=1)} per round |")
    return "\n".join(out)


def table_backdoor_defence(agg="pqbrdfl"):
    """Agent-side defense in depth against the unscaled backdoor (results/backdoor_posthoc)."""
    runs = [json.loads(p.read_text()) for p in Path("results/backdoor_posthoc").glob(f"*__{agg}__*.json")]
    if not runs:
        return ""
    g = defaultdict(list)
    for r in runs:
        g[(r["dataset"], r["attack"], r["trigger"])].append(r)
    cols = [("none", "Global model"), ("gate", "+ benign gate"), ("local1+gate", "+ gate + local (1%)")]
    out = ["| Dataset | Threat | " + " | ".join(f"{c} ASR | {c} FPR" for _, c in cols) + " |",
           "|---|---|" + "---|---|" * len(cols)]
    names = {("backdoor", "oor"): "backdoor", ("backdoor", "inrange"): "backdoor, in-range trigger",
             ("none", "oor"): "no attack (natural rate)"}
    for ds in ("insdn", "cicids"):
        for key in (("backdoor", "oor"), ("backdoor", "inrange"), ("none", "oor")):
            rs = g.get((ds, *key), [])
            if not rs:
                continue
            cells = []
            for d, _ in cols:
                cells += [ms([r["defences"][d]["asr"] for r in rs]), ms([r["defences"][d]["fpr"] for r in rs])]
            out.append(f"| {'InSDN' if ds == 'insdn' else 'CICIDS2017'} | {names[key]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_telemetry(ds):
    runs = load(ds, "telemetry")
    if not runs:
        return ""
    g = group(runs, ["aggregator", "telemetry", "tamper_frac"])
    out = ["| Method | Telemetry | Tampered paths | Attack recall | Attack F1 | FPR | Batches rejected |",
           "|---|---|---|---|---|---|---|"]
    for agg in ["fedavg", "pqbrdfl"]:
        for tel in ["none", "tamper", "tamper_signed"]:
            for tf in [0.3, 0.6]:
                rs = g.get((agg, tel, tf), [])
                if not rs or (tel == "none" and tf == 0.6):
                    continue
                rej = [r["telemetry"]["rejected"] / max(r["telemetry"]["batches"], 1) for r in rs]
                out.append(f"| {AGG_LABEL[agg]} | {tel} | {'—' if tel == 'none' else f'{int(tf*100)}%'} | "
                           f"{ms([r['final']['recall'] for r in rs])} | {ms([r['final']['f1'] for r in rs])} | "
                           f"{ms([r['final']['fpr'] for r in rs])} | {'—' if tel != 'tamper_signed' else ms(rej)} |")
    return "\n".join(out)


def table_committee(ds):
    runs = load(ds, "committee")
    if not runs:
        return ""
    g = group(runs, ["decode_policy", "norm_mode", "m", "t", "byz_frac", "member_attack"])
    out = ["| Policy | Norms | m | t | f/N | Member attack | Pred. capture | Captured | Aborted | "
           "Attempts / round | Banned | Final acc. |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(g):
        pol, mode, m, t, bf, ma = key
        rs = g[key]
        n = rs[0]["config"]["n_agents"]
        rounds = rs[0]["config"]["rounds"]
        pred = capture_probability(n, int(round(bf * n)), m, t)
        cap = [r["protocol"]["captured"] / rounds for r in rs]
        ab = [r["protocol"]["aborted"] / rounds for r in rs]
        att = [np.mean(r["protocol"].get("attempts") or [1]) for r in rs]
        ban = [(r["protocol"].get("banned") or [0])[-1] for r in rs]
        out.append(f"| {pol} | {mode} | {m} | {t} | {bf} | {ma.replace('corrupt_', '')} | {pred:.4f} | {ms(cap)}% | "
                   f"{ms(ab)}% | {ms(att, pct=False)} | {ms(ban, pct=False, digits=1)} | "
                   f"{ms([r['final']['accuracy'] for r in rs])} |")
    return "\n".join(out)


def table_cost():
    p = Path("results/overhead.json")
    if not p.exists():
        return ""
    rows = json.loads(p.read_text())["protocol_scaling"]
    out = ["| Backend | Policy | N | m | t | d | MB / round | Upload / agent (KB) | Plain FedAvg upload (KB) | "
           "Seconds / round | Rel. error vs CC |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        be = {"fips": "ML-KEM/ML-DSA", "classical": "X25519/Ed25519"}[r["backend"]]
        out.append(f"| {be} | {r.get('policy', 'correct')} | {r['n']} | {r['m']} | {r['t']} | {r['d']} | "
                   f"{r['bytes_total'] / 1e6:.1f} | {r['bytes_per_agent_upload'] / 1e3:.0f} | "
                   f"{r['plaintext_fedavg_upload_per_agent'] / 1e3:.0f} | {r['round_wall_s']:.2f} | "
                   f"{r.get('relative_error_vs_plaintext_cc', float('nan')):.3f} |")
    return "\n".join(out)


def write_scalability_latex() -> None:
    """Analytic per-round costs of the abort/sketch protocol (k = 128, 8-byte field
    elements).  Sealed-message overhead: ML-KEM-512 ct 768 B + ML-DSA-44 sig 2420 B
    + AEAD tag 16 B + 32 B header.  Committee sizes from Table (sizing), f/N = 0.2,
    per-round capture <= 1e-6 (N = 20: m = 9 with t = 4 >= f, deterministic)."""
    k, seal = 128, 768 + 2420 + 16 + 32
    sizes = {20: (9, 4), 100: (33, 16), 200: (43, 21), 1000: (53, 26)}
    L = ["\\begin{tabular}{@{}rrrrrrrr@{}}", "\\toprule",
         "$N$ & $m$ & $d$ & Upload/agent & Download/member & Public work & FedAvg upload & Ratio \\\\", "\\midrule"]

    def fmt(b):
        for u, s_ in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
            if b >= s_:
                return f"{b / s_:.1f} {u}"
        return f"{b:.0f} B"
    for N, (m, t) in sizes.items():
        for d in (6535, 100_000, 1_000_000):
            up = m * (8 * d + seal) + m * (k + 1) * 8 + 2420
            down = N * (8 * d + seal)
            work = N * N * k + m * d * t + m * (k + 1) * N * m     # sketch distances + agg check + claims check
            L.append(f"{N} & {m} & {d:,} & {fmt(up)} & {fmt(down)} & {work / 1e6:.1f}M & {fmt(4 * d)} & "
                     f"{up / (4 * d):.0f}$\\times$ \\\\".replace(",", "{,}"))
    L += ["\\botrule", "\\end{tabular}"]
    Path("paper/tables/scalability.tex").write_text("\n".join(L) + "\n")


def md_to_latex(md: str) -> str:
    """Markdown pipe table -> booktabs tabular (the body only; captions live in the .tex)."""
    import re
    lines = [l for l in md.splitlines() if l.startswith("|")]
    if not lines:
        return ""
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines if not set(l) <= set("|-: ")]

    short = {"Centered clipping (plaintext)": "CC (plain)", "Multi-Krum + CC (plaintext)": "MK+CC (plain)",
             "PQ secure agg., no defense": "PQ sec.\\ agg., no def.", "PQ-BRDFL, CC only": "PQ-BRDFL, CC only",
             "FLTrust (root data)": "FLTrust", "Centralized (pooled)": "Centralized",
             "scaled_backdoor": "scaled bd.", "labelflip": "label flip", "signflip": "sign-flip",
             "gaussian": "Gaussian", "backdoor": "backdoor", "alie": "ALIE", "ipm": "IPM",
             "No PQC: FedAvg, classical channel": "FedAvg, classical channel",
             "PQC channel only (FedAvg)": "FedAvg, PQ channel",
             "PQC + secret-shared mean": "PQ + shared mean",
             "PQC + plaintext multi-Krum + CC": "PQ + plaintext MK+CC",
             "PQC + shares + CC, no range check": "PQ + shares + CC, no RC",
             "PQC + shares + MK + CC, no range check": "PQ + shares + MK+CC, no RC",
             "PQC + shares + CC only": "PQ + shares + CC",
             "Upload / agent (KB)": "Up/agent KB", "Plain FedAvg upload (KB)": "FedAvg KB",
             "Seconds / round": "s/round", "Rel. error vs CC": "Rel.\\ err.", "MB / round": "MB/round",
             "Acc. sign-flip + wrap": "Acc. SF+wrap", "ASR scaled backdoor": "ASR scaled bd.",
             "Acc. sign-flip": "Acc. SF", "Updates hidden": "Hidden", "Byz. defense": "Byz. def.",
             "Pred. capture": "Pred.", "Attempts / round": "Attempts", "Member attack": "Attack",
             "Tampered paths": "Tampered", "Batches rejected": "Rejected"}

    def cell(c):
        c = short.get(c.replace("**", ""), c) if c.replace("**", "") in short else c
        c = c.replace(" ± ", "±")
        c = c.replace("±", "$\\pm$").replace("%", "\\%").replace("_", "\\_").replace("—", "--")
        c = c.replace("f/N", "$f/N$").replace("≤", "$\\le$")
        c = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", c)
        return c.replace("*", "$^*$")
    ncol = len(rows[0])
    out = ["\\begin{tabular}{@{}l" + "c" * (ncol - 1) + "@{}}", "\\toprule",
           " & ".join(cell(c) for c in rows[0]) + " \\\\", "\\midrule"]
    out += [" & ".join(cell(c) for c in r) + " \\\\" for r in rows[1:]]
    out += ["\\botrule", "\\end{tabular}"]
    return "\n".join(out)


def write_sizing_latex() -> None:
    p = Path("results/privacy_insdn.json")
    if not p.exists():
        return
    rows = json.loads(p.read_text())["committee_sizing"]

    def f(x):
        return "--" if x is None else f"{x[0]} ({x[1]})"
    L = ["\\begin{tabular}{@{}rrcccccc@{}}", "\\toprule",
         " & & \\multicolumn{2}{c}{Guaranteed, exact} & \\multicolumn{2}{c}{Guaranteed, sketch} & "
         "\\multicolumn{2}{c}{Abort, sketch} \\\\",
         "$N$ & $f/N$ & $10^{-3}$ & $10^{-6}$ & $10^{-3}$ & $10^{-6}$ & $10^{-3}$ & $10^{-6}$ \\\\", "\\midrule"]
    for r in rows:
        if r["n"] in (100, 200, 500, 1000):
            L.append(f"{r['n']} & {r['byz_frac']} & {f(r['exact_0.001'])} & {f(r['exact_1e-06'])} & "
                     f"{f(r['sketch_0.001'])} & {f(r['sketch_1e-06'])} & {f(r['abort_0.001'])} & {f(r['abort_1e-06'])} \\\\")
    L += ["\\botrule", "\\end{tabular}"]
    Path("paper/tables/committee_sizing.tex").write_text("\n".join(L) + "\n")


def write_latex(ds: str) -> None:
    out = Path("paper/tables")
    out.mkdir(parents=True, exist_ok=True)
    write_sizing_latex()
    write_scalability_latex()
    tabs = {"detection": table_detection(ds), "byz_acc": table_byzantine(ds, "accuracy"),
            "byz_f1": table_byzantine(ds, "f1"), "byz_fpr": table_byzantine(ds, "fpr"),
            "byz_asr": table_byzantine(ds, "backdoor_asr"), "byzfrac": table_byzfrac(ds),
            "ablation": table_ablation(ds), "telemetry": table_telemetry(ds), "committee": table_committee(ds),
            "detection_all": table_detection(ds), "wrapdemo": table_wrapdemo(ds)}
    md = table_backdoor_defence()
    if md:
        rows = [[c.strip() for c in l.strip("|").split("|")] for l in md.splitlines()[2:]]
        L = ["{\\tiny\\setlength{\\tabcolsep}{2pt}%", "\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
             " & & \\multicolumn{2}{c}{Global model} & \\multicolumn{2}{c}{+ benign gate} & "
             "\\multicolumn{2}{c}{+ gate + local model} \\\\",
             "Dataset & Threat & ASR & FPR & ASR & FPR & ASR & FPR \\\\", "\\midrule"]
        for r in rows:
            r = [c.replace(" ± ", "$\\pm$").replace("no attack (natural rate)", "no attack (natural)")
                 .replace("backdoor, in-range trigger", "in-range trig.").replace("CICIDS2017", "CICIDS") for c in r]
            L.append(" & ".join(r) + " \\\\")
        L += ["\\botrule", "\\end{tabular}}"]
        (out / "backdoor_defence.tex").write_text("\n".join(L) + "\n")
    (out / "cost.tex").write_text("{\\tiny\\setlength{\\tabcolsep}{1.8pt}%\n" + md_to_latex(table_cost()) + "%\n}\n")
    wide = {"byz_acc", "byz_f1", "byz_fpr", "byz_asr", "ablation", "telemetry", "committee"}
    for name, md in tabs.items():
        body = md_to_latex(md)
        if name in wide and body:
            body = "{\\tiny\\setlength{\\tabcolsep}{1.8pt}%\n" + body + "%\n}"
        (out / f"{ds}_{name}.tex").write_text(body + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids")
    ap.add_argument("--latex", action="store_true", help="also write paper/tables/<ds>_*.tex")
    args = ap.parse_args()
    ds = args.dataset
    if args.latex:
        write_latex(ds)
    parts = [
        f"# Results — {ds}\n\nAll numbers are mean ± s.d. over seeds, final round, percent unless stated. "
        "Generated by experiments/make_tables.py from results/" + ds + "/.\n",
        "## Table 4. Detection performance, no attack\n\n" + table_detection(ds),
        "## Table 5a. Accuracy under Byzantine attack (f/N = 0.2)\n\n" + table_byzantine(ds, "accuracy"),
        "## Table 5b. Attack-class F1 under Byzantine attack (f/N = 0.2)\n\n" + table_byzantine(ds, "f1"),
        "## Table 5c. False-positive rate under Byzantine attack (f/N = 0.2)\n\n" + table_byzantine(ds, "fpr"),
        "## Table 5d. Backdoor attack success rate (triggered attack flows classified benign)\n\n"
        + table_byzantine(ds, "backdoor_asr"),
        "## Table 6. Accuracy vs Byzantine fraction\n\n" + table_byzfrac(ds),
        "## Table 8. Ablation / coverage\n\n" + table_ablation(ds),
        "## Table 8b. Wrap-around attack on exact norms\n\n" + table_wrapdemo(ds),
        "## Table 5e. Agent-side defense in depth against the unscaled backdoor (PQ-BRDFL models)\n\n"
        + table_backdoor_defence(),
        "## Table 9. Telemetry fabrication\n\n" + table_telemetry(ds),
        "## Table 10. Committee sortition at N = 60\n\n" + table_committee(ds),
        "## Table 11. Protocol cost per round\n\n" + table_cost(),
    ]
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
