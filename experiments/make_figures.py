"""Figures for the manuscript (static PDF, print-safe).

Colour follows the entity, in a fixed categorical order.  Every series also
has its own marker and line style, so identity survives grayscale print and
color-vision deficiency.  One y-axis per chart, recessive grid, and a legend
whenever there are two or more series.

    python -m experiments.make_figures --dataset cicids
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pqbrdfl.agg.committee import capture_probability  # noqa: E402

# reference categorical palette, fixed order (validated light-mode steps)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# entity -> (color slot, marker, linestyle): stable across every figure
ENT = {
    "pqbrdfl": (0, "o", "-"), "cc": (1, "s", "--"), "median": (2, "^", "-."),
    "fedavg": (3, "D", ":"), "krum": (4, "v", "--"), "fltrust": (5, "P", "-."),
    "trimmed_mean": (6, "X", ":"), "dfl_gossip": (7, "*", "--"),
}
LABEL = {"pqbrdfl": "PQ-BRDFL (ours)", "cc": "CC (plaintext)", "median": "Median", "fedavg": "FedAvg",
         "krum": "Krum", "fltrust": "FLTrust", "trimmed_mean": "Trimmed mean", "dfl_gossip": "Gossip DFL"}


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 5, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def load(ds, exp):
    return [json.loads(p.read_text()) for p in sorted(Path("results", ds, exp).glob("*.json"))]


DSNAME = {"insdn": "InSDN", "cicids": "CICIDS2017"}


def _mean_hist(runs, agg, atk, metric):
    hs = [r["history"] for r in runs if r["config"]["aggregator"] == agg and r["config"]["attack"] == atk]
    if not hs:
        return None, None
    rounds = [h["round"] for h in hs[0]]
    return rounds, np.array([[h[metric] for h in hh] for hh in hs]).mean(0) * 100


def fig_rounds(ds, out):
    """Accuracy under sign-flip and backdoor success under scaled backdoor, per round."""
    runs = load(ds, "byzantine")
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (atk, met, title, yl) in zip(axes, [("signflip", "accuracy", "Sign-flip: accuracy", "Test accuracy (%)"),
                                                ("scaled_backdoor", "backdoor_asr", "Scaled backdoor: attack success",
                                                 "Backdoor ASR (%)")]):
        for agg in ["fedavg", "median", "krum", "cc", "pqbrdfl"]:
            rounds, v = _mean_hist(runs, agg, atk, met)
            if rounds is None:
                continue
            ci, mk, ls = ENT[agg]
            ax.plot(rounds, v, color=C[ci], marker=mk, ls=ls, label=LABEL[agg])
        ax.set_title(f"{title} ({DSNAME.get(ds, ds)}, f/N = 0.2)", color=INK, fontsize=8.5)
        ax.set_xlabel("Round")
        ax.set_ylabel(yl)
    axes[1].legend(loc="center right", fontsize=7)
    fig.savefig(out / f"fig_rounds_{ds}.pdf")
    plt.close(fig)


def fig_backdoor(out):
    """Final backdoor ASR per aggregator, both datasets, backdoor and scaled backdoor."""
    aggs = ["fedavg", "median", "trimmed_mean", "krum", "fltrust", "cc", "mk_cc", "pqbrdfl_cc", "pqbrdfl"]
    names = ["FedAvg", "Median", "Trim. mean", "Krum", "FLTrust", "CC", "MK+CC", "PQ-BRDFL (CC)", "PQ-BRDFL"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, ds in zip(axes, ["insdn", "cicids"]):
        runs = load(ds, "byzantine")
        x = np.arange(len(aggs))
        for k, (atk, lab) in enumerate([("backdoor", "Backdoor"), ("scaled_backdoor", "Scaled backdoor (x4)")]):
            vals = []
            for a in aggs:
                v = [r["final"]["backdoor_asr"] * 100 for r in runs
                     if r["config"]["aggregator"] == a and r["config"]["attack"] == atk]
                vals.append(np.mean(v) if v else np.nan)
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=C[k], edgecolor="white", linewidth=1.0, label=lab)
        ax.set_xticks(x, names, fontsize=6.5, rotation=40, ha="right")
        ax.set_title(DSNAME[ds], color=INK)
    axes[0].set_ylabel("Backdoor ASR (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.savefig(out / "fig_backdoor.pdf")
    plt.close(fig)


def fig_byzfrac(ds, out):
    runs = load(ds, "byzfrac")
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (atk, met, title, yl) in zip(axes, [("signflip", "accuracy", "Sign-flip", "Final accuracy (%)"),
                                                ("scaled_backdoor", "backdoor_asr", "Scaled backdoor",
                                                 "Backdoor ASR (%)")]):
        g = defaultdict(lambda: defaultdict(list))
        for r in runs:
            if r["config"]["attack"] == atk:
                g[r["config"]["aggregator"]][r["config"]["byz_frac"]].append(r["final"][met] * 100)
        for agg in ["fedavg", "median", "cc", "pqbrdfl"]:
            xs = sorted(g[agg])
            if not xs:
                continue
            ci, mk, ls = ENT[agg]
            ax.errorbar(xs, [np.mean(g[agg][x]) for x in xs], yerr=[np.std(g[agg][x]) for x in xs],
                        color=C[ci], marker=mk, ls=ls, capsize=2, label=LABEL[agg])
        ax.set_title(f"{title} ({DSNAME.get(ds, ds)})", color=INK)
        ax.set_xlabel("Byzantine fraction f/N")
        ax.set_ylabel(yl)
        ax.set_xticks([0.1, 0.2, 0.3])
        # f/N = 0.3 is outside the formal robustness condition (f < n/4): mark it as an empirical stress test
        ax.axvspan(0.25, 0.34, color="0.90", zorder=0, lw=0)
        ax.set_xlim(0.08, 0.34)
        ax.text(0.295, 0.5, "empirical\nstress test\n(outside\nThm. 2)", transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=6, color="0.35")
    axes[1].legend(loc="center left", fontsize=7)
    fig.savefig(out / f"fig_byzfrac_{ds}.pdf")
    plt.close(fig)


def fig_backdoor_tradeoff(out, agg="pqbrdfl"):
    """Backdoor ASR vs FPR as the agent-side defense is tightened."""
    runs = [json.loads(p.read_text()) for p in Path("results/backdoor_posthoc").glob(f"*__{agg}__*.json")]
    if not runs:
        return
    steps = ["none", "gate", "local1+gate", "local2+gate", "local5+gate", "local10+gate", "local20+gate"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, ds in zip(axes, ["insdn", "cicids"]):
        for k, (trig, lab) in enumerate([("oor", "Backdoor"), ("inrange", "Backdoor, in-range trigger")]):
            rs = [r for r in runs if r["dataset"] == ds and r["attack"] == "backdoor" and r["trigger"] == trig]
            if not rs:
                continue
            xs = [np.mean([r["defences"][s]["fpr"] for r in rs]) * 100 for s in steps]
            ys = [np.mean([r["defences"][s]["asr"] for r in rs]) * 100 for s in steps]
            ax.plot(xs, ys, color=C[k], marker="os"[k], ls=["-", "--"][k], label=lab)
        nat = [r for r in runs if r["dataset"] == ds and r["attack"] == "none"]
        if nat:
            xs = [np.mean([r["defences"][s]["fpr"] for r in nat]) * 100 for s in steps]
            ys = [np.mean([r["defences"][s]["asr"] for r in nat]) * 100 for s in steps]
            ax.plot(xs, ys, color=C[2], marker="^", ls=":", label="No attack (natural rate)")
        ax.set_title(DSNAME[ds], color=INK)
        ax.set_xlabel("False-positive rate (%)")
    axes[0].set_ylabel("Backdoor ASR (%)")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.savefig(out / "fig_backdoor_tradeoff.pdf")
    plt.close(fig)


def fig_cost(out):
    p = Path("results/overhead.json")
    if not p.exists():
        return
    rows = [r for r in json.loads(p.read_text())["protocol_scaling"] if r["backend"] == "fips"]
    phases = ["member_keys", "client_share_seal", "member_open", "zero_sharing", "member_validate",
              "public_decode", "aggregate"]
    names = ["Member keys", "Share + seal", "Open", "Zero sharing", "Validate", "Public decode", "Aggregate"]
    labels = [f"N={r['n']}, m={r['m']}\n{r.get('policy', '')}, d={r['d'] // 1000}k" for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    for k, (ph, nm) in enumerate(zip(phases, names)):
        v = np.array([r["phases_s"].get(ph, 0.0) for r in rows])
        if not v.any():
            continue
        ax.bar(x, v, 0.6, bottom=bottom, color=C[k], edgecolor="white", linewidth=1.0, label=nm)
        bottom += v
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylabel("Seconds per round")
    ax.legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out / "fig_cost.pdf")
    plt.close(fig)


def fig_capture(out):
    """Per-round failure probability vs committee size, N = 200, three regimes."""
    regimes = [("Guaranteed output, exact (m = 4t+1)", lambda m: (m - 1) // 4),
               ("Guaranteed output, sketch (m = 3t+1)", lambda m: (m - 1) // 3),
               ("Security with abort, sketch (m = 2t+1)", lambda m: (m - 1) // 2)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    n = 200
    for ax, frac in zip(axes, [0.1, 0.2]):
        f = int(frac * n)
        for k, (lab, tf) in enumerate(regimes):
            ms_, pr = [], []
            for m in range(5, 201, 2):
                v = capture_probability(n, f, m, tf(m))
                if v > 1e-12:
                    ms_.append(m); pr.append(v)
            ax.semilogy(ms_, pr, color=C[k], ls=["-", "--", ":"][k], lw=1.8, label=lab)
        ax.axhline(1e-6, color=INK2, lw=0.8)
        ax.text(ax.get_xlim()[1], 1.6e-6, "10⁻⁶ target", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_title(f"N = 200, f/N = {frac}", color=INK)
        ax.set_xlabel("Committee size m")
        ax.set_ylim(1e-12, 1.5)
    axes[0].set_ylabel("P[c > t] per round")
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.savefig(out / "fig_capture.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids")
    args = ap.parse_args()
    style()
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ds in ("insdn", "cicids"):
        fig_rounds(ds, out)
        fig_byzfrac(ds, out)
    fig_backdoor(out)
    fig_backdoor_tradeoff(out)
    fig_cost(out)
    fig_capture(out)
    print("figures in", out)


if __name__ == "__main__":
    main()
