"""Datasets and agent partitioning for the SDN intrusion-detection task.

Real datasets are loaded from local CSVs (they are not redistributed here;
see docs/DATASETS.md for download instructions):

* InSDN      (Elsayed, Le-Khac & Jurcut, IEEE Access 8, 2020) -- primary: collected
             on an SDN testbed (ONOS + OVS), includes SDN-specific attacks.
* CICIDS2017 (Sharafaldin et al., ICISSP 2018) and UNSW-NB15 (Moustafa & Slay,
             MilCIS 2015) -- secondary, non-SDN, used to check that conclusions
             are not an artefact of one dataset.

``synthetic_sdn`` is a *synthetic* stand-in with the same shape (flow features,
8 classes, heavy-tailed features).  It exists so the pipeline and tests run
without the real data.  Numbers produced on it are smoke tests, not results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

INSDN_FILES = ("Normal_data.csv", "OVS.csv", "metasploitable-2.csv")
INSDN_DROP = ("Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port", "Dst Port", "Label")
UNSW_DROP = ("id", "attack_cat", "label", "proto", "service", "state")


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: list[str]
    normal_class: int
    synthetic: bool = False

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def _standardise(Xtr, Xte):
    Xtr = np.sign(Xtr) * np.log1p(np.abs(Xtr))
    Xte = np.sign(Xte) * np.log1p(np.abs(Xte))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)


def _split(X, y, test_frac, rng):
    idx = rng.permutation(len(y))
    n_te = int(len(y) * test_frac)
    return X[idx[n_te:]], y[idx[n_te:]], X[idx[:n_te]], y[idx[:n_te]]


def load_insdn(root: str | Path, test_frac: float = 0.2, seed: int = 0,
               min_class: int = 50, dedup: bool = True) -> Dataset:
    import pandas as pd
    root = Path(root)
    frames = [pd.read_csv(root / f, low_memory=False) for f in INSDN_FILES if (root / f).exists()]
    if not frames:
        raise FileNotFoundError(f"InSDN CSVs not found under {root} (expected {INSDN_FILES})")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    # OVS.csv labels one class "DDoS " (trailing space); strip so it merges with "DDoS"
    df["Label"] = df["Label"].astype(str).str.strip()
    counts = df["Label"].value_counts()
    df = df[df["Label"].isin(counts[counts >= min_class].index)]
    classes = sorted(df["Label"].unique(), key=lambda c: (c != "Normal", c))
    y = df["Label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    feat = [c for c in df.columns if c not in INSDN_DROP]
    if dedup:
        # 46% of InSDN rows repeat (features, label) exactly; left in, they straddle the
        # train/test split and inflate every score.  Deduplicate before splitting.
        df = df.drop_duplicates(subset=feat + ["Label"])
        counts = df["Label"].value_counts()
        df = df[df["Label"].isin(counts[counts >= min_class].index)]
        classes = sorted(df["Label"].unique(), key=lambda c: (c != "Normal", c))
        y = df["Label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = df[feat]
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.loc[:, X.std() > 0].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = _split(X, y, test_frac, rng)
    Xtr, Xte = _standardise(Xtr, Xte)
    return Dataset("InSDN", Xtr, ytr, Xte, yte, list(classes), 0)


def load_unsw(root: str | Path) -> Dataset:
    import pandas as pd
    root = Path(root)
    tr = pd.read_csv(root / "UNSW_NB15_training-set.csv")
    te = pd.read_csv(root / "UNSW_NB15_testing-set.csv")
    classes = sorted(tr["attack_cat"].unique(), key=lambda c: (c != "Normal", c))
    cmap = {c: i for i, c in enumerate(classes)}
    te = te[te["attack_cat"].isin(cmap)]
    feat = [c for c in tr.columns if c not in UNSW_DROP]
    Xtr, Xte = _standardise(tr[feat].to_numpy(np.float64), te[feat].to_numpy(np.float64))
    return Dataset("UNSW-NB15", Xtr, tr["attack_cat"].map(cmap).to_numpy(),
                   Xte, te["attack_cat"].map(cmap).to_numpy(), list(classes), 0)


def load_cicids(path: str | Path, test_frac: float = 0.2, seed: int = 0,
                min_class: int = 50, dedup: bool = True) -> Dataset:
    """CICIDS2017 (Sharafaldin et al., ICISSP 2018), MachineLearningCVE column
    layout; accepts the full release or a subsample with a 'Label' column."""
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if dedup:
        df = df.drop_duplicates()          # 22% of the local subsample repeats exactly
    counts = df["Label"].value_counts()
    df = df[df["Label"].isin(counts[counts >= min_class].index)]
    classes = sorted(df["Label"].unique(), key=lambda c: (c != "BENIGN", c))
    y = df["Label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.loc[:, X.std() > 0].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = _split(X, y, test_frac, rng)
    Xtr, Xte = _standardise(Xtr, Xte)
    return Dataset("CICIDS2017", Xtr, ytr, Xte, yte, list(classes), 0)


def synthetic_sdn(n: int = 40000, n_features: int = 48, seed: int = 0,
                  test_frac: float = 0.2) -> Dataset:
    """Synthetic SDN-flow-like data: 8 classes, class-specific feature subsets,
    heavy tails and label noise.  For pipeline testing only."""
    rng = np.random.default_rng(seed)
    classes = ["Normal", "DoS", "DDoS", "Probe", "BFA", "Web-Attack", "BOTNET", "U2R"]
    prior = np.array([0.40, 0.14, 0.18, 0.12, 0.06, 0.04, 0.04, 0.02])
    y = rng.choice(len(classes), size=n, p=prior)
    centers = rng.normal(0, 1.0, (len(classes), n_features))
    for c in range(1, len(classes)):             # attacks share structure with each other
        centers[c] = 0.5 * centers[c] + 0.5 * centers[1]
    active = rng.random((len(classes), n_features)) < 0.35
    X = rng.standard_t(df=4, size=(n, n_features)) * 0.9
    X += np.where(active[y], centers[y] * 1.1, 0.0)
    flip = rng.random(n) < 0.02
    y = np.where(flip, rng.integers(0, len(classes), n), y)
    Xtr, ytr, Xte, yte = _split(X, y, test_frac, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return Dataset("synthetic-SDN", ((Xtr - mu) / sd).astype(np.float32), ytr,
                   ((Xte - mu) / sd).astype(np.float32), yte, classes, 0, synthetic=True)


def _cached(name: str, seed: int, build) -> Dataset:
    """Preprocess once, then load a compact float32 .npz (workers stay small in RAM)."""
    import json
    cdir = Path("data/cache")
    path = cdir / f"{name}_seed{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return Dataset(meta["name"], z["X_train"], z["y_train"], z["X_test"], z["y_test"],
                       meta["classes"], meta["normal_class"], meta["synthetic"])
    ds = build()
    cdir.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"name": ds.name, "classes": ds.classes, "normal_class": ds.normal_class,
                       "synthetic": ds.synthetic})
    np.savez(path, X_train=ds.X_train, y_train=ds.y_train, X_test=ds.X_test, y_test=ds.y_test, meta=meta)
    return ds


def load(name: str, root: str | None = None, seed: int = 0) -> Dataset:
    if name in ("insdn", "cicids") and root is None:
        return _cached(name, seed, lambda: load(name, {"insdn": "data/InSDN",
                                                       "cicids": "data/CICIDS2017/CICIDS2017.csv"}[name], seed))
    if name == "insdn":
        return load_insdn(root or "data/InSDN", seed=seed)
    if name == "unsw":
        return load_unsw(root or "data/UNSW-NB15")
    if name == "cicids":
        return load_cicids(root or "data/CICIDS2017/CICIDS2017.csv", seed=seed)
    if name == "synthetic":
        return synthetic_sdn(seed=seed)
    raise ValueError(name)


def partition(y: np.ndarray, n_agents: int, alpha: float, rng: np.random.Generator,
              min_size: int = 30) -> list[np.ndarray]:
    """Dirichlet(alpha) label-skew partition: agent = one switch / edge site.

    Small alpha -> each switch sees few attack types (the non-IID regime where
    statistical Byzantine filters misfire).
    """
    classes = np.unique(y)
    for _ in range(100):
        parts = [[] for _ in range(n_agents)]
        for c in classes:
            idx = rng.permutation(np.nonzero(y == c)[0])
            props = rng.dirichlet([alpha] * n_agents)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for a, chunk in enumerate(np.split(idx, cuts)):
                parts[a].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(rng.permutation(p)) for p in parts]
    raise RuntimeError("could not satisfy min_size; raise alpha or lower min_size")
