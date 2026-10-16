# PQ-BRDFL: code and results

Code, tests and per-run results for the manuscript "Post-Quantum Byzantine-Resilient Federated Learning for SDN Intrusion Detection" (under review). Every table and figure in the manuscript is generated from the JSON files in `results/`, so nothing is typed by hand.

## What is here

| Path | Contents |
|---|---|
| `pqbrdfl/` | The protocol. Crypto backends (standard FIPS 203/204 through liboqs, an optional Ascon backend, a classical baseline), Shamir sharing, committee aggregation, the robust rules, the federated-learning simulator, the attacks and the local backdoor check. |
| `experiments/` | Experiment runners, and the scripts that turn results into the manuscript's tables and figures. |
| `tests/` | 218 automated tests. |
| `results/` | One JSON file per run (937 files in total), plus the generated table text. |
| `data/CICIDS2017/` | The CICIDS2017 subsample used in the manuscript. |
| `docs/DATASETS.md` | Where to get the two datasets, their licences, SHA-256 hashes and the expected file layout. |
| `requirements-pqbrdfl.txt` | Pinned Python packages. |

## Setup

Python 3.13.3 was used. The FIPS backend needs the liboqs C library (0.15 or newer) on the system.

```bash
pip install -r requirements-pqbrdfl.txt
python -m pytest tests/            # 218 tests, about 20 seconds
```

## Data

Both datasets are public. `docs/DATASETS.md` lists the official links, licences, SHA-256 hashes and where files go.

* **InSDN** is not included. Download it from the official page and put the three CSV files in `data/InSDN/`.
* **CICIDS2017**: the 56,661-flow subsample used in the manuscript is included at `data/CICIDS2017/CICIDS2017.csv` (SHA-256 `03ba3626a0f9bb73b90c56772893e68b0577285130891cd7d811542c17b032dd`). It is derived from the public CSV flows of CICIDS2017 (Sharafaldin et al., ICISSP 2018, doi:10.5220/0006639801080116). We do not know how it was drawn from the full release. The dataset's terms ask users to cite that paper.

## Reproduce

Regenerate every table and figure from the included results. No dataset is needed for this step.

```bash
python -m experiments.make_tables --dataset insdn  --latex
python -m experiments.make_tables --dataset cicids --latex
python -m experiments.make_figures
```

Rerun the experiments. This needs the datasets. Times are for one Apple M2 laptop with 4 workers.

| Command | Time |
|---|---|
| `python -m experiments.run_experiments --exp all --dataset insdn` | about 1.5 h |
| `python -m experiments.run_experiments --exp all --dataset cicids` | about 25 min |
| `python -m experiments.run_experiments --exp wrapdemo --dataset insdn` (and `cicids`) | about 5 min each |
| `python -m experiments.backdoor_posthoc pqbrdfl` | about 25 min |
| `python -m experiments.privacy --dataset insdn` (and `cicids`) | about 2 min each |
| `python -m experiments.overhead` | about 10 min |

## Notes

* Plaintext runs are bit-for-bit reproducible from the seed. Protocol runs are reproducible only statistically, because ML-DSA signing is randomized and the range-check seed hashes the signed commitments. Two regenerated runs differed by 0.02 to 0.03 accuracy points.
* Every experiment uses the standard FIPS backend. The Ascon backend is optional, not FIPS-compliant, and was validated for correctness only.
* `--dataset synthetic` generates SDN-like flows for smoke tests. Nothing produced on it should be reported as a result.

## Licence

The code is released under the MIT licence (`LICENSE`). The included CICIDS2017 subsample remains subject to the terms of the original dataset.
