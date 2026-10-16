# Datasets

InSDN is not redistributed here: download it from the official source below and put the files under `data/InSDN/`. The CICIDS2017 subsample used in the manuscript is included (see the last section but one).

| Dataset | Official source | Citation | Terms | Expected layout |
|---|---|---|---|---|
| **InSDN** (primary, SDN traffic) | https://aseados.ucd-cs.ie/datasets/SDN/ (we used `InSDN_DatasetCSV.zip`) | Elsayed, Le-Khac and Jurcut, IEEE Access 8, 165263-165284 (2020), doi:10.1109/ACCESS.2020.3022633 | Open access, CC BY-NC 4.0 (stated in the dataset's README). The old address aseados.ucd.ie redirects here. | `data/InSDN/Normal_data.csv`, `data/InSDN/OVS.csv`, `data/InSDN/metasploitable-2.csv` |
| **CICIDS2017** (secondary, not SDN) | https://www.unb.ca/cic/datasets/ids-2017.html | Sharafaldin, Habibi Lashkari and Ghorbani, ICISSP 2018, doi:10.5220/0006639801080116 | The data are "publicly available for researchers"; users must cite the paper in the previous column. | `data/CICIDS2017/CICIDS2017.csv` (one CSV with a `Label` column) |

The loader also supports UNSW-NB15 (`data/UNSW-NB15/`). The manuscript does not use it.

Both loaders deduplicate before splitting: 46% of the InSDN rows and 22% of our CICIDS2017 subsample are exact duplicates. A cache of the preprocessed arrays is written to `data/cache/`.

## File hashes (SHA-256) used for the manuscript

| File | SHA-256 |
|---|---|
| `InSDN_DatasetCSV.zip` | `ee89ec66705c529fef26b27dd18c6aa56d469006f7ade9cd92fd4a503e320d7d` |
| `data/InSDN/Normal_data.csv` | `75c5ea5e3526c9500e3d73cbdb11b99ff3fbfb8dac8dff8c39e9dd35d60556b3` |
| `data/InSDN/OVS.csv` | `8b218c16769dea961028d511928e7da139302afe341cea29dd65e526d580e092` |
| `data/InSDN/metasploitable-2.csv` | `5cc80c7b5707bf92d3ea501a9af8b488cc4680f0a5b29b5b608e92bbe6e297a7` |
| `data/CICIDS2017/CICIDS2017.csv` | `03ba3626a0f9bb73b90c56772893e68b0577285130891cd7d811542c17b032dd` |

Check with `shasum -a 256 data/InSDN/*.csv data/CICIDS2017/*.csv`.

## The CICIDS2017 subsample

The CICIDS2017 results in `results/cicids/` come from a 56,661-flow subsample of the MachineLearningCVE release. We do not know how this subsample was drawn from the full release, so the CICIDS2017 results are a secondary check only. The subsample has seven classes, and Infiltration (36 flows) is dropped by the 50-flow minimum. It is included in this repository at `data/CICIDS2017/CICIDS2017.csv` (19.9 MB); its hash is given above. If you use it, cite Sharafaldin et al. (ICISSP 2018).

## Reproduce on InSDN

```bash
python -m experiments.run_experiments --exp all --dataset insdn --data-root data/InSDN
python -m experiments.privacy --dataset insdn --data-root data/InSDN
python -m experiments.make_tables --dataset insdn > results/tables_insdn.md
```
