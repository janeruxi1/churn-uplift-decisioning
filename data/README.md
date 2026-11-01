# Data

Two real datasets, neither of which may be redistributed, plus the simulator from the project's
first version. Nothing under `data/raw/` is committed; every notebook falls back to a synthetic
stand-in with the same columns when the real files are missing, which is what CI uses.

## 1. KKBox subscriptions (Part 1, notebooks 01–04)

KKBox's churn challenge on Kaggle ("WSDM - KKBox's Churn Prediction Challenge", 2018): a
Taiwanese music-streaming service's payments (January 2015 to March 2017), member attributes,
March 2017 listening logs, and churn labels for members whose subscription expired in February
2017 (`train.csv`) and March 2017 (`train_v2.csv`). **Churn** = no new paid subscription within
30 days after the current one expires.

**Not committed:** Kaggle competition data. To reproduce:

1. On Kaggle, join the competition (accept its rules) and download `train.csv`, `train_v2.csv`,
   `members_v3.csv`, `transactions.csv`, `transactions_v2.csv` and `user_logs_v2.csv` (about
   1.7 GB as `.7z`). The 30 GB `user_logs.csv` is not needed.
2. Extract them into `data/raw/kkbox/`.
3. `python scripts/convert_kkbox.py` writes Parquet to `data/raw/kkbox/parquet/` (about 1.9 GB),
   in pieces small enough for a laptop; `--seconds 140` stops early and a second run resumes.
   The `.csv` and `.7z` files can then be deleted.

| File | Rows |
|---|---|
| `train` (February 2017 cohort) | 992,931 |
| `train_v2` (March 2017 cohort) | 970,960 |
| `transactions` | 21,547,746 |
| `transactions_v2` | 1,431,009 (overlaps `transactions` in February; kept once) |
| `members_v3` | 6,769,473 |
| `user_logs_v2` (March 2017) | 18,396,362 |

Computed features, scores and lifetime values are cached in `data/raw/kkbox/features/`, which is
ignored by git like the rest of `data/raw/`.

## 2. Telecom retention campaign (Part 2, notebooks 05–06)

Théo Verhelst, Denis Mercier, Jeevan Shrestha and Gianluca Bontempi, "A churn prediction dataset
from the telecom sector: a new benchmark for uplift modeling" (2023). OpenML dataset **45580**
(`churn-uplift-mlg`), licence **CC BY-NC-ND 4.0**: 11,896 high-risk customers of a Belgian
telecom operator in autumn 2020; `t` = called by the call centre with a retention offer, `y` =
churned within two months; 160 PCA components and 18 anonymized categories.

**Not committed:** the licence does not allow modified copies. `src/data/telecom.py` reads the
ARFF saved as `data/raw/telecom/churn_uplift_45580.arff` (download it from
<https://www.openml.org/d/45580>), or fetches it from OpenML if the file is missing. Per-customer
intermediate results (propensity scores) are kept in `data/raw/telecom/cache/`.

## 3. StreamFlix simulator (appendix)

`src/data/simulate.py` generates the simulated subscription business used by the project's first
version, now the appendix in `notebooks/streamflix/`: `subscribers.csv` and
`subscribers_experiment.csv` are committed. Schema and planted truth: [`streamflix.md`](./streamflix.md).
