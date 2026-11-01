"""KKBox subscriptions (WSDM 2018 churn challenge on Kaggle): tables, the churn label, and
point-in-time features.

The data is Kaggle competition data and is never committed. `scripts/convert_kkbox.py` turns the
downloaded files into Parquet under data/raw/kkbox/parquet/; everything here reads that folder
through DuckDB. Without it (CI, a fresh clone), `make_synthetic` writes a small stand-in with the
same columns and a known churn mechanism.

Cohorts: members whose subscription expired in February 2017 (`train`, labels read for training)
and in March 2017 (`train_v2`, the out-of-time test). Features for a cohort use only rows dated
before its cutoff: 1 February 2017 and 1 March 2017.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "kkbox" / "parquet"
CUTOFF = {"feb": 20170201, "mar": 20170301}
LABELS = {"feb": "train", "mar": "train_v2"}
NUMERIC = ["plan_days", "list_price", "paid", "discount", "auto_renew", "last_cancel",
           "n_tx_30", "n_tx_90", "n_tx_365", "n_cancel_30", "n_cancel_90", "n_cancel_365", "n_tx_total",
           "share_auto_renew", "tenure_days", "days_to_expiry", "age", "reg_days", "has_member_row"]
CATEGORICAL = ["payment_method", "city", "gender", "registered_via"]


def use_synthetic() -> bool:
    return os.environ.get("KKBOX_SYNTHETIC", "") in ("1", "true", "yes") or not (RAW / "train.parquet").exists()


def connect(raw: Path | None = None) -> tuple[duckdb.DuckDBPyConnection, str]:
    """A DuckDB connection with views tx, members, train, train_v2 (and logs if present)."""
    source = "synthetic" if raw is None and use_synthetic() else "real"
    if raw is None:
        raw = make_synthetic() if source == "synthetic" else RAW
    con = duckdb.connect()
    try:
        con.execute("SET enable_progress_bar=false")
    except duckdb.Error:   # inside Jupyter without ipywidgets the setting cannot be changed; harmless
        pass
    con.execute("SET memory_limit='2500MB'")    # spill to disk rather than exhaust a small machine
    con.execute(f"SET temp_directory='{Path(tempfile.gettempdir()) / 'kkbox_duckdb'}'")
    tx_parts = [f"SELECT * FROM read_parquet('{raw}/transactions/*.parquet')"] if (raw / "transactions").exists() else []
    if (raw / "transactions_v2.parquet").exists():
        tx_parts.append(f"SELECT * FROM read_parquet('{raw}/transactions_v2.parquet')")
    # transactions_v2 repeats some February rows of transactions: keep each row once
    con.execute(f"CREATE TABLE tx AS SELECT DISTINCT * FROM ({' UNION ALL '.join(tx_parts)})")
    con.execute(f"CREATE VIEW members AS SELECT * FROM read_parquet('{raw}/members_v3.parquet')")
    for name in ("train", "train_v2"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{raw}/{name}.parquet')")
    if (raw / "user_logs_v2").exists():
        con.execute(f"CREATE VIEW logs AS SELECT * FROM read_parquet('{raw}/user_logs_v2/*.parquet')")
    return con, source


def features_dir(source: str) -> Path:
    """Where computed features are cached: inside the ignored raw-data folder for the real data."""
    d = (RAW.parent / "features") if source == "real" else Path(tempfile.gettempdir()) / "kkbox_features_synthetic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _d(col: str) -> str:
    return f"strptime(CAST({col} AS VARCHAR), '%Y%m%d')"


def rebuild_label(con, cohort: str = "feb") -> pd.DataFrame:
    """Churn rebuilt from payments: E = the member's latest expiry inside the cohort month; renewed if
    a non-cancelled transaction extending past E is dated within 30 days after E."""
    lo = CUTOFF[cohort]
    hi = lo + 27 if cohort == "feb" else lo + 30
    return con.sql(f"""
        WITH e AS (SELECT l.msno, max(x.membership_expire_date) AS expiry
                   FROM {LABELS[cohort]} l JOIN tx x USING (msno)
                   WHERE x.membership_expire_date BETWEEN {lo} AND {hi} GROUP BY 1)
        SELECT e.msno, e.expiry,
               1 - max(CASE WHEN x.is_cancel = 0 AND x.membership_expire_date > e.expiry
                             AND {_d('x.transaction_date')} <= {_d('e.expiry')} + INTERVAL 30 DAY
                            THEN 1 ELSE 0 END) AS churn_rebuilt
        FROM e LEFT JOIN tx x USING (msno) GROUP BY 1, 2""").df()


def features(con, cohort: str, tx_table: str = "tx") -> pd.DataFrame:
    """One row per cohort member, built only from rows dated before the cohort's cutoff."""
    cut = CUTOFF[cohort]
    return con.sql(f"""
        WITH c AS (SELECT msno FROM {LABELS[cohort]}),
        h AS (SELECT x.* FROM {tx_table} x JOIN c USING (msno) WHERE x.transaction_date < {cut}),
        last AS (SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY msno
                     ORDER BY transaction_date DESC, membership_expire_date DESC, is_cancel DESC,
                              actual_amount_paid DESC, plan_list_price DESC, payment_plan_days DESC,
                              payment_method_id DESC, is_auto_renew DESC) AS rn FROM h)
                 WHERE rn = 1),
        agg AS (SELECT msno,
                  count(*) AS n_tx_total,
                  sum(CASE WHEN {_d('transaction_date')} >= {_d(str(cut))} - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS n_tx_30,
                  sum(CASE WHEN {_d('transaction_date')} >= {_d(str(cut))} - INTERVAL 90 DAY THEN 1 ELSE 0 END) AS n_tx_90,
                  sum(CASE WHEN {_d('transaction_date')} >= {_d(str(cut))} - INTERVAL 365 DAY THEN 1 ELSE 0 END) AS n_tx_365,
                  sum(CASE WHEN is_cancel = 1 AND {_d('transaction_date')} >= {_d(str(cut))} - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS n_cancel_30,
                  sum(CASE WHEN is_cancel = 1 AND {_d('transaction_date')} >= {_d(str(cut))} - INTERVAL 90 DAY THEN 1 ELSE 0 END) AS n_cancel_90,
                  sum(CASE WHEN is_cancel = 1 AND {_d('transaction_date')} >= {_d(str(cut))} - INTERVAL 365 DAY THEN 1 ELSE 0 END) AS n_cancel_365,
                  avg(is_auto_renew) AS share_auto_renew,
                  date_diff('day', min({_d('transaction_date')}), {_d(str(cut))}) AS tenure_days,
                  date_diff('day', {_d(str(cut))}, max({_d('membership_expire_date')})) AS days_to_expiry
                FROM h GROUP BY 1)
        SELECT c.msno,
               l.payment_plan_days AS plan_days, l.plan_list_price AS list_price, l.actual_amount_paid AS paid,
               l.plan_list_price - l.actual_amount_paid AS discount, l.is_auto_renew AS auto_renew,
               l.is_cancel AS last_cancel, CAST(l.payment_method_id AS VARCHAR) AS payment_method,
               a.n_tx_30, a.n_tx_90, a.n_tx_365, a.n_cancel_30, a.n_cancel_90, a.n_cancel_365, a.n_tx_total,
               a.share_auto_renew, a.tenure_days, a.days_to_expiry,
               CAST(m.city AS VARCHAR) AS city,
               CASE WHEN m.bd BETWEEN 10 AND 80 THEN m.bd END AS age,
               coalesce(nullif(m.gender, ''), 'unknown') AS gender,
               CAST(m.registered_via AS VARCHAR) AS registered_via,
               CASE WHEN m.registration_init_time < {cut}
                    THEN date_diff('day', {_d('m.registration_init_time')}, {_d(str(cut))}) END AS reg_days,
               CASE WHEN m.msno IS NULL THEN 0 ELSE 1 END AS has_member_row
        FROM c LEFT JOIN last l USING (msno) LEFT JOIN agg a USING (msno) LEFT JOIN members m USING (msno)
        ORDER BY c.msno""").df()


def labels(con, cohort: str) -> pd.DataFrame:
    return con.sql(f"SELECT msno, is_churn FROM {LABELS[cohort]} ORDER BY msno").df()


# ---------------------------------------------------------------- synthetic stand-in
TRUTH = {"base_logit": -3.2, "auto_renew_off": 2.2, "cancel": 1.8, "discount": 0.6}


def make_synthetic(n: int = 6000, seed: int = 11, out: Path | None = None) -> Path:
    """Members, two years of monthly payments, and Feb/Mar 2017 cohorts whose churn depends on
    auto-renew, a recent cancellation and discounts (TRUTH). Written as Parquet; returns the folder."""
    rng = np.random.default_rng(seed)
    out = out or Path(tempfile.gettempdir()) / "kkbox_synthetic"
    (out / "transactions").mkdir(parents=True, exist_ok=True)
    ids = np.array([f"m{i:06d}" for i in range(n)])
    members = pd.DataFrame({"msno": ids, "city": rng.integers(1, 22, n), "bd": np.where(rng.random(n) < 0.6, 0, rng.integers(15, 60, n)),
                            "gender": rng.choice(["male", "female", ""], n), "registered_via": rng.choice([3, 4, 7, 9], n),
                            "registration_init_time": 20140101 + rng.integers(0, 2, n) * 10000 + rng.integers(1, 12, n) * 100 + 1})
    rows = []
    auto = rng.random(n) < 0.8
    disc = rng.random(n) < 0.1
    cohort = np.where(rng.random(n) < 0.5, "feb", "mar")
    for i, m in enumerate(ids):
        exp_day = int(rng.integers(1, 28))
        end = pd.Timestamp(2017, 2 if cohort[i] == "feb" else 3, exp_day)
        months = int(rng.integers(3, 24))
        cancel_last = rng.random() < (0.25 if not auto[i] else 0.03)
        for k in range(months, 0, -1):
            exp = end - pd.DateOffset(months=k - 1)
            txd = exp - pd.DateOffset(days=30)
            price = 149
            rows.append((m, 41 if auto[i] else 38, 30, price, price - (50 if disc[i] else 0), int(auto[i]),
                         int(txd.strftime("%Y%m%d")), int(exp.strftime("%Y%m%d")), int(cancel_last and k == 1)))
        logit = (TRUTH["base_logit"] + TRUTH["auto_renew_off"] * (not auto[i]) + TRUTH["cancel"] * cancel_last
                 + TRUTH["discount"] * disc[i])
        churn = rng.random() < 1 / (1 + np.exp(-logit))
        if not churn:   # renewal shortly after expiry
            nd = end + pd.DateOffset(days=int(rng.integers(0, 5)))
            rows.append((m, 41, 30, 149, 149, int(auto[i]), int(nd.strftime("%Y%m%d")),
                         int((nd + pd.DateOffset(days=30)).strftime("%Y%m%d")), 0))
        members.loc[i, "churn"] = int(churn)
    tx = pd.DataFrame(rows, columns=["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                                     "actual_amount_paid", "is_auto_renew", "transaction_date",
                                     "membership_expire_date", "is_cancel"])
    tx[tx["transaction_date"] < 20170301].to_parquet(out / "transactions" / "part-000.parquet", index=False)
    tx[tx["transaction_date"] >= 20170201].to_parquet(out / "transactions_v2.parquet", index=False)
    members.drop(columns="churn").to_parquet(out / "members_v3.parquet", index=False)
    lab = members[["msno", "churn"]].rename(columns={"churn": "is_churn"}).astype({"is_churn": int})
    lab[cohort == "feb"].to_parquet(out / "train.parquet", index=False)
    lab[cohort == "mar"].to_parquet(out / "train_v2.parquet", index=False)
    return out
