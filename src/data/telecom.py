"""The telecom retention campaign (Verhelst et al. 2023; OpenML dataset 45580, CC BY-NC-ND 4.0).

11,896 high-risk customers of a Belgian telecom operator, autumn 2020. A random 24% were held back;
the rest were called by a call centre with a retention offer. `t` = 1 if called, `y` = 1 if the
customer churned within two months. 160 PCA components (`PC1`-`PC160`) and 18 anonymized
categorical features (`FACTOR1`-`FACTOR18`).

The licence does not allow redistribution of modified copies, so the data is never committed. It
is read from data/raw/telecom/ (the ARFF downloaded from OpenML) or fetched from OpenML; CI and
anyone without it get a synthetic stand-in with the same columns and a known effect.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "telecom" / "churn_uplift_45580.arff"
OPENML_ID = 45580
TREAT, OUTCOME = "t", "y"
N_PC, N_FACTOR = 160, 18
PCS = [f"PC{i}" for i in range(1, N_PC + 1)]
FACTORS = [f"FACTOR{i}" for i in range(1, N_FACTOR + 1)]
FEATURES = PCS + FACTORS

# planted truth for the stand-in: churn 3.6% if held back; the call cuts it by 0.6 points on average,
# more for customers with high PC2, and slightly raises it for some ("do not disturb")
TRUTH = {"control_rate": 0.036, "ate": -0.006}


def read_arff(path: Path) -> pd.DataFrame:
    names, start = [], None
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if s.upper().startswith("@ATTRIBUTE"):
                names.append(s.split()[1])
            elif s.upper().startswith("@DATA"):
                start = i + 1
                break
    df = pd.read_csv(path, skiprows=start, header=None, names=names, quotechar="'",
                     na_values=["?"], keep_default_na=False, low_memory=False)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FACTORS:
        df[c] = df[c].astype("string").fillna("missing").astype("category")
    df[TREAT] = df[TREAT].astype(int)
    df[OUTCOME] = df[OUTCOME].astype(int)
    return df


def cache_dir(source: str) -> Path:
    """Per-customer intermediate results (propensity scores) stay with the raw data, never committed."""
    import tempfile

    d = RAW.parent / "cache" if source == "real" else Path(tempfile.gettempdir()) / "telecom_cache_synthetic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def use_synthetic() -> bool:
    return os.environ.get("TELECOM_SYNTHETIC", "") in ("1", "true", "yes")


def load() -> tuple[pd.DataFrame, str]:
    """(data, "real" | "synthetic")."""
    if not use_synthetic():
        if RAW.exists():
            return prepare(read_arff(RAW)), "real"
        try:  # fetch from OpenML when the file was not downloaded by hand
            from sklearn.datasets import fetch_openml

            raw = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
            return prepare(raw), "real"
        except Exception:  # noqa: BLE001 - offline, blocked, or no OpenML: fall back to the stand-in
            pass
    return make_synthetic(), "synthetic"


def make_synthetic(n: int = 11_896, seed: int = 7, treat_share: float = 0.7574) -> pd.DataFrame:
    """Same columns and sizes, a randomized call and a known average effect."""
    rng = np.random.default_rng(seed)
    pcs = rng.normal(0, 1, (n, N_PC)) / np.sqrt(np.arange(1, N_PC + 1))[None, :] * 3
    df = pd.DataFrame(pcs, columns=PCS)
    for j, c in enumerate(FACTORS):
        k = 2 + j % 5
        df[c] = rng.choice([f"L{i}" for i in range(k)], n, p=np.ones(k) / k)
    t = (rng.random(n) < treat_share).astype(int)
    base = 1 / (1 + np.exp(-(np.log(TRUTH["control_rate"] / (1 - TRUTH["control_rate"])) - 0.45
                             + 0.35 * df["PC1"] + 0.25 * (df["FACTOR1"] == "L0"))))
    # effect: larger for high PC2; scaled so the population-average effect equals TRUTH["ate"]
    shape = 1 / (1 + np.exp(-1.5 * df["PC2"].to_numpy()))
    tau = TRUTH["ate"] * shape / shape.mean()
    p = np.clip(base + t * tau, 0, 1)
    df[TREAT], df[OUTCOME] = t, (rng.random(n) < p).astype(int)
    df["true_effect"] = tau
    return prepare(df)
