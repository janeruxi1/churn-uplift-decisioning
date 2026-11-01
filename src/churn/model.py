"""Churn model helpers for the KKBox cohorts: design matrices, metrics and the paired bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data.kkbox import CATEGORICAL, NUMERIC


class Prep:
    """Fitted on the training cohort only: winsorizing bounds, medians, frequent category levels."""

    def __init__(self, min_count: int = 500):
        self.min_count = min_count

    def fit(self, df: pd.DataFrame) -> "Prep":
        num = df[NUMERIC].astype(float)
        self.lo, self.hi = num.quantile(0.01), num.quantile(0.99)
        self.med = num.clip(self.lo, self.hi, axis=1).median()
        self.levels = {c: [v for v, n in df[c].fillna("missing").value_counts().items() if n >= self.min_count]
                       for c in CATEGORICAL}
        return self

    def _cat(self, df):
        return {c: df[c].fillna("missing").where(df[c].fillna("missing").isin(self.levels[c]), "other")
                for c in CATEGORICAL}

    def linear(self, df: pd.DataFrame, scale: tuple | None = None):
        """Standardized, winsorized numeric + missing flags + one-hot categories (for logistic regression)."""
        num = df[NUMERIC].astype(float).clip(self.lo, self.hi, axis=1)
        miss = num.isna().astype(float).add_suffix("_missing")
        miss = miss.loc[:, [c for c in miss.columns if c.replace("_missing", "") in ("age", "reg_days", "plan_days")]]
        num = num.fillna(self.med)
        if scale is None:
            scale = (num.mean(), num.std().replace(0, 1))
        num = (num - scale[0]) / scale[1]
        cats = pd.DataFrame(self._cat(df))
        dummies = pd.concat([pd.get_dummies(cats[c].astype(pd.CategoricalDtype(self.levels[c] + ["other"])),
                                            prefix=c, dtype=float) for c in CATEGORICAL], axis=1)
        return pd.concat([num, miss, dummies], axis=1).to_numpy(np.float32), scale

    def trees(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw numeric (trees handle scale and missing) + categorical dtype (for LightGBM)."""
        out = df[NUMERIC].astype(float).copy()
        for c, v in self._cat(df).items():
            out[c] = v.astype(pd.CategoricalDtype(self.levels[c] + ["other"]))
        return out


def capture(y: np.ndarray, p: np.ndarray, share: float) -> float:
    """Share of all churners among the top `share` of members by score."""
    k = int(round(share * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "capture_5": capture(y, p, 0.05), "capture_10": capture(y, p, 0.10), "capture_20": capture(y, p, 0.20)}


def paired_bootstrap_pr_auc(y, p_a, p_b, n: int = 1000, seed: int = 0) -> dict:
    """PR-AUC(a) - PR-AUC(b) on the same resampled members (multinomial weights)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        w = rng.multinomial(len(y), np.full(len(y), 1 / len(y))) if len(y) < 50_000 else rng.poisson(1.0, len(y))
        diffs[i] = average_precision_score(y, p_a, sample_weight=w) - average_precision_score(y, p_b, sample_weight=w)
    point = average_precision_score(y, p_a) - average_precision_score(y, p_b)
    return {"diff": float(point), "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "draws": n}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    return pd.DataFrame({"predicted": p, "observed": y, "bin": q}).groupby("bin").mean()
