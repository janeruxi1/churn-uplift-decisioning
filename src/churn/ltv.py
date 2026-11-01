"""Lifetime value from real payments: subscription spells, Kaplan-Meier survival by plan type and
auto-renew, and expected remaining paid months."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH = 30.4


def spells(con, before: int) -> pd.DataFrame:
    """Continuous subscription spells from non-cancelled payments dated before `before` (a gap of
    more than 30 days after the paid-through date starts a new spell). One row per spell: start,
    paid-through date, months, whether it is still running at `before` (censored), plan group."""
    return con.sql(f"""
        WITH p AS (SELECT msno, strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d') AS td,
                          strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS ed,
                          payment_plan_days, is_auto_renew, actual_amount_paid
                   FROM tx WHERE is_cancel = 0 AND transaction_date < {before}
                     AND membership_expire_date >= transaction_date AND membership_expire_date < 20200101),
        r AS (SELECT *, max(ed) OVER (PARTITION BY msno ORDER BY td, ed ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_ed
              FROM p),
        s AS (SELECT *, sum(CASE WHEN prev_ed IS NULL OR td > prev_ed + INTERVAL 30 DAY THEN 1 ELSE 0 END)
                          OVER (PARTITION BY msno ORDER BY td, ed ROWS UNBOUNDED PRECEDING) AS spell
              FROM r)
        SELECT msno, spell, min(td) AS start, max(ed) AS paid_through,
               date_diff('day', min(td), max(ed)) / {MONTH} AS months,
               max(ed) >= strptime('{before}', '%Y%m%d') - INTERVAL 1 DAY AS censored,
               arg_max(CASE WHEN payment_plan_days <= 31 THEN 'monthly' ELSE 'longer' END, td) AS plan,
               arg_max(is_auto_renew, td) AS auto_renew,
               arg_max(actual_amount_paid * {MONTH} / nullif(payment_plan_days, 0), td) AS monthly_paid
        FROM s GROUP BY 1, 2""").df()


def kaplan_meier(months: np.ndarray, censored: np.ndarray, horizon: int = 60) -> np.ndarray:
    """S(k) for k = 0..horizon whole months: share of spells still running after k months."""
    m = np.floor(months).astype(int)
    events = np.bincount(m[~censored].clip(0, horizon), minlength=horizon + 1)
    cens = np.bincount(m[censored].clip(0, horizon), minlength=horizon + 1)
    at_risk = len(m) - np.concatenate([[0], np.cumsum(events + cens)[:-1]])
    hazard = np.where(at_risk > 0, events / np.maximum(at_risk, 1), 0)
    return np.concatenate([[1.0], np.cumprod(1 - hazard)])[: horizon + 1]


def extend(surv: np.ndarray, observed: int, horizon: int, tail: int = 6) -> np.ndarray:
    """Beyond `observed` months, continue with the average monthly hazard of the last `tail` months."""
    s = surv[: observed + 1].copy()
    h = 1 - (s[observed] / s[observed - tail]) ** (1 / tail)
    extra = s[observed] * (1 - h) ** np.arange(1, horizon - observed + 1)
    return np.concatenate([s, extra]) if horizon > observed else s[: horizon + 1]


def remaining_months(surv: np.ndarray, age: np.ndarray, horizon: int) -> np.ndarray:
    """Expected paid months over the next `horizon` months for spells already `age` months old."""
    a = np.clip(np.floor(age).astype(int), 0, len(surv) - horizon - 1)
    idx = a[:, None] + np.arange(1, horizon + 1)[None, :]
    return (surv[idx] / surv[a][:, None]).sum(axis=1)
