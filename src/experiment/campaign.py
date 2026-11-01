"""Reading a randomized retention campaign: randomization check, effect, and targeting.

Outcomes are churn (1 = left). Effects are reported as the change in churn rate from calling
(called minus held back), so a negative number means the call kept customers. For targeting, every
score is oriented so that higher means "call this customer first".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

Z = stats.norm.ppf(0.975)


# ---------------------------------------------------------------- design matrix
def design(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    num = df[numeric].to_numpy(float)
    num = (num - num.mean(0)) / np.where(num.std(0) > 0, num.std(0), 1)
    cat = pd.get_dummies(df[categorical], drop_first=True, dtype=float).to_numpy() if categorical else np.empty((len(df), 0))
    return np.hstack([num, cat])


# ---------------------------------------------------------------- randomization
def oof_scores(X: np.ndarray, t: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold scores from a ridge classifier predicting treatment from the features."""
    out = np.empty(len(t))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, t):
        out[te] = RidgeClassifier(alpha=10.0).fit(X[tr], t[tr]).decision_function(X[te])
    return out


def randomization_check(X: np.ndarray, t: np.ndarray, permutations: int = 1000, seed: int = 0) -> dict:
    """Cross-validated AUC of predicting treatment, with a permutation p-value."""
    auc = roc_auc_score(t, oof_scores(X, t, seed=seed))
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(tp := rng.permutation(t), oof_scores(X, tp, seed=seed)) for _ in range(permutations)])
    p = (1 + np.sum(null >= auc)) / (1 + permutations)
    return {"auc": float(auc), "p_value": float(p), "null_95th": float(np.percentile(null, 95)),
            "permutations": permutations, "passed": bool(p >= 0.01)}


# ---------------------------------------------------------------- effect
def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


@dataclass
class Effect:
    rate_called: float
    rate_held_back: float
    diff: float
    ci: tuple[float, float]
    relative: float
    p_value: float
    n_called: int
    n_held_back: int

    def as_dict(self) -> dict:
        return {**self.__dict__, "ci": list(self.ci)}


def newcombe(k1: int, n1: int, k0: int, n0: int) -> Effect:
    """Difference in proportions (1 minus 0) with Newcombe's hybrid score interval (method 10)."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    lo = d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    pool = (k1 + k0) / (n1 + n0)
    z = d / np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
    return Effect(p1, p0, d, (float(lo), float(hi)), d / p0, float(2 * stats.norm.sf(abs(z))), n1, n0)


def mde(p0: float, n1: int, n0: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute change in a rate p0 detectable at the given power (normal approximation)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(p0 * (1 - p0) * (1 / n1 + 1 / n0)))


def select_prognostic(X: np.ndarray, y: np.ndarray, k: int = 20, seed: int = 0) -> np.ndarray:
    """Indices of the k features with the largest L1-logistic coefficients, the penalty chosen by
    5-fold cross-validation (log-loss), fitted on the held-back group only."""
    best, best_c = np.inf, None
    for c in (0.003, 0.01, 0.03, 0.1, 0.3):
        losses = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = LogisticRegression(penalty="l1", C=c, solver="liblinear").fit(X[tr], y[tr])
            p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
        if np.mean(losses) < best:
            best, best_c = np.mean(losses), c
    coef = LogisticRegression(penalty="l1", C=best_c, solver="liblinear").fit(X, y).coef_.ravel()
    order = np.argsort(-np.abs(coef))
    return order[:k][np.abs(coef[order[:k]]) > 0] if np.any(coef) else order[:k]


def adjusted_effect(X: np.ndarray, t: np.ndarray, y: np.ndarray, boot: int = 500, seed: int = 0) -> dict:
    """Covariate-adjusted change in churn: logistic regression of churn on the call and the
    features, averaged over all customers as if everyone / no one were called (g-computation),
    with a bootstrap interval."""
    def est(idx):
        Xi, ti, yi = X[idx], t[idx], y[idx]
        m = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack([ti, Xi]), yi)
        p1 = m.predict_proba(np.column_stack([np.ones(len(idx)), Xi]))[:, 1].mean()
        p0 = m.predict_proba(np.column_stack([np.zeros(len(idx)), Xi]))[:, 1].mean()
        return p1 - p0
    rng = np.random.default_rng(seed)
    point = est(np.arange(len(y)))
    draws = np.array([est(rng.integers(0, len(y), len(y))) for _ in range(boot)])
    return {"diff": float(point), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "se": float(draws.std(ddof=1))}


# ---------------------------------------------------------------- when assignment depends on the features
def propensity_oof(X: np.ndarray, t: np.ndarray, folds: int = 5, seed: int = 0, C: float = 0.1) -> np.ndarray:
    """Cross-fitted probability of being called: L2 logistic regression, out of fold."""
    e = np.empty(len(t))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, t):
        e[te] = LogisticRegression(C=C, max_iter=3000).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    return e


def outcome_oof(X: np.ndarray, t: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0,
                params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted churn predictions under each arm (mu0 held back, mu1 called)."""
    import lightgbm as lgb

    p = dict(LGBM_PARAMS, random_state=seed)
    p.update(params or {})
    mu0, mu1 = np.empty(len(t)), np.empty(len(t))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, t * 2 + y):
        c, tt = tr[t[tr] == 0], tr[t[tr] == 1]
        mu0[te] = lgb.LGBMClassifier(**p).fit(X[c], y[c]).predict_proba(X[te])[:, 1]
        mu1[te] = lgb.LGBMClassifier(**p).fit(X[tt], y[tt]).predict_proba(X[te])[:, 1]
    return mu0, mu1


def aipw(t, y, e, mu0, mu1) -> dict:
    """Doubly robust (augmented IPW) change in churn from calling, with an influence-function CI."""
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    est, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
    return {"diff": float(est), "ci": [float(est - Z * se), float(est + Z * se)], "se": float(se),
            "p_value": float(2 * stats.norm.sf(abs(est / se)))}


def hajek(t, y, e, boot: int = 1000, seed: int = 0) -> dict:
    """Inverse-probability-weighted change in churn (normalized weights), bootstrap CI."""
    def est(i):
        w1, w0 = t[i] / e[i], (1 - t[i]) / (1 - e[i])
        return (w1 * y[i]).sum() / w1.sum() - (w0 * y[i]).sum() / w0.sum()
    rng = np.random.default_rng(seed)
    point = est(np.arange(len(t)))
    draws = np.array([est(rng.integers(0, len(t), len(t))) for _ in range(boot)])
    return {"diff": float(point), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "se": float(draws.std(ddof=1))}


def ipw_weights(t, e) -> np.ndarray:
    return np.where(t == 1, 1 / e, 1 / (1 - e))


def weighted_smd(X: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Standardized mean difference of every column between arms (pooled unweighted SD)."""
    w = np.ones(len(t)) if w is None else w
    m1 = np.average(X[t == 1], axis=0, weights=w[t == 1])
    m0 = np.average(X[t == 0], axis=0, weights=w[t == 0])
    sd = np.sqrt((X[t == 1].var(0) + X[t == 0].var(0)) / 2)
    return np.abs(m1 - m0) / np.where(sd > 0, sd, 1)


# ---------------------------------------------------------------- targeting
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1)


def qini_curve(score: np.ndarray, t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative churn prevented when calling the top-k by `score` (higher = call first), in
    customers: held-back churners scaled to the called count, minus called churners. Optional
    weights (inverse propensity) reweight each arm."""
    order = np.argsort(-score, kind="stable")
    t, y = t[order], y[order]
    w = np.ones(len(t)) if w is None else w[order]
    nt, nc = np.cumsum(w * t), np.cumsum(w * (1 - t))
    yt, yc = np.cumsum(w * y * t), np.cumsum(w * y * (1 - t))
    prevented = np.where(nc > 0, yc * nt / np.maximum(nc, 1), 0.0) - yt
    frac = np.arange(1, len(t) + 1) / len(t)
    return frac, prevented


def qini_coefficient(score, t, y, w=None) -> float:
    """Area between the Qini curve and the random-targeting line, per customer."""
    frac, q = qini_curve(score, t, y, w)
    total = len(t) if w is None else w.sum()
    return float(np.mean(q - frac * q[-1]) / total)


def top_share_effect(score, t, y, share: float, w=None) -> float:
    """Churn prevented per customer called (held-back minus called churn rate) in the top share."""
    k = int(round(share * len(t)))
    top = np.argsort(-score, kind="stable")[:k]
    tt, yy = t[top], y[top]
    ww = np.ones(k) if w is None else w[top]
    return float(np.average(yy[tt == 0], weights=ww[tt == 0]) - np.average(yy[tt == 1], weights=ww[tt == 1]))


def bootstrap_targeting(scores: dict[str, np.ndarray], t, y, w=None, boot: int = 1000, seed: int = 0,
                        shares=(0.2, 0.4)) -> pd.DataFrame:
    """Qini coefficient and top-share effects for each ranking, on the same resampled customers."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(boot + 1):
        idx = np.arange(len(t)) if b == 0 else rng.integers(0, len(t), len(t))
        wi = None if w is None else w[idx]
        for name, s in scores.items():
            r = {"draw": b, "ranking": name, "qini": qini_coefficient(s[idx], t[idx], y[idx], wi)}
            for sh in shares:
                r[f"top_{int(sh * 100)}"] = top_share_effect(s[idx], t[idx], y[idx], sh, wi)
            rows.append(r)
    return pd.DataFrame(rows)


def cross_fit_scores(X: np.ndarray, t: np.ndarray, y: np.ndarray, propensity: float, folds: int = 5,
                     seed: int = 0, params: dict | None = None) -> dict[str, np.ndarray]:
    """Out-of-fold scores, each oriented so higher = call first:
    churn_risk (model of churn fitted on held-back customers), t_learner and x_learner (predicted
    reduction in churn from the call)."""
    import lightgbm as lgb

    p = dict(LGBM_PARAMS, random_state=seed)
    p.update(params or {})
    out = {k: np.empty(len(t)) for k in ("churn_risk", "t_learner", "x_learner")}
    strata = t * 2 + y
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, strata):
        c, tr_t = tr[t[tr] == 0], tr[t[tr] == 1]
        m0 = lgb.LGBMClassifier(**p).fit(X[c], y[c])
        m1 = lgb.LGBMClassifier(**p).fit(X[tr_t], y[tr_t])
        mu0, mu1 = m0.predict_proba(X[te])[:, 1], m1.predict_proba(X[te])[:, 1]
        out["churn_risk"][te] = mu0
        out["t_learner"][te] = mu0 - mu1
        # X-learner: imputed reductions, regressed on the features within each arm
        d1 = m0.predict_proba(X[tr_t])[:, 1] - y[tr_t]           # called: churn they would have had - churn they had
        d0 = y[c] - m1.predict_proba(X[c])[:, 1]                 # held back: churn they had - churn if called
        rp = {k: v for k, v in p.items() if k != "verbose"}
        r1 = lgb.LGBMRegressor(**rp, verbose=-1).fit(X[tr_t], d1)
        r0 = lgb.LGBMRegressor(**rp, verbose=-1).fit(X[c], d0)
        out["x_learner"][te] = propensity * r0.predict(X[te]) + (1 - propensity) * r1.predict(X[te])
    return out
