"""Campaign readout tools (src/experiment/campaign.py) on cases with a known answer."""
import numpy as np
import pytest

from src.data import telecom
from src.experiment import campaign


def test_newcombe_matches_a_textbook_example():
    # Newcombe (1998), example (a): 56/70 vs 48/80 -> difference 0.2, interval 0.0524 to 0.3339
    e = campaign.newcombe(56, 70, 48, 80)
    assert e.diff == pytest.approx(0.2)
    assert e.ci[0] == pytest.approx(0.0524, abs=5e-4) and e.ci[1] == pytest.approx(0.3339, abs=5e-4)


def test_mde_shrinks_with_sample_size():
    assert campaign.mde(0.036, 9010, 2886) == pytest.approx(0.011, abs=0.001)
    assert campaign.mde(0.036, 40000, 40000) < campaign.mde(0.036, 9010, 2886)


def test_randomization_check_passes_random_and_fails_targeted_assignment():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 10))
    random_t = (rng.random(3000) < 0.75).astype(int)
    targeted_t = (rng.random(3000) < 1 / (1 + np.exp(-(1 + 1.2 * X[:, 0])))).astype(int)
    assert campaign.randomization_check(X, random_t, permutations=30)["passed"]
    assert not campaign.randomization_check(X, targeted_t, permutations=199)["passed"]   # p can reach 0.005


def test_aipw_recovers_the_effect_when_assignment_depends_on_a_feature():
    """Called customers are riskier by design; the naive comparison is biased, AIPW is not."""
    rng = np.random.default_rng(1)
    n = 20000
    X = rng.normal(size=(n, 5))
    e_true = 1 / (1 + np.exp(-(0.8 + 1.0 * X[:, 0])))
    t = (rng.random(n) < e_true).astype(int)
    p0 = 1 / (1 + np.exp(-(-2.5 + 0.9 * X[:, 0])))
    y = (rng.random(n) < p0 - 0.03 * t).astype(int)          # true effect -3 points for everyone
    naive = y[t == 1].mean() - y[t == 0].mean()
    e = campaign.propensity_oof(X, t)
    mu0, mu1 = campaign.outcome_oof(X, t, y, params={"n_estimators": 150})
    dr = campaign.aipw(t, y, e, mu0, mu1)
    assert naive > 0                                           # confounding makes the call look harmful
    assert dr["ci"][0] < -0.03 < dr["ci"][1]


def test_qini_is_near_zero_for_random_and_positive_for_the_true_effect():
    s = telecom.make_synthetic(n=20000, seed=3)
    t, y = s["t"].to_numpy(), s["y"].to_numpy()
    benefit = -s["true_effect"].to_numpy()                     # larger = call helps more
    rand = np.random.default_rng(0).random(len(t))
    assert abs(campaign.qini_coefficient(rand, t, y)) < 0.002
    assert campaign.qini_coefficient(benefit, t, y) > campaign.qini_coefficient(rand, t, y)


def test_weighted_smd_is_zero_for_identical_arms():
    X = np.vstack([np.eye(3), np.eye(3)])
    t = np.array([1, 1, 1, 0, 0, 0])
    assert np.allclose(campaign.weighted_smd(X, t), 0)


def test_synthetic_stand_in_has_the_real_shape():
    s = telecom.make_synthetic()
    assert s.shape[0] == 11_896 and set(telecom.FEATURES) <= set(s.columns)
    assert s["t"].mean() == pytest.approx(0.7574, abs=0.01)
    assert s["true_effect"].mean() == pytest.approx(telecom.TRUTH["ate"])
