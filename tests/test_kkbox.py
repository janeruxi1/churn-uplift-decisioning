"""KKBox features, label and lifetime value on the synthetic stand-in, where the answer is known."""
import numpy as np
import pandas as pd
import pytest

from src.churn import ltv
from src.churn.model import capture, paired_bootstrap_pr_auc
from src.data import kkbox


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    raw = kkbox.make_synthetic(n=1500, seed=5, out=tmp_path_factory.mktemp("kkbox"))
    c, _ = kkbox.connect(raw)
    return c


def test_features_use_no_rows_after_the_cutoff(con):
    full = kkbox.features(con, "feb")
    con.execute(f"CREATE OR REPLACE TEMP TABLE tx_cut AS SELECT * FROM tx WHERE transaction_date < {kkbox.CUTOFF['feb']}")
    pd.testing.assert_frame_equal(full, kkbox.features(con, "feb", tx_table="tx_cut"), check_exact=False)


def test_features_are_deterministic_with_tied_payments(con):
    con.execute("CREATE OR REPLACE TEMP TABLE tx_tie AS SELECT * FROM tx UNION ALL "
                "SELECT msno, 99, payment_plan_days, plan_list_price, actual_amount_paid - 1, is_auto_renew, "
                "transaction_date, membership_expire_date, is_cancel FROM tx")
    a = kkbox.features(con, "feb", tx_table="tx_tie")
    b = kkbox.features(con, "feb", tx_table="tx_tie")
    pd.testing.assert_frame_equal(a, b)


def test_label_rebuilt_from_payments_matches(con):
    r = kkbox.rebuild_label(con, "feb").merge(kkbox.labels(con, "feb"), on="msno")
    assert (r["churn_rebuilt"] == r["is_churn"]).mean() > 0.98


def test_kaplan_meier_recovers_a_constant_hazard():
    rng = np.random.default_rng(0)
    life = rng.exponential(1 / 0.1, 50000)              # 10% a month
    cens_at = rng.uniform(0, 30, 50000)
    months, censored = np.minimum(life, cens_at), cens_at < life
    s = ltv.kaplan_meier(months, censored, horizon=12)
    assert s[12] == pytest.approx(np.exp(-0.1 * 12), abs=0.03)


def test_remaining_months_for_a_flat_curve_is_the_horizon():
    surv = np.ones(100)
    assert ltv.remaining_months(surv, np.array([0.0, 5.0]), 24) == pytest.approx([24, 24])


def test_capture_and_bootstrap():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0] * 50)
    perfect = y + np.random.default_rng(0).random(len(y)) * 0.1
    assert capture(y, perfect, 0.2) == 1.0
    b = paired_bootstrap_pr_auc(y, perfect, np.random.default_rng(1).random(len(y)), n=50)
    assert b["ci"][0] > 0
