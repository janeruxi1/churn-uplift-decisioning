"""Every headline number in the README and the decision memo is read from the recorded results.

Each check formats a value from reports/metrics/ the way the docs write it and asserts the text
appears there. If a notebook's result changes, these fail until the docs are updated.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = {n: (ROOT / p).read_text(encoding="utf-8") for n, p in {"readme": "README.md", "memo": "reports/decision_memo.md"}.items()}


def rec(name):
    return json.loads((ROOT / "reports" / "metrics" / f"{name}.json").read_text(encoding="utf-8"))


K1, K2, K3, K4 = rec("01_kkbox_data_checks"), rec("02_kkbox_churn_model"), rec("03_kkbox_listening"), rec("04_kkbox_lifetime_value")
T5, T6, D7 = rec("05_campaign_effect"), rec("06_campaign_targeting"), rec("07_retention_decision")
SHIP = K2["march"][K2["ships"]]
LR, RULE = K2["march"]["logistic"], K2["march"]["rule"]
DR, RAW = T5["effect_aipw"], T5["effect_unadjusted"]
R = D7["relative_reduction"]
PT = next(r for r in D7["policy"] if r["effect"] == "point")
LOW = next(r for r in D7["policy"] if r["effect"] == "low")


def pts(x):
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.2f}"


CLAIMS = [
    ("memo", f"**{K1['label_agreement_feb']:.1%}**"),
    ("readme", f"agrees {K1['label_agreement_feb']:.1%}"),
    ("both", f"{SHIP['pr_auc']:.3f}"),
    ("memo", f"| **{SHIP['pr_auc']:.3f}** | {SHIP['roc_auc']:.3f} | **{SHIP['capture_10']:.0%}** |"),
    ("memo", f"| {LR['pr_auc']:.3f} | {LR['roc_auc']:.3f} | {LR['capture_10']:.0%} |"),
    ("memo", f"| {RULE['pr_auc']:.3f} | {RULE['roc_auc']:.3f} | {RULE['capture_10']:.0%} |"),
    ("both", f"{K2['mar_churn']:.1%}"),
    ("both", f"{K2['feb_churn']:.1%}"),
    ("both", f"{SHIP['capture_10']:.0%}"),
    ("memo", f"**+{K2['lightgbm_minus_logistic']['diff']:.3f}** PR-AUC (95% CI +{K2['lightgbm_minus_logistic']['ci'][0]:.3f} to "
             f"+{K2['lightgbm_minus_logistic']['ci'][1]:.3f})"),
    ("readme", f"+{K2['lightgbm_minus_logistic']['diff']:.3f} PR-AUC"),
    ("memo", f"**{K1['churn_by_auto_renew_feb']['0']:.0%}**, against **{K1['churn_by_auto_renew_feb']['1']:.1%}**"),
    ("both", f"+{K3['listening_minus_payments']['diff']:.3f}"),
    ("memo", f"churned at {K3['churn_by_days_active']['0']:.1%}, against {K3['churn_by_days_active']['11–14']:.1%}"),
    ("both", f"NT${K4['ltv_mean']:,.0f}"),
    ("memo", f"at a median NT${K4['median_monthly_paid']:.0f} a"),
    ("readme", f"AUC **{T5['randomization']['auc']:.2f}**"),
    ("memo", f"AUC of **{T5['randomization']['auc']:.2f}** (permutation p = {T5['randomization']['p_value']:.3f})"),
    ("memo", f"from **{T5['balance']['max_smd_before']:.2f}** to **{T5['balance']['max_smd_after']:.2f}**"),
    ("memo", f"| **{pts(DR['diff'])}** [{pts(DR['ci'][0])}, {pts(DR['ci'][1])}] |"),
    ("memo", f"| {pts(RAW['diff'])} [{pts(RAW['ci'][0])}, {pts(RAW['ci'][1])}] |"),
    ("readme", f"{pts(DR['diff'])} points of churn (95% CI {pts(DR['ci'][0])} to {pts(DR['ci'][1])})"),
    ("memo", f"about {T5['mde_80'] * 100:.1f} points"),
    ("memo", f"more than {-DR['ci'][0] * 100:.1f} points"),
    ("memo", f"**{R['point']:.1%}** fewer churners, 95% CI from {-R['low']:.0%} more to {R['high']:.0%}"),
    ("memo", f"**{PT['members_worth_offering']:,}** March members"),
    ("memo", f"**NT${PT['net_value_ntd'] / 1e6:.2f}M**"),
    ("memo", f"**{D7['breakeven']['top_1']:.0%}** to pay for itself"),
    ("memo", f"**{D7['test_size']['members']:,}** members"),
    ("memo", f"churn at **{D7['test_size']['target_churn']:.1%}**"),
    ("both", f"**{D7['test_size']['n_per_arm']:,}**"),
    ("memo", f"the {D7['test_size']['relative_effect']:.0%} reduction"),
]


@pytest.mark.parametrize("where,text", CLAIMS, ids=[c[1][:40] for c in CLAIMS])
def test_doc_quotes_the_recorded_result(where, text):
    for name in (["readme", "memo"] if where == "both" else [where]):
        assert text in DOCS[name], f"{name} should contain {text!r}"


def test_records_are_real_and_the_decisions_match():
    for r in (K1, K2, K3, K4, T5, T6, D7):
        assert r["source"] == "real"
    assert K1["no_future_data"] and K1["label_agreement_feb"] >= 0.95
    assert K2["ships"] == "lightgbm" and "LightGBM (ships)" in DOCS["memo"]
    assert not T5["randomization"]["passed"]
    assert DR["ci"][0] < 0 < DR["ci"][1] and "no detectable effect" in DOCS["memo"].lower()
    assert not any(T6["beats_risk"].values()) and not any(T6["better_than_random"].values())
    assert LOW["members_worth_offering"] == 0 and "pays for no one" in DOCS["memo"]
