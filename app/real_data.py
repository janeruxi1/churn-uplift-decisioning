"""Churn and retention on real data -- the main page of the demo app (app/streamlit_app.py).

Four tabs, all read from the committed records in reports/metrics/ (neither dataset may be
redistributed, so the app ships aggregates only):
  1. Who will churn?         -- the KKBox model tested on March, and how many churners a contact list reaches
  2. Did the call work?      -- the telecom retention experiment: randomization check, effect, targeting
  3. Is an offer worth it?   -- the free-month decision across effects and margins
  4. Size a retention test   -- members per arm for a given churn rate and effect
The simulated first version is the app's second page (app/streamflix_app.py).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

REPO_URL = "https://github.com/janeruxi1/churn-uplift-decisioning"
METRICS = PROJECT_ROOT / "reports" / "metrics"
BLUE, ORANGE, GREEN, GREY = "#2C5282", "#DD6B20", "#2F855A", "#A0AEC0"
MODEL_NAMES = {"lightgbm": "LightGBM", "logistic": "Logistic regression", "rule": "Rule (auto-renew off or cancelled)"}



@st.cache_data
def record(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))


def md(text: str) -> str:
    """Escape dollar signs so Streamlit's markdown does not read them as LaTeX."""
    return text.replace("$", "\\$")


M1, M2, M3, M4 = (record(n) for n in ("01_kkbox_data_checks", "02_kkbox_churn_model", "03_kkbox_listening",
                                       "04_kkbox_lifetime_value"))
C5, C6, D7 = record("05_campaign_effect"), record("06_campaign_targeting"), record("07_retention_decision")

st.title("💸 Churn and retention on real data")
st.markdown("Who will churn, what are they worth, and is a retention offer worth making? KKBox music "
            "subscriptions (2015–2017) and a randomized retention campaign at a telecom operator (2020). "
            "The project's first version, on a simulated subscription business, is the **StreamFlix case study** "
            "page in the menu at the top.")
st.markdown(f"[Code and notebooks]({REPO_URL}) · [Decision memo]({REPO_URL}/blob/main/reports/decision_memo.md)")

tab_model, tab_call, tab_offer, tab_size = st.tabs(
    ["📉 Who will churn?", "📞 Did the call work?", "🎁 Is an offer worth it?", "📐 Size a retention test"])

# =====================================================================
with tab_model:
    ship = M2["ships"]
    m = M2["march"][ship]
    st.success(md(f"**The {MODEL_NAMES[ship]} model ships.** Trained on February 2017, tested once on March: PR-AUC "
                  f"{m['pr_auc']:.3f} against a {M2['mar_churn']:.1%} churn rate; the top 10% of scores hold "
                  f"{m['capture_10']:.0%} of the members who churned."))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PR-AUC on March", f"{m['pr_auc']:.3f}", f"base rate {M2['mar_churn']:.1%}", delta_color="off", border=True,
              help="How well the score ranks churners above non-churners when churners are rare. "
                   "Guessing at random would score about the churn rate.")
    k2.metric("ROC-AUC on March", f"{m['roc_auc']:.3f}", border=True,
              help="Chance that a random churner scores higher than a random member who stayed (0.5 = coin flip).")
    k3.metric("Churners in the top 10%", f"{m['capture_10']:.0%}", border=True)
    k4.metric("Label rebuilt from payments", f"{M1['label_agreement_feb']:.1%}", "agreement", delta_color="off", border=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("How many churners does a contact list reach?")
        reach = pd.DataFrame(D7["reach_curve"])
        share = st.slider("Contact the top … of members by churn score", 5, 50, 10, 5, format="%d%%", key="reach")
        hit = float(np.interp(share / 100, reach["share_contacted"], reach["churners_reached"]))
        line = alt.Chart(reach).mark_line(color=BLUE, strokeWidth=3).encode(
            x=alt.X("share_contacted:Q", title="members contacted, highest score first", axis=alt.Axis(format="%")),
            y=alt.Y("churners_reached:Q", title="churners reached", axis=alt.Axis(format="%")))
        rnd = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=GREY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        dot = alt.Chart(pd.DataFrame({"x": [share / 100], "y": [hit]})).mark_circle(size=160, color=ORANGE).encode(x="x:Q", y="y:Q")
        st.altair_chart((rnd + line + dot).properties(height=300), width="stretch")
        st.caption(f"Contacting the top {share}% reaches about {hit:.0%} of March churners (random contact would reach {share}%).")
    with c2:
        st.subheader("Models compared on March")
        tbl = pd.DataFrame(M2["march"]).T[["pr_auc", "roc_auc", "brier", "capture_10"]]
        tbl.columns = ["PR-AUC", "ROC-AUC", "Brier", "Top 10%"]
        tbl.index = [MODEL_NAMES[i] for i in tbl.index]
        st.dataframe(tbl.style.format({"PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Brier": "{:.4f}",
                                       "Top 10%": "{:.0%}"}), width="stretch")
        st.caption("Higher is better except Brier (probability error, lower is better). "
                   "Top 10%: share of March churners in the highest-scored 10% of members.")
        b = M2["lightgbm_minus_logistic"]
        st.caption(f"LightGBM minus logistic PR-AUC: {b['diff']:+.3f} (95% CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f}).")
        st.markdown(md(f"**Listening adds little:** {M3['listening_minus_payments']['diff']:+.3f} PR-AUC within March. "
                       f"**Worth keeping:** NT${M4['ltv_mean']:,.0f} of margin per member on average over 24 months."))
        st.warning(f"Churn rose from {M2['feb_churn']:.1%} in February to {M2['mar_churn']:.1%} in March; a model "
                   "calibrated on February predicts too little churn for March, so calibration needs a monthly check.")

# =====================================================================
with tab_call:
    rand, dr, raw = C5["randomization"], C5["effect_aipw"], C5["effect_unadjusted"]
    st.markdown(f"{C5['n']:,} high-risk customers of a telecom operator; {C5['share_called']:.0%} were called with a "
                "retention offer, the rest held back. Churn measured over the next two months.")
    st.error(f"**The randomization check fails:** a model predicts who was called from the customer profile with "
             f"AUC {rand['auc']:.2f} (random would be about 0.50; permutation p = {rand['p_value']:.3f}). "
             "The effect is therefore estimated with propensity weighting and a doubly robust estimator.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Effect of the call (doubly robust)", f"{dr['diff'] * 100:+.2f} pts",
              f"95% CI {dr['ci'][0] * 100:+.2f} to {dr['ci'][1] * 100:+.2f}", delta_color="off", border=True)
    k2.metric("Smallest effect the test could detect", f"{C5['mde_80'] * 100:.1f} pts", border=True)
    k3.metric("Largest imbalance after weighting", f"{C5['balance']['max_smd_after']:.2f} SD",
              f"from {C5['balance']['max_smd_before']:.2f} SD", delta_color="off", border=True)
    est = pd.DataFrame([
        {"estimate": "doubly robust (primary)", "d": dr["diff"], "lo": dr["ci"][0], "hi": dr["ci"][1]},
        {"estimate": "inverse-propensity weighted", "d": C5["effect_ipw"]["diff"], "lo": C5["effect_ipw"]["ci"][0], "hi": C5["effect_ipw"]["ci"][1]},
        {"estimate": "unadjusted", "d": raw["diff"], "lo": raw["ci"][0], "hi": raw["ci"][1]}])
    est[["d", "lo", "hi"]] *= 100
    base = alt.Chart(est).encode(y=alt.Y("estimate:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)))
    chart = (alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY).encode(x="x:Q")
             + base.mark_rule(strokeWidth=4, color=BLUE).encode(x=alt.X("lo:Q", title="change in two-month churn (points), 95% CI"), x2="hi:Q")
             + base.mark_circle(size=140, color=BLUE).encode(x="d:Q", tooltip=["estimate", alt.Tooltip("d:Q", format="+.2f")]))
    st.altair_chart(chart.properties(height=160), width="stretch")
    st.subheader("Who should be called?")
    q = pd.DataFrame(C6["summary"]).set_index("ranking").loc[["churn_risk", "t_learner", "x_learner"]]
    q.index = ["churn risk", "T-learner (uplift)", "X-learner (uplift)"]
    st.dataframe(pd.DataFrame({"Qini": q["qini"], "95% CI low": q["qini_2.5"], "95% CI high": q["qini_97.5"]})
                 .style.format("{:+.5f}"), width="stretch")
    st.caption("Qini above zero means a ranking finds customers the call helps better than random calling. "
               f"Recommendation under the plan's rule: {C6['recommendation']}.")

# =====================================================================
with tab_offer:
    R = D7["relative_reduction"]
    st.markdown(md("A free month for members likely to churn: worth it if it keeps enough of them. Expected value per "
                   "member = P(churn) × relative reduction × lifetime value − one free month for everyone who stays. "
                   f"The telecom experiment's effect, as a relative reduction: **{R['point']:.0%}** "
                   f"(95% CI {R['low']:.0%} to {R['high']:.0%})."))
    grid = pd.DataFrame(D7["value_grid"])
    c1, c2 = st.columns(2)
    r = c1.slider("Relative churn reduction from the offer", -0.2, 0.6, float(np.round(R["point"] / 0.02) * 0.02), 0.02,
                  format="%.2f", key="offer_r")
    margin = c2.select_slider("Gross margin", options=[0.2, 0.3, 0.4], value=0.3, key="offer_margin")
    row = grid[(grid["margin"] == margin) & np.isclose(grid["relative_reduction"], r)].iloc[0]
    k1, k2 = st.columns(2)
    k1.metric("Members worth an offer (March)", f"{int(row['members']):,}", border=True)
    k2.metric("Net value of the best policy", md(f"NT${row['net_value_ntd'] / 1e6:.2f}M"), border=True)
    sub = grid[grid["margin"] == margin].assign(value=lambda d: d["net_value_ntd"] / 1e6)
    band = alt.Chart(pd.DataFrame({"lo": [R["low"]], "hi": [R["high"]]})).mark_rect(color=ORANGE, opacity=0.15).encode(x="lo:Q", x2="hi:Q")
    line = alt.Chart(sub).mark_line(color=BLUE, strokeWidth=3).encode(
        x=alt.X("relative_reduction:Q", title="relative churn reduction", axis=alt.Axis(format="%")),
        y=alt.Y("value:Q", title="net value (NT$ million, one month)"))
    dot = alt.Chart(pd.DataFrame({"x": [r], "y": [row["net_value_ntd"] / 1e6]})).mark_circle(size=160, color=ORANGE).encode(x="x:Q", y="y:Q")
    st.altair_chart((band + line + dot).properties(height=300), width="stretch")
    t = D7["test_size"]
    if "n_per_arm" in t:
        st.info(f"The shaded band is the experiment's 95% interval: it includes no effect. {t['members']:,} high-risk "
                f"members could plausibly be worth an offer; a KKBox test needs {t['n_per_arm']:,} of them per arm to "
                f"detect the {t['relative_effect']:.0%} reduction they need to break even.")

# =====================================================================
with tab_size:
    st.markdown("How many members per arm does a retention test need? Two-sided 5% test, 80% power.")
    c1, c2 = st.columns(2)
    p0 = c1.slider("Churn rate without the offer", 0.01, 0.80, float(round(D7["test_size"].get("target_churn", 0.09), 2)),
                   0.01, format="%.2f", key="size_p0")
    rel = c2.slider("Relative reduction to detect", 0.05, 0.60, 0.20, 0.05, format="%.2f", key="size_rel")
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    p1 = p0 * (1 - rel)
    n = int(np.ceil(2 * z ** 2 * p0 * (1 - p0) / (p0 - p1) ** 2))   # same formula as notebook 07
    st.metric("Members per arm", f"{n:,}", f"churn {p0:.0%} → {p1:.1%}", delta_color="off", border=True)

st.divider()
st.caption(f"Read from the committed records in reports/metrics/ · [Decision memo]({REPO_URL}/blob/main/reports/decision_memo.md) "
           f"· [Source on GitHub]({REPO_URL})")
