# Decision memo: who will churn, and is a retention offer worth it?

**Author:** Xi Ru · **Data:** KKBox subscriptions (Kaggle, 2015–2017); a randomized retention
campaign at a Belgian telecom operator (OpenML 45580, 2020) · Every number below is read from
`reports/metrics/` and checked by `tests/test_reported_numbers.py`.

---

## Recommendation

**Do not roll out a retention offer on borrowed evidence. Use the churn model to find the members
at risk, and run a small KKBox test of a free-month offer on the highest-risk members first.**

- The churn model is good and ready: on a later month it was never tuned on, its top 10% of
  scores hold **60%** of the members who churned.
- The only real evidence on what a retention contact does (a randomized telecom campaign) shows
  **no detectable effect** on churn, and no way of choosing whom to call that beats calling at
  random.
- At that campaign's best estimate, an offer would pay for about 11,000 of 971,000 members a
  month, but its interval includes no effect at all. A test needs only a few hundred members per
  arm among the 33,000 highest-risk members to settle it.

## Part 1 · KKBox: who will churn, and what are they worth?

The analysis plan ([`kkbox_analysis_plan.md`](./kkbox_analysis_plan.md)) was committed before
the data was downloaded.

**The data can be trusted.** KKBox's churn label (no new paid subscription within 30 days of
expiry) was rebuilt from the raw payments for the February 2017 cohort and agrees **97.6%** of the
time. Every feature uses only payments dated before its cohort's cutoff, checked by rebuilding
the features with all later rows deleted; that check found, and the code now fixes, 23 members
whose "latest payment" was a tie picked at random.

**The model.** Trained and tuned on February 2017 (6.4% churn), then scored once on March 2017:

| Model | PR-AUC (March, 9.0% churn) | ROC-AUC | Churners in the top 10% |
|---|---|---|---|
| **LightGBM (ships)** | **0.594** | 0.863 | **60%** |
| Logistic regression | 0.503 | 0.846 | 56% |
| Rule: auto-renew off or last payment cancelled | 0.321 | 0.784 | 50% |

LightGBM beats logistic regression by **+0.092** PR-AUC (95% CI +0.090 to +0.094), so it ships
under the plan's rule. Auto-renew status is the strongest signal: February members with
auto-renew off churned at **33%**, against **3.0%** with it on. SHAP values on 20,000 March
members agree (added after the primary analysis): auto-renew accounts for **25%** of the model's
total |SHAP|, then payment method (14%), list price (9%) and tenure (8%); a recent cancellation
adds 4%.

**One thing to watch.** Churn rose from 6.4% in February to 9.0% in March, and the model,
calibrated on February, predicts **6.5%** on average for March. It ranks members well but its
probabilities run low when churn rises, so a deployed model needs its calibration checked every
month.

**Where it is weaker, and what moved** (diagnostics added after the primary analysis). Among
auto-renew members, 88% of March, churn is 4.5% and PR-AUC **0.338**: still 7.5 times better
than picking at random inside the group, but these are the hardest churners to find. Every
segment is under-predicted, most of all members on longer plans (51% predicted against 83%
observed). Male and female members are ranked equally well (PR-AUC 0.618 and 0.623). From
February to March the score's own distribution barely moved (PSI **0.030**) while churn rose,
so a monitor that watches only scores would have missed the drift; the features that moved most
are tenure (PSI 0.70) and the number of payments in the last 30 days (0.48).

**Listening adds little.** Within March, members who did not listen at all in the 14 days before
expiry churned at 9.3%, against 8.9% for those who listened almost daily; adding listening
features moves PR-AUC by only **+0.011**. Payment behaviour is the signal.

**What a member is worth.** From 2.95 million real subscription spells, a March member is worth
**NT$687** of margin on average over 24 months (17.6 expected paid months at a median NT$151 a
month, 30% assumed margin).

## Part 2 · A real retention campaign: did calling customers keep them?

11,896 high-risk telecom customers; 76% were called with a retention offer, 24% held back. Plan:
[`telecom_analysis_plan.md`](./telecom_analysis_plan.md).

**The "randomized" campaign was not randomized the same way for everyone.** A model predicts who
was called from the customer profile with an AUC of **0.64** (permutation p = 0.001), and the
share called ranges from 64% to 83% across customer segments. The authors' own check, which
compares a classifier with always guessing "called", misses this. A dated note in the plan
switched the analysis, before any adjusted result was computed, to a propensity-adjusted design;
after weighting, the largest imbalance on any feature falls from **0.35** to **0.09** standard
deviations.

**The effect: none detectable.**

| Estimate | Change in two-month churn (points, 95% CI) |
|---|---|
| **Doubly robust (primary)** | **−0.49** [−1.49, +0.51] |
| Unadjusted (reference) | −0.28 [−1.10, +0.46] |

The experiment could detect a change of about 1.1 points; it rules out the call cutting churn by
more than 1.5 points.

**How much hidden confounding would change that** (added after the primary analysis). As a risk
ratio the effect is 0.86 [0.59, 1.14]. An unmeasured factor tied to both being called and
churning by a risk ratio of **1.58** could have produced the estimate from no effect at all; to
hide a real reduction as large as the interval allows, it would need **2.29**, on top of the 160
profile components already adjusted for. The estimate covers the 97.7% of customers inside the
propensity overlap region.

**Targeting doesn't help either.** Neither ranking by churn risk nor two uplift models (T- and
X-learner) beats calling at random within the eligible group.

## Part 3 · The decision

Carrying the campaign's relative effect (**13.5%** fewer churners, 95% CI from 14% more to 41%
fewer) to a free-month offer at KKBox is an assumption, so the decision is reported across the
interval:

- **At the point estimate**, the offer pays for **10,722** March members, worth about
  **NT$0.57M** net for the month. At the bottom of the interval it pays for no one.
- **Break-even:** even the most promising 1% of members need the offer to cut their churn by
  **12%** to pay for itself, because the free month is also given to members who would have
  stayed anyway.
- **The test that settles it:** **33,333** members could plausibly be worth an offer (their
  break-even lies inside the interval); they churn at **61.6%**. Detecting the 20% reduction they
  need takes **256** members per arm.

## Limits

- KKBox data is from 2015–2017 and the margin (30%) is an assumption; lifetime value beyond 24
  months of history is extrapolated.
- The telecom effect assumes the customer features capture everything that decided who was held
  back, which the data cannot confirm; and a telecom call is not a music-service offer.
- The simulated first version of this project, where the true effects are known, is kept as the
  [StreamFlix appendix](../notebooks/streamflix/README.md).
