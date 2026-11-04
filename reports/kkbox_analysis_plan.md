# Analysis plan: KKBox subscriber churn

Status: **written and committed before the KKBox data was downloaded or any label was read.**
Any later change is a new dated section at the end, not an edit.

## The data

KKBox, a Taiwanese music-streaming service, released subscription data for the WSDM 2018 churn
challenge on Kaggle: payments (`transactions`), member attributes (`members_v3`), daily listening
(`user_logs_v2`, March 2017 only) and churn labels for two cohorts:

| Cohort | File | Who |
|---|---|---|
| February 2017 | `train.csv` | members whose subscription expired in February 2017 |
| March 2017 | `train_v2.csv` | members whose subscription expired in March 2017 |

**Churn** (KKBox's definition): no new valid subscription within 30 days after the current one
expires. The data is Kaggle competition data, so it is not committed; CI runs on a synthetic
stand-in with the same columns.

## The decision

KKBox can offer a retention incentive (a free extra month) to members before their subscription
expires, on a fixed monthly budget. Who should get it? This needs, per member: the probability of
churning, what keeping them is worth, and the effect of the offer. This plan covers the first two;
the offer's effect comes from a real randomized retention campaign (see
`telecom_analysis_plan.md`), because KKBox ran no experiment.

## Checks before modeling (a failure stops the analysis)

1. **The label can be reproduced.** Rebuild churn from the raw transactions (no new
   non-cancelled subscription starting within 30 days of expiry) and report the agreement with
   KKBox's label for each cohort. Below 95% agreement, the disagreement is investigated and
   reported before any model is fitted.
2. **No future data.** Every feature for a cohort uses only rows dated before the cohort's
   cutoff: **1 February 2017** for the February cohort, **1 March 2017** for the March cohort. A
   test rebuilds the features with all later rows deleted and requires identical values.
3. **Data quality:** duplicate members, impossible ages (`bd` outside 10–80 is treated as
   missing), negative or zero payments, expiry dates before transaction dates; counts reported.

## Features (fixed now)

From payments before the cutoff: plan length, list price, amount paid and discount, auto-renew
flag, cancellation flag on the latest transaction, number of transactions and cancellations in
the last 30/90/365 days, months as a member, days from the cutoff to the current expiry, payment
method. From members: city, age (cleaned), gender, registration channel, registration date.

## The model

- **Train on the February cohort, test once on the March cohort.** This is the out-of-time test
  a real deployment faces. Hyperparameters are tuned by 5-fold cross-validation inside February
  only. The March labels are read once, by the final evaluation.
- **Candidates:** a rule baseline (churn if auto-renew is off or the latest transaction was a
  cancellation), logistic regression (standardized features, one-hot categories), and LightGBM.
- **Primary metric:** PR-AUC on March. Also ROC-AUC, Brier score, calibration curve, and the
  share of churners caught in the top 5%, 10% and 20% of scores.
- **Which model ships:** LightGBM ships only if its March PR-AUC beats logistic regression with
  a paired bootstrap 95% interval (1,000 resamples of members) that excludes zero; otherwise
  logistic regression ships. Calibration is fixed by isotonic regression fitted on a held-out
  20% of February if the Brier score improves there.
- **Listening (secondary, reported separately):** `user_logs_v2` covers March only, after the
  March cutoff, so it cannot enter the out-of-time model. A separate analysis asks whether
  listening in the 14 days before each March member's expiry adds to the payment-only model,
  with a random 50/50 split of the March cohort, and is labelled as a within-month result.

## What a member is worth

- **Lifetime value from real payments:** subscription spells from the transaction history;
  Kaplan–Meier survival in months by plan length and auto-renew status; value = expected
  remaining paid months over a **24-month horizon** times the member's monthly net payment times
  an assumed **30% gross margin** (music licensing takes most of a subscription). No discounting.
  Sensitivity: 12 and 36 months, 20% and 40% margin.

## The contact policy (applied in Part 3)

Expected value of an offer = P(churn) × relative churn reduction × lifetime value − cost of the
offer. The offer is one free month, costing one month of the member's net payment when it is
taken. Members are contacted in order of expected value until the budget runs out, and only while
expected value is positive. The churn reduction comes from the telecom experiment's measured
effect and its interval; the transfer from a telecom call-centre offer to a music service is an
assumption, so the decision is also reported as the **break-even** churn reduction.

## What would change the plan

Nothing in the March labels. If a file's columns differ from the description above, the change
is recorded here before any label is read.

## Note added 2 November 2026 (after the primary analysis)

Three diagnostics were added to notebook 02, after March had been scored: SHAP contributions of
the LightGBM model on a sample of March members, performance and calibration within segments
(auto-renew, plan, tenure, gender, registration channel), and feature and score drift from
February to March (population stability index). They are descriptive: they do not change the
model, the calibration decision or which model ships, and they are recorded separately
(`02b_kkbox_model_diagnostics.json`). Model runs are also logged to MLflow from this date.
