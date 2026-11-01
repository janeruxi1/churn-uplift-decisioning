# Analysis plan: a randomized telecom retention campaign

Status: **written and committed before the data was downloaded or any outcome was read.** Any
later change is a new dated section at the end.

## The data

Verhelst, Mercier, Shrestha and Bontempi, "A churn prediction dataset from the telecom sector: a
new benchmark for uplift modeling" (2023); OpenML dataset 45580 (`churn-uplift-mlg`), licence
**CC BY-NC-ND 4.0**. The data is not committed: notebooks download it from OpenML, and CI uses a
synthetic stand-in.

As described by the authors: 11,896 customers of a European telecom judged at high risk of
churning, in three monthly campaigns between September and December 2020. Within that group a
random 24% were held back as control; the rest were called by a call centre with a retention
offer or a tariff recommendation. Churn is measured in the two months after the campaign.
178 features, anonymized (numeric ones by a PCA projection).

## The decision

Does a retention call reduce churn enough to pay for itself, and should calls go to the customers
the offer helps most (uplift targeting) rather than those most likely to churn?

## Checks first

1. **Randomization:** a classifier predicting treatment from the features, cross-validated;
   its AUC with a permutation p-value (1,000 shuffles). An AUC whose permutation p < 0.01 stops
   the readout. Arm sizes against the stated 76/24 split.
2. **Data:** missing values, duplicate rows, constant features.

## The effect

- **Primary:** churn rate in the called group minus the held-back group, with a 95% interval
  (Newcombe hybrid score) and the relative change. Also the regression-adjusted difference
  (logistic regression on the 20 features most predictive of churn among the held-back group,
  chosen by cross-validation inside that group only) as a precision check.
- **What it can detect:** the minimum detectable effect at 80% power for these arm sizes,
  reported next to the result, so "no detectable effect" says how big an effect is ruled out.

## Targeting

- **Candidates:** (a) churn score, a model of churn fitted on the held-back group only, so it
  measures risk without the call; (b) a T-learner and (c) an X-learner, both LightGBM; all
  cross-fitted with 5 folds so every customer is scored by a model that did not see them.
- **Metric:** Qini coefficient and the uplift in the top 20% / 40% of each ranking, with a
  bootstrap 95% interval (1,000 resamples of customers), against random targeting.
- **Decision rule:** uplift targeting is recommended only if its Qini beats the churn score with
  a paired bootstrap interval excluding zero. Otherwise the recommendation is "call by risk" or,
  if no ranking beats random, "call at random within the eligible group".

## What would change the plan

Nothing in the outcomes. If the downloaded file's columns or sizes differ from the description,
that is recorded here before any outcome is read.
