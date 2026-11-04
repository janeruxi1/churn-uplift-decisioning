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

## 2026-11-01 · The randomization check failed; how the analysis continues

**What happened.** The downloaded file matches the description (11,896 rows; 160 components,
18 categories, `t`, `y`; 9,010 called, 2,886 held back = 75.74%). The randomization check
failed: a ridge classifier predicts who was called with a cross-validated **AUC of 0.64**, far
outside what shuffled labels produce. The call rate differs by category (for `FACTOR18`, from 64%
to 83% across its larger levels), and the numeric components still predict the call within each
level of `FACTOR18` (AUC 0.59 to 0.64). So calls were not assigned with the same probability to
everyone: the "random" hold-back depended on the customer's profile, perhaps through campaign
waves or segments with different hold-back rates. (The authors' own check compared a
classifier's loss with always guessing "called", which is much less sensitive.)

**Disclosure.** The check and the readout ran in the same first test run of the notebook, so
the unadjusted difference was printed before this note was written: called 3.36%, held back
3.64%, a change of −0.28 points (95% CI −1.10 to +0.46). It is reported, but it is no longer the
primary estimate, because it can be biased by who was held back.

**How the analysis continues** (fixed now, before any adjusted or targeting result is computed):

1. **Treat the campaign as randomized given the features.** Assignment is modelled with a
   propensity score: L2 logistic regression on all features, 5-fold cross-fitted. Overlap is
   checked; customers with a propensity below 0.05 or above 0.95 are dropped and counted.
2. **Primary effect: doubly robust (AIPW).** Cross-fitted propensity plus cross-fitted churn
   models for each arm (the LightGBM settings already in the plan's targeting section), with an
   influence-function 95% interval. Secondary: inverse-probability weighting (Hajek). The
   unadjusted difference is shown beside them.
3. **Balance after weighting:** every feature's standardized mean difference below 0.1, and the
   weighted treatment classifier's AUC reported.
4. **Targeting:** the Qini and top-share comparisons are computed with inverse-propensity
   weights, so each ranking is judged on a reweighted comparison of called and held-back
   customers. The decision rule is unchanged.
5. **The limit, stated in the memo:** this assumes the features capture everything that decided
   who was held back. That cannot be checked from the data.


## Note added 2 November 2026 (after the primary analysis)

A sensitivity analysis for unmeasured confounding was added to notebook 05 after the effect had
been estimated: E-values (VanderWeele and Ding) for the doubly robust estimate, as a risk ratio
against the held-back churn rate, both for explaining the estimate away and for hiding a
reduction as large as the interval's edge. It is recorded separately
(`05b_campaign_sensitivity.json`) and does not change the primary result or the decision rule.
