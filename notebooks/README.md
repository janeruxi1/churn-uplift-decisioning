# Notebooks: reading guide

Read them in order. Each notebook opens with a short table (question, answer, next) and ends
with **Takeaways**; every figure has a "how to read it" note under it.

| # | Notebook | The question in plain words | Main figure |
|---|---|---|---|
| **Part 1 · KKBox: who will churn, and what are they worth?** | | | |
| 01 | [Data checks](01_kkbox_data_checks.ipynb) | Can the churn label and the payment data be trusted? | Churn by auto-renew and by tenure |
| 02 | [Churn model](02_kkbox_churn_model.ipynb) | Who will churn next month, tested on a month the model never saw? | Calibration and churners reached; SHAP, segments and February-to-March drift |
| 03 | [Listening](03_kkbox_listening.ipynb) | Does listening data improve the model? | Churn by days of listening |
| 04 | [Lifetime value](04_kkbox_lifetime_value.ipynb) | What is keeping a member worth? | Survival curves by plan and auto-renew |
| **Part 2 · A real retention campaign (telecom)** | | | |
| 05 | [Effect](05_campaign_effect.ipynb) | Did calling at-risk customers reduce churn? | The call's effect, several estimators; E-values for hidden confounding |
| 06 | [Targeting](06_campaign_targeting.ipynb) | Could a ranking find the customers the call helps? | Qini curves |
| **Part 3 · The decision** | | | |
| 07 | [Retention decision](07_retention_decision.ipynb) | Should KKBox offer a free month, and to whom? | Net value across the plausible effects |

The [StreamFlix appendix](streamflix/README.md) is the project's first version on simulated
data, where the true effects are known.

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source); see the main [README](../README.md#reproduce) for the data and the build command.
