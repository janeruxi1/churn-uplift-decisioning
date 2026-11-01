# Streamlit demo

Interactive companion to the repository. Neither dataset may be redistributed, so the app reads
only the aggregate results committed in `reports/metrics/`, and runs anywhere without the data.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

1. **📉 Who will churn?** The KKBox model tested on March: metrics against the rule and logistic
   regression, and a slider for how many churners a contact list of a given size reaches.
2. **📞 Did the call work?** The telecom retention experiment: the failed randomization check, the
   doubly robust effect with its interval, and whether any ranking beats random calling.
3. **🎁 Is an offer worth it?** The free-month decision: members worth an offer and net value as
   the assumed effect and margin change, with the experiment's interval shaded.
4. **📐 Size a retention test:** members per arm for a churn rate and the reduction to detect.

The simulated first version has its own app, `app/streamflix_app.py` (see
[`streamflix_README.md`](./streamflix_README.md)). `tests/test_app_smoke.py` runs this app and checks
it shows the reported numbers.
