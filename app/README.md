# Streamlit demo

Interactive companion to the repository, with two pages chosen from the menu at the top:

- **Real data** (`app/real_data.py`, the default): neither dataset may be redistributed, so this
  page reads only the aggregate results committed in `reports/metrics/`, and runs anywhere
  without the data.
- **StreamFlix case study** (`app/streamflix_app.py`): the project's simulated first version.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The real-data page has four tabs:

1. **📉 Who will churn?** The KKBox model tested on March: metrics against the rule and logistic
   regression, and a slider for how many churners a contact list of a given size reaches.
2. **📞 Did the call work?** The telecom retention experiment: the failed randomization check, the
   doubly robust effect with its interval, and whether any ranking beats random calling.
3. **🎁 Is an offer worth it?** The free-month decision: members worth an offer and net value as
   the assumed effect and margin change, with the experiment's interval shaded.
4. **📐 Size a retention test:** members per arm for a churn rate and the reduction to detect.

`app/streamlit_app.py` is the entry point that joins the two pages; deploy that file. The
StreamFlix page is described in [`streamflix_README.md`](./streamflix_README.md).
`tests/test_app_smoke.py` and `tests/test_streamflix_app_smoke.py` run both pages and check they
show the reported numbers.
