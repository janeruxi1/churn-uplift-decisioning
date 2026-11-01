"""The demo app runs end to end on the committed records and shows the reported numbers.

The app has two pages: real data (app/real_data.py, tested here widget by widget) and the
StreamFlix case study (app/streamflix_app.py, tested in test_streamflix_app_smoke.py)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP = str(APP_DIR / "real_data.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_shows_the_shipped_model_and_the_experiment(app):
    import json

    root = APP_DIR.parent / "reports" / "metrics"
    m = json.loads((root / "02_kkbox_churn_model.json").read_text())
    e = json.loads((root / "05_campaign_effect.json").read_text())["effect_aipw"]
    values = [x.value for x in app.metric]
    assert f"{m['march'][m['ships']]['pr_auc']:.3f}" in values
    assert f"{e['diff'] * 100:+.2f} pts" in values


def test_controls_update_without_errors(app):
    app.slider(key="reach").set_value(30).run()
    app.slider(key="offer_r").set_value(0.4).run()
    app.select_slider(key="offer_margin").set_value(0.4).run()
    app.slider(key="size_rel").set_value(0.1).run()
    assert not app.exception, app.exception


def test_entry_point_opens_on_the_real_data_page():
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Churn and retention on real data" in t.value for t in at.title)
