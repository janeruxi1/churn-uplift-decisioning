"""MLflow helpers: a run records what it was given, and bad metric values don't fail it."""
import math

import pytest

mlflow = pytest.importorskip("mlflow")


def test_log_run_records_params_metrics_and_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'runs.db').as_posix()}")
    from src.models.tracking import log_run

    run_id = log_run("test_experiment", "a_run", params={"num_leaves": 31},
                     metrics={"pr_auc": 0.5, "nan": math.nan, "missing": None}, tags={"ships": True})
    run = mlflow.get_run(run_id)
    assert run.data.params == {"num_leaves": "31"}
    assert run.data.metrics == {"pr_auc": 0.5}
    assert run.data.tags["ships"] == "True"
    assert run.info.run_name == "a_run"
