"""PSI for drift and the E-value for unmeasured confounding, on cases with known answers."""
import math

import numpy as np
import pandas as pd

from src.churn.model import psi
from src.experiment.campaign import e_value


def test_psi_is_zero_for_the_same_distribution_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 50_000))
    assert psi(ref, ref) < 1e-9
    small = psi(ref, pd.Series(rng.normal(0.1, 1, 50_000)))
    large = psi(ref, pd.Series(rng.normal(1.0, 1, 50_000)))
    assert small < 0.1 < 0.25 < large


def test_psi_counts_missing_values_and_categories():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0] * 100)
    more_missing = pd.Series([1.0, 2.0, np.nan, np.nan] * 100)
    assert psi(ref, more_missing) > 0.25
    cats = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi(cats, cats) < 1e-9
    assert psi(cats, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.25


def test_e_value_matches_the_formula_and_is_symmetric():
    assert math.isclose(e_value(2.0), 2 + math.sqrt(2))
    assert math.isclose(e_value(0.5), e_value(2.0))
    assert e_value(1.0) == 1.0
