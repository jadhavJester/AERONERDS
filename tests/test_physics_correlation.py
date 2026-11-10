"""Tests for the correlation helpers in src/physics_correlation.py."""

import numpy as np
import pandas as pd
import pytest

from src.physics_correlation import compute_pair_correlation, describe_pair


def make_df(x, y):
    return pd.DataFrame({"feat": x, "target": y})


def test_perfect_positive_correlation():
    df = make_df([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    r_val, rho_val = compute_pair_correlation(df, "feat", "target")
    assert r_val == 1.0
    assert rho_val == 1.0


def test_perfect_negative_correlation():
    df = make_df([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0])
    r_val, rho_val = compute_pair_correlation(df, "feat", "target")
    assert r_val == -1.0
    assert rho_val == -1.0


def test_non_finite_rows_are_ignored():
    df = make_df(
        [1.0, 2.0, np.nan, 4.0, np.inf],
        [2.0, 4.0, 6.0, 8.0, 10.0],
    )
    r_val, rho_val = compute_pair_correlation(df, "feat", "target")
    assert r_val == 1.0
    assert rho_val == 1.0


def test_monotonic_but_nonlinear_scores_spearman_above_pearson():
    df = make_df([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 4.0, 9.0, 16.0, 25.0])
    r_val, rho_val = compute_pair_correlation(df, "feat", "target")
    assert rho_val == pytest.approx(1.0)
    assert r_val < 1.0


def test_describe_pair_known_targets():
    assert "compressor health closely" in describe_pair("CompressorHealth")
    assert "surrogate for overall engine health" in describe_pair("OverallHealth")
    assert "thrust output" in describe_pair("Thrust_N")
    assert "Thrust Specific Fuel Consumption" in describe_pair("TSFC_g_N_s")


def test_describe_pair_unknown_target_falls_back_to_combustor_text():
    assert "Combustor temperature rise" in describe_pair("SomethingElse")
