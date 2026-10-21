"""Tests for src/degradation_trend.py — RUL trend fitting and threshold crossover."""

import numpy as np
import pandas as pd
import pytest

from src.degradation_trend import DegradationTrend, exp_func, linear_func

THRESHOLD = 0.75


def linear_health(cycles, slope=0.002, intercept=1.0, noise=0.0, seed=0):
    cycles = np.asarray(cycles, dtype=float)
    y = intercept - slope * cycles
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.shape)
    return y


def exponential_health(cycles, decay=0.005, noise=0.0, seed=0):
    cycles = np.asarray(cycles, dtype=float)
    y = np.exp(-decay * cycles)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.shape)
    return y


def test_fit_rejects_too_few_points():
    dt = DegradationTrend(THRESHOLD)
    assert dt.fit([], []) is False
    assert dt.fit([10], [0.95]) is False
    assert dt.func_type is None
    assert dt.popt is None


def test_fit_linear_degradation_selects_linear_model():
    cycles = np.arange(1, 101)
    health = linear_health(cycles, slope=0.002, noise=0.002, seed=0)

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(cycles, health) is True
    assert dt.func_type == "linear"
    # True crossover: (1.0 - 0.75) / 0.002 = 125
    assert dt.estimate_rul_crossover() == pytest.approx(125.0, abs=2.0)


def test_fit_exponential_degradation_selects_exponential_model():
    cycles = np.arange(1, 101)
    health = exponential_health(cycles, decay=0.005, noise=0.001, seed=0)

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(cycles, health) is True
    assert dt.func_type == "exponential"
    # True crossover: ln(0.75) / -0.005 ~= 57.54
    assert dt.estimate_rul_crossover() == pytest.approx(57.54, abs=1.5)


def test_linear_crossover_matches_closed_form():
    cycles = np.arange(1, 51)
    health = linear_health(cycles, slope=0.002, intercept=1.0)

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(cycles, health) is True
    assert dt.func_type == "linear"
    a, b = dt.popt
    assert dt.estimate_rul_crossover() == pytest.approx((THRESHOLD - b) / a)


def test_exponential_crossover_matches_closed_form():
    cycles = np.arange(1, 51)
    health = exponential_health(cycles, decay=0.005)

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(cycles, health) is True
    assert dt.func_type == "exponential"
    a, b = dt.popt
    assert dt.estimate_rul_crossover() == pytest.approx(np.log(THRESHOLD / a) / b)


def test_predict_health_matches_fitted_function():
    cycles = np.arange(1, 51)
    health = linear_health(cycles, slope=0.002, intercept=1.0)

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(cycles, health) is True
    future = np.array([60, 80, 100])
    assert np.allclose(dt.predict_health(future), linear_func(future, *dt.popt))

    dt_exp = DegradationTrend(THRESHOLD)
    assert dt_exp.fit(cycles, exponential_health(cycles)) is True
    assert dt_exp.func_type == "exponential"
    assert np.allclose(dt_exp.predict_health(future), exp_func(future, *dt_exp.popt))


def test_unfitted_model_returns_none():
    dt = DegradationTrend(THRESHOLD)
    assert dt.predict_health([10, 20]) is None
    assert dt.estimate_rul_crossover() is None


def test_non_degrading_health_never_crosses_threshold():
    cycles = np.arange(1, 51)
    # Health improves over time: no crossover should ever be reported.
    health = 0.80 + 0.001 * cycles + np.random.default_rng(7).normal(0, 0.001, 50)

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(cycles, health) is True
    assert dt.estimate_rul_crossover() == np.inf


def test_real_engine_data_end_to_end():
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[1] / "data"
    train_csv = data_dir / "train.csv"
    gt_csv = data_dir / "ground_truth.csv"
    if not train_csv.exists() or not gt_csv.exists():
        pytest.skip("engine datasets not present")

    df = pd.merge(
        pd.read_csv(train_csv), pd.read_csv(gt_csv),
        on=["EngineID", "Cycle"], how="inner",
    )
    engine = df[df["EngineID"] == 1].sort_values("Cycle")

    dt = DegradationTrend(THRESHOLD)
    assert dt.fit(engine["Cycle"].to_numpy(), engine["OverallHealth"].to_numpy()) is True
    crossover = dt.estimate_rul_crossover()
    # A usable RUL estimate: finite and beyond the observed cycles.
    assert np.isfinite(crossover)
    assert crossover > engine["Cycle"].max()
