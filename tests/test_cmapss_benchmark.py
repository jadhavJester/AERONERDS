"""Tests for the pure helpers in src/cmapss_benchmark.py.

These cover the C-MAPSS feature engineering and RUL target construction
without needing the NASA dataset or xgboost (the heavy ML imports live
inside run_cmapss_benchmark, so this module imports with light deps only).
"""

import numpy as np
import pandas as pd
import pytest

from src.cmapss_benchmark import (
    DROP_SENSORS,
    add_physics_proxies,
    build_feature_columns,
    compute_piecewise_rul,
)


def _sensor_frame(**overrides):
    data = {f"Sensor_{i}": [100.0] for i in range(1, 22)}
    data.update(overrides)
    return pd.DataFrame(data)


def test_proxies_match_sensor_ratios():
    df = _sensor_frame(
        Sensor_1=[200.0],
        Sensor_2=[300.0],
        Sensor_3=[600.0],
        Sensor_4=[900.0],
        Sensor_5=[50.0],
        Sensor_7=[500.0],
        Sensor_8=[1000.0],
        Sensor_9=[2000.0],
    )
    out = add_physics_proxies(df)
    assert out["PR_HPC"].iloc[0] == pytest.approx(500.0 / 50.0)
    assert out["TR_HPC"].iloc[0] == pytest.approx(600.0 / 200.0)
    assert out["TR_LPT"].iloc[0] == pytest.approx(900.0 / 300.0)
    assert out["Speed_Ratio"].iloc[0] == pytest.approx(2000.0 / 1000.0)


def test_proxies_do_not_mutate_input_and_stay_finite_on_zero_denominator():
    df = _sensor_frame(Sensor_5=[0.0], Sensor_1=[0.0])
    out = add_physics_proxies(df)
    assert "PR_HPC" not in df.columns
    assert np.isfinite(out["PR_HPC"]).all()
    assert np.isfinite(out["TR_HPC"]).all()


def test_piecewise_rul_counts_down_to_zero_per_engine():
    df = pd.DataFrame(
        {
            "EngineID": [1, 1, 1, 2, 2],
            "Cycle": [1, 2, 3, 1, 2],
        }
    )
    out = compute_piecewise_rul(df)
    assert out["RUL"].tolist() == [2, 1, 0, 1, 0]
    assert "RUL" not in df.columns


def test_piecewise_rul_clips_at_max_rul():
    df = pd.DataFrame(
        {
            "EngineID": [1] * 200,
            "Cycle": list(range(1, 201)),
        }
    )
    out = compute_piecewise_rul(df, max_rul=125)
    assert out["RUL"].iloc[0] == 125
    assert out["RUL"].iloc[-1] == 0
    assert out["RUL"].max() <= 125


def test_feature_columns_drop_constant_sensors_and_include_proxies():
    features = build_feature_columns()
    for sensor in DROP_SENSORS:
        assert f"Sensor_{sensor}" not in features
    for proxy in ["PR_HPC", "TR_HPC", "TR_LPT", "Speed_Ratio"]:
        assert proxy in features
    # 3 settings + 14 kept sensors + 4 proxies
    assert len(features) == 3 + (21 - len(DROP_SENSORS)) + 4
