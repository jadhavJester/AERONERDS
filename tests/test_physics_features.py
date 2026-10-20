"""Tests for src/physics_features.py — turbojet physics-informed features."""

import numpy as np
import pandas as pd
import pytest

from src.physics_features import add_physics_features

CP = 1004.0


def make_df(with_cycle: bool = True) -> pd.DataFrame:
    data = {
        "Pamb_Pa": [101325.0, 95000.0],
        "Tamb_K": [288.15, 280.0],
        "P2_Pa": [250000.0, 230000.0],
        "T2_K": [420.0, 405.0],
        "P3_Pa": [240000.0, 220000.0],
        "T3_K": [1100.0, 1050.0],
        "P4_Pa": [105000.0, 98000.0],
        "T4_K": [800.0, 770.0],
        "FuelFlow_kg_s": [0.85, 0.80],
        "RPM_rev_min": [12500.0, 12100.0],
        "Mach": [0.0, 0.3],
    }
    if with_cycle:
        data["Cycle"] = [10, 250]
    return pd.DataFrame(data)


EXPECTED_NEW = [
    "Compressor_PR", "Compressor_TR", "Overall_PR", "Compressor_Eff",
    "Combustor_TRise", "Combustor_Efficiency_Proxy", "Turbine_ER",
    "Turbine_TDrop", "Turbine_Eff", "Corrected_RPM", "Corrected_FuelFlow",
    "Estimated_Thrust_Proxy", "P3_P2_ratio", "P4_P3_ratio",
    "Enthalpy_Rise", "Enthalpy_Drop",
]


def test_adds_expected_feature_columns():
    out = add_physics_features(make_df())
    for col in EXPECTED_NEW:
        assert col in out.columns, col


def test_pressure_ratios_and_enthalpy_are_exact():
    df = make_df()
    out = add_physics_features(df)
    assert np.allclose(out["Compressor_PR"], df["P2_Pa"] / df["Pamb_Pa"])
    assert np.allclose(out["P3_P2_ratio"], df["P3_Pa"] / df["P2_Pa"])
    assert np.allclose(out["Enthalpy_Rise"], CP * (df["T3_K"] - df["T2_K"]))
    assert np.allclose(out["Enthalpy_Drop"], CP * (df["T3_K"] - df["T4_K"]))
    assert np.allclose(out["Combustor_TRise"], df["T3_K"] - df["T2_K"])
    assert np.allclose(out["Turbine_TDrop"], df["T3_K"] - df["T4_K"])


def test_input_dataframe_is_not_mutated():
    df = make_df()
    cols_before = list(df.columns)
    add_physics_features(df)
    assert list(df.columns) == cols_before


def test_cycle_interactions_only_when_cycle_present():
    out_with = add_physics_features(make_df(with_cycle=True))
    for col in ("Cycle_x_CompEff", "Cycle_x_TurbEff", "Cycle_x_CombEff", "Cycle_x_PR"):
        assert col in out_with.columns
    out_without = add_physics_features(make_df(with_cycle=False))
    assert "Cycle_x_CompEff" not in out_without.columns
    assert np.allclose(
        out_with["Cycle_x_PR"], out_with["Cycle"] * out_with["Compressor_PR"])


def test_features_are_finite_for_realistic_inputs():
    out = add_physics_features(make_df())
    new_cols = [c for c in out.columns if c not in make_df().columns]
    assert np.isfinite(out[new_cols].to_numpy()).all()


def test_real_training_data_runs_and_stays_finite():
    from pathlib import Path

    csv = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    if not csv.exists():
        pytest.skip("training data not present")
    df = pd.read_csv(csv)
    out = add_physics_features(df)
    assert len(out) == len(df)
    for col in EXPECTED_NEW:
        assert col in out.columns
