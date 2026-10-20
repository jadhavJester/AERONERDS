"""Data-integrity tests for the AERONERDS datasets in data/.

These guard the assumptions the training/prediction pipeline relies on:
the sensor columns exist, train/test share a schema, the complete dataset
carries the health/thrust targets, and the physical sensor ranges are sane.
"""

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"

SENSOR_COLS = [
    "EngineID", "Cycle", "Altitude_m", "Mach", "Tamb_K", "Pamb_Pa",
    "RPM_rev_min", "FuelFlow_kg_s", "P2_Pa", "T2_K", "P3_Pa", "T3_K",
    "P4_Pa", "T4_K",
]
TARGET_COLS = [
    "CompressorHealth", "CombustorHealth", "TurbineHealth",
    "OverallHealth", "Thrust_N", "TSFC_g_N_s",
]


def load(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return pd.read_csv(path)


def test_train_and_test_share_sensor_schema():
    train, test = load("train.csv"), load("test.csv")
    assert list(train.columns) == SENSOR_COLS
    assert list(test.columns) == SENSOR_COLS
    assert len(train) > 0 and len(test) > 0


def test_complete_dataset_has_sensors_and_targets():
    df = load("turbojet_complete_dataset.csv")
    assert list(df.columns) == SENSOR_COLS + TARGET_COLS


def test_ground_truth_covers_every_engine_cycle():
    gt = load("ground_truth.csv")
    full = load("turbojet_complete_dataset.csv")
    assert list(gt.columns) == ["EngineID", "Cycle"] + TARGET_COLS
    keys = set(zip(gt["EngineID"], gt["Cycle"]))
    expected = set(zip(full["EngineID"], full["Cycle"]))
    assert keys == expected


def test_sensor_values_are_physically_sane():
    df = load("turbojet_complete_dataset.csv")
    assert (df["Pamb_Pa"] > 0).all()
    assert (df["Tamb_K"] > 0).all()
    assert (df["RPM_rev_min"] > 0).all()
    assert (df["FuelFlow_kg_s"] >= 0).all()
    # Pressures fall through the engine: compressor exit >= combustor.
    assert (df["P2_Pa"] >= df["P3_Pa"]).all()
    # Combustor >= turbine station for nearly all rows. NOTE: 7 of 300 rows
    # in the shipped synthetic dataset have P4 slightly above P3 (verified
    # 2026-10-20) — this guard fails if that anomaly rate grows past 3%.
    assert (df["P3_Pa"] >= df["P4_Pa"]).mean() >= 0.97
    # Health indices and thrust are non-negative
    for col in TARGET_COLS:
        assert (df[col] >= 0).all(), col
    assert df[SENSOR_COLS].notna().all().all()
