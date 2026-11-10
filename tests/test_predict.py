"""Tests for the stacked-inference helpers in src/predict.py.

Uses stub models with fixed predictions — no trained model files or
xgboost/lightgbm needed to run them.
"""

import numpy as np

from src.predict import bootstrap_percentiles, stack_predict


class StubModel:
    """A model that always returns the same predictions."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X):
        return self._preds


class MeanMetaModel:
    """Meta-learner stub: returns the row-wise mean of the base predictions."""

    def predict(self, meta_features):
        return np.mean(meta_features, axis=1)


X = np.zeros((3, 2))  # feature values never read by the stubs


def test_stack_predict_feeds_base_outputs_to_meta_learner():
    base = [StubModel([1.0, 2.0, 3.0]), StubModel([4.0, 5.0, 6.0])]
    result = stack_predict(base, MeanMetaModel(), X)
    np.testing.assert_allclose(result, [2.5, 3.5, 4.5])


def test_stack_predict_single_base_model():
    base = [StubModel([10.0, 20.0, 30.0])]
    result = stack_predict(base, MeanMetaModel(), X)
    np.testing.assert_allclose(result, [10.0, 20.0, 30.0])


def test_bootstrap_percentiles_known_values():
    models = [StubModel([0.0, 0.0]), StubModel([10.0, 10.0]), StubModel([20.0, 20.0])]
    p10, p90 = bootstrap_percentiles(models, X)
    # np.percentile linearly interpolates: p10 of [0, 10, 20] is 2.0, p90 is 18.0
    np.testing.assert_allclose(p10, [2.0, 2.0])
    np.testing.assert_allclose(p90, [18.0, 18.0])


def test_bootstrap_percentiles_identical_models_collapse_to_point():
    models = [StubModel([5.0, 7.0]) for _ in range(5)]
    p10, p90 = bootstrap_percentiles(models, X)
    np.testing.assert_allclose(p10, [5.0, 7.0])
    np.testing.assert_allclose(p90, [5.0, 7.0])


def test_bootstrap_percentiles_band_contains_median():
    models = [StubModel([float(i), float(i)]) for i in range(10)]
    p10, p90 = bootstrap_percentiles(models, X)
    assert np.all(p10 <= 4.5) and np.all(p90 >= 4.5)
    assert np.all(p10 <= p90)
