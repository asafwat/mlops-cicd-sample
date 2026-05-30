"""
Unit tests for training. CI runs these on every PR.

The key invariants tested:
  - build_pipeline returns something with `predict` / `predict_proba`
  - make_data returns arrays of the right shape
  - train_and_evaluate hits a minimum accuracy threshold

The accuracy threshold is the **model promotion gate** — if a code change
drops accuracy below it, the PR check fails and the model is NOT promoted.
This is the simplest form of ML CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.train import build_pipeline, make_data, train_and_evaluate

MIN_ACCURACY = 0.90
MIN_ROC_AUC = 0.90


def test_build_pipeline_has_predict():
    pipe = build_pipeline()
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict")
    assert hasattr(pipe, "predict_proba")


def test_make_data_shape():
    X, y = make_data(n_samples=1000)
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    assert set(np.unique(y).tolist()) == {0, 1}


def test_accuracy_above_threshold():
    """Model promotion gate. Drop below MIN_ACCURACY and the PR fails."""
    _, metrics = train_and_evaluate(n_samples=5000)
    assert metrics["accuracy"] >= MIN_ACCURACY, (
        f"accuracy {metrics['accuracy']:.4f} below threshold {MIN_ACCURACY}"
    )
    assert metrics["roc_auc"] >= MIN_ROC_AUC, (
        f"roc_auc {metrics['roc_auc']:.4f} below threshold {MIN_ROC_AUC}"
    )


@pytest.mark.parametrize("n", [500, 1000, 2000])
def test_train_reproducible(n):
    """Same RANDOM_STATE → same model. If this breaks, someone removed the seed."""
    _, m1 = train_and_evaluate(n_samples=n)
    _, m2 = train_and_evaluate(n_samples=n)
    assert m1["accuracy"] == m2["accuracy"]
    assert m1["roc_auc"] == m2["roc_auc"]
