"""
Train a tiny fraud-detection sklearn Pipeline and save it to model.joblib.

This is the simplest possible ML training script the CI/CD pipeline operates
on — same sklearn pattern as the kserve-model-serving exercise, ~30 lines.
The point of THIS exercise is the CI/CD pipeline, not the model.

Output: model.joblib in the repo root (baked into the Docker image at build time).
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent.parent / "model.joblib"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def make_data(n_samples: int = 10_000):
    return make_classification(
        n_samples=n_samples, n_features=20, n_informative=15, n_redundant=5,
        weights=[0.95, 0.05], random_state=RANDOM_STATE,
    )


def train_and_evaluate(n_samples: int = 10_000) -> tuple[Pipeline, dict[str, float]]:
    X, y = make_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return pipe, metrics


def main():
    print("Training fraud-detection pipeline...")
    pipe, metrics = train_and_evaluate()
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  roc_auc:  {metrics['roc_auc']:.4f}")
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
