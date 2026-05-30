"""
FastAPI server tests. The server module loads a model at import time, so we
generate model.joblib first via train.py if it doesn't exist.

CI runs train then these tests sequentially.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "model.joblib"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """Train a model before importing the server (server.py loads it on import)."""
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, "-m", "src.train"], check=True, cwd=ROOT)
    assert MODEL_PATH.exists(), "train.py didn't produce model.joblib"


@pytest.fixture(scope="module")
def client(ensure_model):
    # Import here, after the model exists.
    import os
    os.environ["MODEL_PATH"] = str(MODEL_PATH)
    from src.server import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_input(client):
    instances = [[0.5] * 20, [-0.5] * 20]
    r = client.post("/predict", json={"instances": instances})
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert p["class_id"] in (0, 1)
        assert "not_fraud" in p["probs"]
        assert "fraud" in p["probs"]
        assert 0 <= p["probs"]["fraud"] <= 1


def test_predict_empty_instances_rejected(client):
    r = client.post("/predict", json={"instances": []})
    assert r.status_code == 400


def test_predict_wrong_shape_rejected(client):
    r = client.post("/predict", json={"instances": [[1.0, 2.0]]})  # only 2 features, need 20
    assert r.status_code == 400
