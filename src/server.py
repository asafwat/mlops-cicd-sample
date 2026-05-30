"""
FastAPI server for the trained fraud-detection model.

Endpoint shape mirrors the rest of the lab's serving patterns:
  POST /predict
  {"instances": [[<20 floats>], ...]}

  Response:
  {"predictions": [{"class_id": int, "probs": {...}}, ...]}

The model is BAKED INTO the Docker image at /app/model.joblib via the
Dockerfile — same pattern as the TF + PyTorch + HF exercises.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model.joblib"))

print(f"[server] loading model from {MODEL_PATH}...")
_model = joblib.load(MODEL_PATH)
print("[server] model ready")


class PredictRequest(BaseModel):
    instances: list[list[float]]


class Prediction(BaseModel):
    class_id: int
    probs: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]


app = FastAPI(title="mlops-cicd-sample")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.instances:
        raise HTTPException(status_code=400, detail="instances must not be empty")

    X = np.asarray(req.instances, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 20:
        raise HTTPException(
            status_code=400,
            detail=f"each instance must be 20 floats, got shape {X.shape}",
        )

    preds = _model.predict(X)
    probs = _model.predict_proba(X)
    return PredictResponse(
        predictions=[
            Prediction(
                class_id=int(p),
                probs={"not_fraud": float(pr[0]), "fraud": float(pr[1])},
            )
            for p, pr in zip(preds, probs)
        ]
    )
