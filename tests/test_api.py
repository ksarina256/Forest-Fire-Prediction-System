import os, json, pathlib
import pytest

# Only run API test if model exists (after training).
MODEL_PATH = pathlib.Path("models/model.joblib")

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not trained yet")
def test_predict_endpoint():
    from fastapi.testclient import TestClient
    from src.inference.app import app

    client = TestClient(app)
    payload = {
        "FFMC": 86.2, "DMC": 26.2, "DC": 94.3, "ISI": 5.1,
        "temp": 18.0, "RH": 45, "wind": 4.3, "rain": 0.0,
        "month": "aug", "day": "fri"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "risk" in body and "probability" in body
