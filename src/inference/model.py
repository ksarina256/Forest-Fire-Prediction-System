from __future__ import annotations
from pathlib import Path
import json
from joblib import load

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

class ModelService:
    def __init__(self):
        self.model = None
        self.metadata = None

    def load(self):
        model_path = MODELS_DIR / "model.joblib"
        meta_path = MODELS_DIR / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run training first.")
        self.model = load(model_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def predict(self, X):
        return self.model.predict(X)
