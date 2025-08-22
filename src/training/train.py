from __future__ import annotations
import os, json
import mlflow
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump
import yaml
from pathlib import Path

from .features import build_preprocessor, engineer_label, NUM_COLS, CAT_COLS

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "src" / "training" / "config.yaml"

@dataclass
class Config:
    dataset_url: str
    test_size: float
    random_state: int
    label_threshold_area: float
    target: str

def load_config() -> Config:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def fetch_dataset(url: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    dest = DATA_DIR / "forestfires.csv"
    if not dest.exists():
        import urllib.request
        print(f"Downloading dataset from {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
    return dest

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # month/day may already be lowercase strings in UCI dataset
    return df

def train(cfg: Config) -> Tuple[Pipeline, dict]:
    # MLflow local tracking by default
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("forest-fire-risk")

    csv_path = fetch_dataset(cfg.dataset_url)
    df = load_data(csv_path)
    df = engineer_label(df, target_col=cfg.target)

    features = NUM_COLS + CAT_COLS
    X = df[features]
    y = df[cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    preproc = build_preprocessor()
    clf = RandomForestClassifier(random_state=cfg.random_state, n_estimators=300)

    pipe = Pipeline([("pre", preproc), ("clf", clf)])

    param_grid = {
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }

    with mlflow.start_run():
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)
        if hasattr(gs, "predict_proba"):
            y_proba = gs.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", float(auc))
        report = classification_report(y_test, y_pred, output_dict=True)
        for k, v in report["weighted avg"].items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"weighted_{k}", float(v))

        mlflow.log_params(gs.best_params_)
        best_model = gs.best_estimator_

        MODELS_DIR.mkdir(exist_ok=True, parents=True)
        model_path = MODELS_DIR / "model.joblib"
        dump(best_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        meta = {
            "features": features,
            "numeric": NUM_COLS,
            "categorical": CAT_COLS,
            "target": cfg.target,
            "best_params": gs.best_params_,
        }
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(str(MODELS_DIR / "metadata.json"), artifact_path="model")

        print("Training complete. Model saved to", model_path)
        return best_model, meta

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
