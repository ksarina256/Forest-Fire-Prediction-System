from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import FireFeatures, Prediction
from .model import ModelService
from .utils import dict_to_df

app = FastAPI(title="Forest Fire Risk API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = ModelService()

@app.on_event("startup")
def _load_model():
    svc.load()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(payload: FireFeatures):
    df = dict_to_df(payload.model_dump())
    proba = float(svc.predict_proba(df)[0])
    risk = int(proba >= 0.5)
    # Use global feature importances as a lightweight "explanation"
    try:
        import numpy as np
        importances = svc.model.named_steps["clf"].feature_importances_
        pre = svc.model.named_steps["pre"]
        cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(["month","day"]))
        num_names = ["FFMC","DMC","DC","ISI","temp","RH","wind","rain"]
        names = num_names + cat_names
        top_idx = list(np.argsort(importances)[-3:][::-1])
        top_features = [names[i] for i in top_idx]
    except Exception:
        top_features = ["FFMC","DMC","ISI"]
    return Prediction(risk=risk, probability=proba, top_features=top_features)
