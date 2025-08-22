# Forest Fire Prediction System 🔥

A production-style ML MVP that predicts **wildfire ignition risk** from weather and FWI (Fire Weather Index) features.
It includes a training pipeline, MLflow experiment tracking, a FastAPI inference service, tests, Docker, and GitHub Actions CI.

## Tech Stack
- **Python** (pandas, scikit-learn)
- **FastAPI** for inference API
- **MLflow** for experiment tracking
- **Docker** for containerization
- **Pytest** for unit tests
- **GitHub Actions** for CI

## Project Structure
```text
src/
  training/
    train.py
    features.py
    config.yaml
  inference/
    app.py
    schemas.py
    model.py
    utils.py
tests/
  test_features.py
  test_api.py
models/
  (trained model + metadata saved here)
.github/workflows/ci.yml
docker/Dockerfile.api
docker-compose.yml
requirements.txt
Makefile
```

## Quickstart

1) Create environment & install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Set MLflow tracking URI (defaults to local `./mlruns` folder)
```bash
export MLFLOW_TRACKING_URI=./mlruns
```

3) Train model
```bash
make train
```

4) Run API locally
```bash
make api
# POST http://localhost:8000/predict
```

5) Docker
```bash
make docker-build
make docker-run
```

## Example Request
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "FFMC": 86.2, "DMC": 26.2, "DC": 94.3, "ISI": 5.1,
  "temp": 18.0, "RH": 45, "wind": 4.3, "rain": 0.0,
  "month": "aug", "day": "fri"
}'
```

## Resume Pitch (copy/paste)
*Built and deployed a production-style wildfire risk prediction system with an ML pipeline (scikit‑learn), experiment tracking (MLflow), and a FastAPI inference service; containerized with Docker and wired with GitHub Actions CI to run tests and build images.*
