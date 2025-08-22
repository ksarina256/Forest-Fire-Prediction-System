.PHONY: train api test format docker-build docker-run

    PY=python
    APP=src/inference/app.py

    train:
		$(PY) src/training/train.py

    api:
		uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload

    test:
		pytest -q

    format:
		python -m pip install black isort >/dev/null || true
		black src tests
		isort src tests

    docker-build:
		docker build -f docker/Dockerfile.api -t forest-fire-api:latest .

    docker-run:
		docker run --rm -p 8000:8000 forest-fire-api:latest
