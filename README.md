# MLflow Wine Classifier — ML Pipeline & API

A production-style machine learning project demonstrating the full ML lifecycle:
train → track → compare → register → serve.

## What This Project Does

- Trains a Random Forest classifier on the Wine dataset (178 samples, 13 features, 3 classes)
- Tracks experiments with MLflow — logging parameters, metrics, and model artifacts
- Compares multiple hyperparameter combinations (9 runs) using MLflow's parallel coordinates UI
- Registers the best model to the MLflow Model Registry with staging promotion
- Serves the model as a REST API using FastAPI with input validation, error handling, and confidence scores

## Tech Stack

- Python
- MLflow — experiment tracking, model registry, model serving
- Scikit-learn — Random Forest classifier
- FastAPI — production API layer
- Uvicorn — ASGI server
- NumPy / Pandas

## Project Structure

mlflow-wine-classifier/
├── train.py          # MLflow experiment tracking — trains 9 runs, logs params/metrics
├── register.py       # Registers best model to MLflow Model Registry, promotes to Staging
├── api.py            # FastAPI app — /health and /predict endpoints with confidence scores
├── README.md
└── .gitignore

## API Endpoints

### GET /health
Returns server status.
```json
{"status": "ok"}
```

### POST /predict
Takes 13 wine chemical features and returns predicted class with confidence scores.

**Request:**
```json
{
  "alcohol": 13.2,
  "malic_acid": 2.77,
  "ash": 2.51,
  "alcalinity_of_ash": 18.5,
  "magnesium": 96.6,
  "total_phenols": 1.04,
  "flavanoids": 2.55,
  "nonflavanoid_phenols": 0.57,
  "proanthocyanins": 1.47,
  "color_intensity": 6.2,
  "hue": 1.05,
  "od280_od315": 3.33,
  "proline": 820
}
```

**Response:**
```json
{
  "predicted_class": 0,
  "confidence": {
    "class_0": 0.6769,
    "class_1": 0.1575,
    "class_2": 0.1656
  }
}
```

## How to Run

**Install dependencies:**
```bash
pip install mlflow scikit-learn fastapi uvicorn numpy pandas
```

**Step 1 — Run experiments and track with MLflow:**
```bash
python train.py
```

**Step 2 — View MLflow UI and compare runs:**
```bash
mlflow ui
```
Go to `http://localhost:5000`

**Step 3 — Register best model to staging:**
```bash
python register.py
```

**Step 4 — Serve the API:**
```bash
uvicorn api:app --reload --port 8000
```
Go to `http://localhost:8000/docs` for interactive API docs.

## What I Learned

- Full ML lifecycle management using MLflow
- Hyperparameter comparison across multiple experiment runs using parallel coordinates
- Model registration and staging promotion workflows
- Building production-grade REST APIs with FastAPI
- Input validation, error handling, and confidence scoring in ML APIs
- Version control with Git for ML projects