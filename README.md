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