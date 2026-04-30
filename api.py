import mlflow.sklearn 
import numpy as np 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


model = mlflow.sklearn.load_model("mlruns/1/models/m-0d7d7b2fdc5d4e299d16507499fd9868/artifacts")

app = FastAPI()
Instrumentator().instrument(app).expose(app)

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: WineFeatures):
    try:
        data = np.array([[
            features.alcohol, features.malic_acid, features.ash,
            features.alcalinity_of_ash, features.magnesium, features.total_phenols,
            features.flavanoids, features.nonflavanoid_phenols, features.proanthocyanins,
            features.color_intensity, features.hue, features.od280_od315, features.proline
        ]])
        prediction = model.predict(data)
        probabilities = model.predict_proba(data)[0]
        return {
            "predicted_class": int(prediction[0]),
            "confidence": {
                "class_0": round(float(probabilities[0]), 4),
                "class_1": round(float(probabilities[1]), 4),
                "class_2": round(float(probabilities[2]), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))