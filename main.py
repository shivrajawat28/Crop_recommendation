from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from train_model import MODEL_PATH, train_and_save_model


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="CropWise ANN Predictor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


FIELD_CONFIG = [
    {
        "name": "N",
        "label": "Nitrogen",
        "unit": "kg/ha",
        "min": 0,
        "max": 140,
        "step": 1,
        "placeholder": "90",
    },
    {
        "name": "P",
        "label": "Phosphorus",
        "unit": "kg/ha",
        "min": 5,
        "max": 145,
        "step": 1,
        "placeholder": "42",
    },
    {
        "name": "K",
        "label": "Potassium",
        "unit": "kg/ha",
        "min": 5,
        "max": 205,
        "step": 1,
        "placeholder": "43",
    },
    {
        "name": "temperature",
        "label": "Temperature",
        "unit": "C",
        "min": 8,
        "max": 45,
        "step": 0.01,
        "placeholder": "20.87",
    },
    {
        "name": "humidity",
        "label": "Humidity",
        "unit": "%",
        "min": 14,
        "max": 100,
        "step": 0.01,
        "placeholder": "82.00",
    },
    {
        "name": "ph",
        "label": "Soil pH",
        "unit": "pH",
        "min": 3,
        "max": 10,
        "step": 0.01,
        "placeholder": "6.50",
    },
    {
        "name": "rainfall",
        "label": "Rainfall",
        "unit": "mm",
        "min": 20,
        "max": 300,
        "step": 0.01,
        "placeholder": "202.93",
    },
]


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_crop(values: dict[str, float]) -> dict:
    artifact = load_artifact()
    features = artifact["features"]
    row = pd.DataFrame([[values[name] for name in features]], columns=features)

    probabilities = artifact["model"].predict_proba(row)[0]
    best_index = int(probabilities.argmax())
    crop = artifact["encoder"].inverse_transform([best_index])[0]
    confidence = float(probabilities[best_index])

    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = [
        {
            "crop": artifact["encoder"].inverse_transform([int(index)])[0],
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]

    return {
        "crop": crop,
        "confidence": round(confidence * 100, 2),
        "top_predictions": top_predictions,
        "accuracy": round(float(artifact.get("accuracy", 0.0)) * 100, 2),
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    artifact = load_artifact()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": FIELD_CONFIG,
            "values": {},
            "result": None,
            "model_accuracy": round(float(artifact.get("accuracy", 0.0)) * 100, 2),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
):
    values = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }
    result = predict_crop(values)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": FIELD_CONFIG,
            "values": values,
            "result": result,
            "model_accuracy": result["accuracy"],
        },
    )
