"""
toolbox.py
Production-ready FastAPI backend for:
- Jasmine Leaf Disease CNN model (.h5)
- Soil Health XGBoost model (.pkl)

Start on Render with:
uvicorn toolbox:create_app --host 0.0.0.0 --port $PORT --factory
"""

import os
import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =========================
# PATH CONFIG (Render Safe)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CNN_PATH = MODELS_DIR / "CNN_model_fixed.h5"
XGB_PATH = MODELS_DIR / "xgboost_soil_health_model.pkl"

DEFAULT_IMG_SIZE = (224, 224)
CLASS_LABELS = ["healthy", "diseased"]

# Soil feature order (IMPORTANT: must match training order)
SOIL_FEATURES = [
    "PH", "EC ds/m", "OC %", "N kg/hectre", "P kg/hectre", "K kg/hectre",
    "S ppm", "Zn ppm", "B ppm", "Fe ppm", "Mn ppm", "Cu ppm"
]


# =========================
# MODEL LOADERS
# =========================
def load_cnn():
    if not CNN_PATH.exists():
        raise FileNotFoundError(f"CNN model not found at {CNN_PATH}")
    print("Loading CNN from:", CNN_PATH)
    model = load_model(str(CNN_PATH))
    return model


def load_xgb():
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found at {XGB_PATH}")
    print("Loading XGBoost from:", XGB_PATH)
    model = joblib.load(str(XGB_PATH))
    return model


# =========================
# FASTAPI APP FACTORY
# =========================
def create_app():
    print("Starting Jasmine AI backend...")

    cnn = None
    xgb = None

    # Load CNN
    try:
        cnn = load_cnn()
    except Exception as e:
        print("CNN load error:", e)

    # Load XGBoost
    try:
        xgb = load_xgb()
    except Exception as e:
        print("XGB load error:", e)

    app = FastAPI()

    # CORS (Required for Flutter / Web apps)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change later for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================
    # ROOT CHECK
    # =========================
    @app.get("/")
    async def root():
        return {"status": "ok", "message": "Jasmine AI Backend Running"}

    # =========================
    # PREDICT ENDPOINT
    # =========================
    @app.post("/predict")
    async def predict(request: Request, image: UploadFile = File(None)):
        form = await request.form()

        # =========================
        # CHECK SOIL INPUT
        # =========================
        soil_values = []
        soil_present = False

        for feature in SOIL_FEATURES:
            value = form.get(feature)
            if value:
                soil_present = True
                try:
                    soil_values.append(float(value))
                except:
                    soil_values.append(0.0)
            else:
                soil_values.append(0.0)

        image_present = image is not None

        if not image_present and not soil_present:
            return JSONResponse(
                {"error": "Provide either leaf image or soil data"},
                status_code=400,
            )

        # =========================
        # IMAGE MODE
        # =========================
        if image_present and cnn is not None:
            contents = await image.read()

            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
            except:
                return JSONResponse({"error": "Invalid image"}, status_code=400)

            img_resized = img.resize(DEFAULT_IMG_SIZE)
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = cnn.predict(img_array)[0]
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = CLASS_LABELS[idx]

            severity = "Low" if label == "healthy" else (
                "High" if confidence > 0.85 else "Medium"
            )

            return {
                "model_used": "image",
                "label": label,
                "confidence": confidence,
                "severity": severity
            }

        # =========================
        # SOIL MODE
        # =========================
        if soil_present and xgb is not None:
            model_input = np.array(soil_values).reshape(1, -1)

            try:
                probs = xgb.predict_proba(model_input)[0]
                idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                label = CLASS_LABELS[idx]
            except:
                label = "unknown"
                confidence = 0.0

            severity = "Low" if label == "healthy" else "High"

            return {
                "model_used": "soil",
                "label": label,
                "confidence": confidence,
                "severity": severity
            }

        return JSONResponse(
            {"error": "Model not available"},
            status_code=500,
        )

    return app
