
import io
import cv2
from pathlib import Path

import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CNN_PATH = MODELS_DIR / "CNN_model_clean.h5"
XGB_PATH = MODELS_DIR / "soil_xgboost_model.pkl"

DEFAULT_IMG_SIZE = (224, 224)
CLASS_LABELS = ["healthy", "diseased"]

# =========================
# LOAD MODELS
# =========================
def load_cnn():
    print("Loading CNN from:", CNN_PATH)
    return load_model(str(CNN_PATH))


def load_xgb():
    print("Loading XGB from:", XGB_PATH)
    return joblib.load(str(XGB_PATH))


# =========================
# APP FACTORY
# =========================
def create_app():

    print("Starting backend...")

    cnn = load_cnn()
    xgb = load_xgb()

    app = FastAPI()

    # ✅ CORS FIX
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"status": "ok"}

    # =========================
    # PREDICT API
    # =========================
    @app.post("/predict")
    async def predict(request: Request, image: UploadFile = File(None)):

        form = await request.form()

        # =========================
        # SOIL INPUT
        # =========================
        keys = [
            "PH", "EC_ds_m", "OC", "N", "P", "K",
            "S", "Zn", "B", "Fe", "Mn", "Cu"
        ]

        soil_values = []
        soil_present = False

        for key in keys:
            value = form.get(key)
            if value:
                soil_present = True
                try:
                    soil_values.append(float(value))
                except:
                    soil_values.append(0.0)
            else:
                soil_values.append(0.0)

        # =========================
        # SAFE IMAGE CHECK (FIXED)
        # =========================
        image_present = False
        contents = None

        if image is not None:
            try:
                contents = await image.read()
                if contents and len(contents) > 0:
                    image_present = True
            except:
                image_present = False

        # =========================
        # VALIDATION
        # =========================
        if not image_present and not soil_present:
            return JSONResponse(
                {"error": "Provide image or soil data"},
                status_code=400,
            )

        # =========================
        # IMAGE MODE
        # =========================
        if image_present:
            img = Image.open(io.BytesIO(contents)).convert("RGB")

            img = img.resize(DEFAULT_IMG_SIZE)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = cnn.predict(img_array)[0]
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = CLASS_LABELS[idx]

            return {
                "model_used": "image",
                "label": label,
                "confidence": confidence,
                "severity": "Low" if label == "healthy" else "High",
                "advice": "Leaf healthy"
                if label == "healthy"
                else "Disease detected",
            }

        # =========================
        # SOIL MODE
        # =========================
        if soil_present:
            model_input = np.array(soil_values).reshape(1, -1)

            probs = xgb.predict_proba(model_input)[0]
            idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            label = CLASS_LABELS[idx]

            return {
                "model_used": "soil",
                "label": label,
                "confidence": confidence,
                "severity": "Low" if label == "healthy" else "High",
                "advice": "Soil balanced"
                if label == "healthy"
                else "Soil needs improvement",
            }

        return JSONResponse({"error": "Something went wrong"}, status_code=500)

    return app
