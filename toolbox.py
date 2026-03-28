"""
toolbox.py
Final Fixed Version
- Image + Soil prediction working
- Swagger shows soil fields
- Correct model loading
- GradCAM working
"""

import io
import base64
import cv2
from pathlib import Path

import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CNN_PATH = MODELS_DIR / "CNN_model_clean.h5"
XGB_PATH = MODELS_DIR / "soil_ml_model.pkl"

DEFAULT_IMG_SIZE = (224, 224)
CLASS_LABELS = ["healthy", "diseased"]

# =========================
# MODEL LOADERS
# =========================
def load_cnn():
    print("Loading CNN from:", CNN_PATH)
    return load_model(str(CNN_PATH))


def load_xgb():
    print("Loading XGBoost from:", XGB_PATH)
    return joblib.load(str(XGB_PATH))


# =========================
# GRADCAM
# =========================
def generate_gradcam(model, img_array, class_index):

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        return None

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, DEFAULT_IMG_SIZE)
    return heatmap


def overlay_heatmap(original_img, heatmap):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        np.array(original_img.resize(DEFAULT_IMG_SIZE)),
        0.6,
        heatmap_color,
        0.4,
        0,
    )
    return overlay


# =========================
# APP FACTORY
# =========================
def create_app():

    print("Starting Jasmine AI backend...")

    cnn = load_cnn()
    xgb = load_xgb()

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "Jasmine AI Backend Running"}

    # =========================
    # PREDICT API
    # =========================
    @app.post("/predict")
    async def predict(
        image: UploadFile = File(None),

        # Soil Inputs (Swagger Friendly)
        PH: float = Form(None),
        EC_ds_m: float = Form(None),
        OC: float = Form(None),
        N: float = Form(None),
        P: float = Form(None),
        K: float = Form(None),
        S: float = Form(None),
        Zn: float = Form(None),
        B: float = Form(None),
        Fe: float = Form(None),
        Mn: float = Form(None),
        Cu: float = Form(None),
    ):

        # =========================
        # SOIL DATA
        # =========================
        soil_values = [
            PH or 0,
            EC_ds_m or 0,
            OC or 0,
            N or 0,
            P or 0,
            K or 0,
            S or 0,
            Zn or 0,
            B or 0,
            Fe or 0,
            Mn or 0,
            Cu or 0,
        ]

        soil_present = any(v != 0 for v in soil_values)
        image_present = image is not None and image.filename != ""

        if not image_present and not soil_present:
            return JSONResponse(
                {"error": "Provide either image or soil data"},
                status_code=400,
            )

        # =========================
        # IMAGE MODE
        # =========================
        if image_present:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")

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

            advice = (
                "Leaf looks healthy."
                if label == "healthy"
                else "Disease detected. Apply treatment."
            )

            gradcam_overlay_b64 = None
            try:
                heatmap = generate_gradcam(cnn, img_array, idx)
                if heatmap is not None:
                    overlay = overlay_heatmap(img, heatmap)
                    buff = io.BytesIO()
                    Image.fromarray(overlay).save(buff, format="PNG")
                    gradcam_overlay_b64 = base64.b64encode(buff.getvalue()).decode()
            except Exception as e:
                print("GradCAM error:", e)

            return {
                "model_used": "image",
                "label": label,
                "confidence": confidence,
                "severity": severity,
                "advice": advice,
                "gradcam_overlay_b64": gradcam_overlay_b64
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

            severity = "Low" if label == "healthy" else "High"

            advice = (
                "Soil nutrients balanced."
                if label == "healthy"
                else "Soil imbalance detected. Improve nutrients."
            )

            return {
                "model_used": "soil",
                "label": label,
                "confidence": confidence,
                "severity": severity,
                "advice": advice,
            }

        return JSONResponse({"error": "Something went wrong"}, status_code=500)

    return app

