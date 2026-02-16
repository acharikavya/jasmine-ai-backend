"""
toolbox.py
Unified production backend:
- CNN (.h5)
- XGBoost (.pkl)
- GradCAM
- SHAP
- Unified JSON structure
"""

import io
import base64
import cv2
import shap
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
XGB_PATH = MODELS_DIR / "xgboost_soil_health_model.pkl"

DEFAULT_IMG_SIZE = (224, 224)
CLASS_LABELS = ["healthy", "diseased"]

SOIL_FEATURES = [
    "PH", "EC ds/m", "OC %", "N kg/hectre", "P kg/hectre", "K kg/hectre",
    "S ppm", "Zn ppm", "B ppm", "Fe ppm", "Mn ppm", "Cu ppm"
]

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
        if "conv" in layer.name.lower():
            last_conv_layer = layer.name
            break

    if not last_conv_layer:
        return None

    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), DEFAULT_IMG_SIZE)

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

    shap_explainer = shap.TreeExplainer(xgb)

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

    @app.post("/predict")
    async def predict(request: Request, image: UploadFile = File(None)):

        form = await request.form()

        # =========================
        # SOIL CHECK
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

            # Severity logic
            severity = "Low" if label == "healthy" else (
                "High" if confidence > 0.85 else "Medium"
            )

            # Separate leaf advice
            if label == "healthy":
                advice = "Leaf looks healthy. Maintain watering and sunlight schedule."
            else:
                if severity == "High":
                    advice = "Severe infection detected. Remove infected leaves and apply treatment immediately."
                elif severity == "Medium":
                    advice = "Moderate disease detected. Monitor closely and consider treatment."
                else:
                    advice = "Minor symptoms observed. Keep monitoring plant."

            # GradCAM
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

            # Separate soil advice
            if label == "healthy":
                advice = "Soil nutrients appear balanced. Continue current fertilization plan."
            else:
                advice = "Soil imbalance detected. Adjust pH and nutrient levels. Consider soil treatment."

            shap_top_features = None
            try:
                shap_values = shap_explainer.shap_values(model_input)
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                values = values.flatten()
                top_indices = np.argsort(np.abs(values))[-3:][::-1]

                shap_top_features = [
                    {
                        "feature": SOIL_FEATURES[i],
                        "impact": float(values[i]),
                    }
                    for i in top_indices
                ]
            except Exception as e:
                print("SHAP error:", e)

            return {
                "model_used": "soil",
                "label": label,
                "confidence": confidence,
                "severity": severity,
                "advice": advice,
                "shap_top_features": shap_top_features
            }

        return JSONResponse(
            {"error": "Model not available"},
            status_code=500,
        )

    return app
