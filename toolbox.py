import io
import base64
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
        return {"status": "ok", "message": "Backend Running"}

    # =========================
    # PREDICT API
    # =========================
    @app.post("/predict")
    async def predict(request: Request, image: UploadFile = File(None)):

        form = await request.form()

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

        image_present = image is not None and image.filename != ""

        if not image_present and not soil_present:
            return JSONResponse(
                {"error": "Provide image or soil data"},
                status_code=400,
            )

        # ================= IMAGE =================
        if image_present:
            contents = await image.read()
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
            }

        # ================= SOIL =================
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
            }

        return JSONResponse({"error": "Something went wrong"}, status_code=500)

    return app

