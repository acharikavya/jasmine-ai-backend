# gradcam_debug.py (robust Grad-CAM for nested/sequential models)
import os, sys, numpy as np, io
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

MODEL_PATH = r"models\Cnn_model.h5"   # adjust if needed
IMG_PATH = r"sample_images\4.14.jpeg"  # choose any sample image from your folder
OUT_PATH = "overlay4_14.png"

def call_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, class_index=None, IMG_SIZE=(224,224)):
    """
    Robust Grad-CAM: performs forward pass inside GradientTape so gradients can be computed
    for nested/sequential models. Returns heatmap (H,W) and predicted class index.
    """
    # pick last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("No conv layer found and last_conv_layer_name not provided.")

    # helper to call layer in inference mode
    def call_layer(layer, x):
        try:
            return layer(x, training=False)
        except TypeError:
            return layer(x)

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Forward pass inside the tape so TensorFlow records ops
    conv_outputs = None
    out = x
    with tf.GradientTape() as tape:
        # we will watch conv_outputs once we create it
        for layer in model.layers:
            out = call_layer(layer, out)
            if layer.name == last_conv_layer_name:
                conv_outputs = out
                tape.watch(conv_outputs)
        preds = out  # final predictions after full forward pass
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    # Compute gradients of the loss w.r.t. the conv outputs
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — model may not be differentiable for this output.")
    grads = grads[0]  # remove batch dim
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_out = conv_outputs[0]  # remove batch dim
    cam = tf.zeros(conv_out.shape[0:2], dtype=tf.float32)
    for i in range(conv_out.shape[-1]):
        cam += pooled_grads[i] * conv_out[:, :, i]
    cam = tf.maximum(cam, 0)
    denom = tf.math.reduce_max(cam) + 1e-8
    cam = cam / denom
    cam = cam.numpy()
    cam = cv2.resize(cam, (IMG_SIZE[1], IMG_SIZE[0]))
    return cam, int(class_index)


def overlay_heatmap_on_pil(img_pil, heatmap, alpha=0.4):
    img = np.array(img_pil.resize((heatmap.shape[1], heatmap.shape[0]))).astype("uint8")
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return Image.fromarray(overlay)

if not os.path.exists(MODEL_PATH):
    print("Model not found:", MODEL_PATH); sys.exit(1)
if not os.path.exists(IMG_PATH):
    print("Image not found:", IMG_PATH); sys.exit(1)

print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded. input shape:", model.input_shape)

# load and preprocess image
img_pil = Image.open(IMG_PATH).convert("RGB").resize((model.input_shape[2], model.input_shape[1]))
arr = np.array(img_pil).astype("float32") / 255.0
x = np.expand_dims(arr, 0)

# prediction
preds = model.predict(x)[0]
print("Pred probs:", preds, "top idx:", int(np.argmax(preds)))

# compute grad-cam
try:
    heatmap, idx = make_gradcam_heatmap(x, model, last_conv_layer_name="conv2d_3",
                                       class_index=None, IMG_SIZE=(model.input_shape[1], model.input_shape[2]))
    overlay = overlay_heatmap_on_pil(Image.open(IMG_PATH), heatmap)
    overlay.save(OUT_PATH)
    print("Saved Grad-CAM overlay to", OUT_PATH)
except Exception as e:
    print("Grad-CAM failed:", e)
    raise
