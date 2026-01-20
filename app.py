import os
import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

from gradcam import grad_cam_densenet, detect_orientation
from utils import preprocess_image


# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/outputs"
MODEL_PATH = "dfu_densenet_ce_model.h5"
CLASS_MAP_PATH = "class_map.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- APP INIT ----------------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded")

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    if request.method == "GET":
        return jsonify({
            "message": "Use POST with an image file",
            "endpoint": "/analyze"
        }), 405

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Load image
        image = Image.open(file).convert("RGB")
        img_arr = preprocess_image(image, IMG_SIZE)

        # Prediction
        raw_preds = model.predict(img_arr, verbose=0)
        preds = raw_preds[0] if isinstance(raw_preds, list) else raw_preds[0]

        class_idx = int(np.argmax(preds))
        grade = class_idx + 1
        confidence = float(preds[class_idx] * 100)

        # Grad-CAM
        heatmap = grad_cam_densenet(model, img_arr, class_idx)
        orientation = detect_orientation(heatmap)

        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )

        overlay = (
            heatmap_color * 0.4 +
            np.array(image.resize(IMG_SIZE))
        )

        output_path = os.path.join(UPLOAD_FOLDER, "gradcam_result.png")

        cv2.imwrite(
            output_path,
            cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_RGB2BGR)
        )

        return jsonify({
            "grade": grade,
            "confidence": round(confidence, 2),
            "orientation": orientation,
            "gradcam_image": output_path
        })

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
