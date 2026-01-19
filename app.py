from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image

from gradcam import grad_cam_densenet, detect_orientation
from utils import preprocess_image

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow frontend → backend calls
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(
    "dfu_densenet_ce_model (1).h5",
    compile=False
)

with open("class_map.json") as f:
    class_map = json.load(f)

# ---------------- ROUTES ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # -------- Load Image --------
    image = Image.open(file).convert("RGB")
    img_arr = preprocess_image(image, IMG_SIZE)

    # -------- Prediction --------
    raw_preds = model.predict(img_arr, verbose=0)

    # 🔥 handle multi-output models
    preds = raw_preds[0] if isinstance(raw_preds, list) else raw_preds[0]

    class_idx = int(np.argmax(preds))
    grade = class_idx + 1
    confidence = float(preds[class_idx] * 100)

    # -------- Grad-CAM --------
    heatmap = grad_cam_densenet(model, img_arr, class_idx)
    orientation = detect_orientation(heatmap)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_col = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    overlay = (
        heatmap_col * 0.4
        + np.array(image.resize(IMG_SIZE))
    )

    output_path = os.path.join(
        UPLOAD_FOLDER, "gradcam_result.png"
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            overlay.astype("uint8"),
            cv2.COLOR_RGB2BGR
        )
    )

    # -------- Response --------
    return jsonify({
        "grade": grade,
        "confidence": round(confidence, 2),
        "orientation": orientation,
        "gradcam_image": output_path
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
