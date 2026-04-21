from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TEMPLATES = {
    "1st": "1st.jpg",
    "2nd": "2nd.jpg",
    "3rd": "3rd.jpg",
    "4th": "4th.jpg",
}

FIXED_X = {
    "1st": 231,
    "2nd": 467,
    "3rd": 701,
    "4th": 933,
}

ROI_MARGIN = 120


def y_to_index(label, y):
    if label in ["1st", "4th"]:
        base = 1470
        max_val = 1580
    else:
        base = 1470
        max_val = 1650

    if y < base or y > max_val:
        return None

    return int((y - base) / 2) + 1


def match_template(img_gray, template_gray, label):
    h, w = template_gray.shape

    x_center = FIXED_X[label]
    x1 = max(0, x_center - ROI_MARGIN)
    x2 = min(img_gray.shape[1], x_center + ROI_MARGIN)

    roi = img_gray[:, x1:x2]

    result = cv2.matchTemplate(roi, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.6:
        return None

    cy = max_loc[1] + h // 2

    cy = int(round(cy / 2) * 2)

    cx = FIXED_X[label]

    return (label, cx, cy)


def detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = {}

    for label, path in TEMPLATES.items():
        template = cv2.imread(path, 0)
        if template is None:
            continue

        r = match_template(img_gray, template, label)
        if r:
            _, cx, cy = r
            index = y_to_index(label, cy)

            result[label] = {
                "x": int(cx),
                "y": int(cy),
                "index": index
            }

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"] 

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    result = detect(img)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
