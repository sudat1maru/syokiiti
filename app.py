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


def match_template(img_gray, template_gray, label):
    h, w = template_gray.shape

    x_center = FIXED_X[label]
    x1 = max(0, x_center - ROI_MARGIN)
    x2 = min(img_gray.shape[1], x_center + ROI_MARGIN)

    roi = img_gray[:, x1:x2]

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)

    result = cv2.matchTemplate(roi_blur, template_blur, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.6:
        return None

    cy = max_loc[1] + h // 2
    cx = FIXED_X[label]

    return (label, cx, cy)


def detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = []

    for label, path in TEMPLATES.items():
        template = cv2.imread(path, 0)
        if template is None:
            continue

        r = match_template(img_gray, template, label)
        if r:
            results.append(r)

    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")

    all_data = []

    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = cv2.imread(path)
        results = detect(img)

        for label, cx, cy in results:
            all_data.append({
                "file": file.filename,
                "label": label,
                "x": int(cx),
                "y": int(cy)
            })

    ranked = {}

    for label in TEMPLATES.keys():
        filtered = [d for d in all_data if d["label"] == label]

        sorted_list = sorted(filtered, key=lambda x: x["y"])

        for i, item in enumerate(sorted_list):
            item["rank"] = i + 1

        ranked[label] = sorted_list

    return jsonify(ranked)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
