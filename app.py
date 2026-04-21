from flask import Flask, render_template, request, jsonify, send_file
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

def match_template(img_gray, template_gray, label):
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)

    result = cv2.matchTemplate(img_blur, template_blur, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    h, w = template_gray.shape
    
    if max_val < 0.6:
        return []

    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2

    return [(label, (cx, cy), max_loc, w, h)]


def detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    all_results = []

    for label, path in TEMPLATES.items():
        template = cv2.imread(path, 0)

        if template is None:
            print(f"{path} 読み込み失敗")
            continue

        results = match_template(img_gray, template, label)
        all_results.extend(results)

    for label, (cx, cy), pt, w, h in all_results:
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 2)
        cv2.putText(img, label, (pt[0], pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, all_results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)

    result_img, results = detect(img)

    cv2.imwrite("result.jpg", result_img)

    output = {}
    for label, (cx, cy), pt, w, h in results:
        output[label] = [int(cx), int(cy)]

    return jsonify(output)


@app.route("/result")
def result():
    return send_file("result.jpg", mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
