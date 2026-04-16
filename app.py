from flask import Flask, render_template, request, jsonify
import cv2
import os

app = Flask(__name__)

from flask import Flask, render_template, request, send_file
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
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.7
    locations = np.where(result >= threshold)

    h, w = template_gray.shape

    points = []

    for pt in zip(*locations[::-1]):
        cx = pt[0] + w // 2
        cy = pt[1] + h // 2
        points.append((cx, cy, pt, w, h))

    filtered = []
    for cx, cy, pt, w, h in points:
        too_close = False
        for fx, fy, *_ in filtered:
            if abs(cx - fx) < 20 and abs(cy - fy) < 20:
                too_close = True
                break
        if not too_close:
            filtered.append((cx, cy, pt, w, h))

    return [(label, (cx, cy), pt, w, h) for cx, cy, pt, w, h in filtered]


def detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    all_results = []

    for label, path in TEMPLATES.items():
        template = cv2.imread(path, 0)

        if template is None:
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

    text = "\n".join([f"{r[0]} -> {r[1]}" for r in results])

    return f"<pre>{text}</pre><br><a href='/result'>結果画像を見る</a>"


@app.route("/result")
def result():
    return send_file("result.jpg", mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)