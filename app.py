from flask import Flask, render_template, request, jsonify
import cv2
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

MATCH_THRESHOLD = 0.75


def match_template(img_gray, template_gray):

    h, w = template_gray.shape

    result = cv2.matchTemplate(
        img_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < MATCH_THRESHOLD:
        return None

    center_x = max_loc[0] + (w // 2)
    center_y = max_loc[1] + (h // 2)

    return {
        "x": int(center_x),
        "y": int(center_y),
        "score": round(float(max_val), 4)
    }


def detect(img):

    img_gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    output = {}

    for label, path in TEMPLATES.items():

        template = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if template is None:
            output[label] = {
                "error": f"{path} が見つかりません"
            }
            continue

        result = match_template(
            img_gray,
            template
        )

        if result:
            output[label] = result
        else:
            output[label] = {
                "error": "検出失敗"
            }

    return output


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    save_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(save_path)

    img = cv2.imread(save_path)

    result = detect(img)

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
