from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

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

MATCH_THRESHOLD = 0.65


def match_template(img_gray, template_gray):

    h, w = template_gray.shape

    result = cv2.matchTemplate(
        img_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return {
        "score": float(max_val),
        "top_left": max_loc,
        "width": w,
        "height": h,
        "x": max_loc[0] + w // 2,
        "y": max_loc[1] + h // 2
    }


def detect(img):

    img_gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    debug_img = img.copy()

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

        score = result["score"]

        if score < MATCH_THRESHOLD:

            output[label] = {
                "x": result["x"],
                "y": result["y"],
                "score": round(score, 4),
                "warning": "一致率が低い"
            }

        else:

            output[label] = {
                "x": result["x"],
                "y": result["y"],
                "score": round(score, 4)
            }

        x = result["x"]
        y = result["y"]

        top_left = result["top_left"]

        w = result["width"]
        h = result["height"]

        cv2.rectangle(
            debug_img,
            top_left,
            (
                top_left[0] + w,
                top_left[1] + h
            ),
            (0, 255, 0),
            2
        )

        cv2.circle(
            debug_img,
            (x, y),
            15,
            (0, 0, 255),
            3
        )

        # ラベル＋score
        cv2.putText(
            debug_img,
            f"{label} {score:.3f}",
            (
                top_left[0],
                max(30, top_left[1] - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return output, debug_img


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:

        return jsonify({
            "error": "画像が選択されていません"
        })

    file = request.files["image"]

    save_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(save_path)

    img = cv2.imread(save_path)

    if img is None:

        return jsonify({
            "error": "画像の読み込みに失敗しました"
        })

    result, debug_img = detect(img)

    debug_path = os.path.join(
        UPLOAD_FOLDER,
        "debug.png"
    )

    cv2.imwrite(
        debug_path,
        debug_img
    )

    result["debug_image"] = (
        "/uploads/debug.png"
    )

    return jsonify(result)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
