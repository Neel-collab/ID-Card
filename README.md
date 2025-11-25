# ID-Card
A python based software to cleanup the photos of ID Cards of university students and make it a deployable project 


[app.py](https://github.com/user-attachments/files/23736479/app.py)
from flask import Flask, render_template, request
from rembg import new_session, remove
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Create rembg session for newest versions (supports Python 3.12/3.13)
session = new_session("u2net")   # or "isnet-general-use" (better quality)


def resize_to_square(img, size=512, fill_color=(255, 255, 255)):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    new_img = Image.new("RGB", (size, size), fill_color)
    new_img.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return new_img


def enhance_photo(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")

    # Background remove (new rembg API)
    img_no_bg = remove(img, session=session)

    white_bg = Image.new("RGBA", img_no_bg.size, (255, 255, 255, 255))
    img_no_bg = Image.alpha_composite(white_bg, img_no_bg).convert("RGB")

    # Convert for OpenCV
    cv_img = cv2.cvtColor(np.array(img_no_bg), cv2.COLOR_RGB2BGR)

    # Face detection
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        margin = int(0.5 * h)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(cv_img.shape[1], x + w + margin), min(cv_img.shape[0], y + h + margin)
        cv_img = cv_img[y1:y2, x1:x2]

    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # Enhancements
    img_pil = ImageEnhance.Color(img_pil).enhance(1.2)
    img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.2)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)

    img_pil = resize_to_square(img_pil, 512)

    img_pil.save(output_path)
    return output_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():[readme.md](https://github.com/user-attachments/files/23736489/readme.md)

    file = request.files["file"]
    if not file:
        return "No file uploaded", 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")

    file.save(input_path)
    enhance_photo(input_path, result_path)

    return render_template("index.html", result_img=result_path)


if __name__ == "__main__":
    app.run(debug=True)





[Uploading readme.md…]()
steps to run:
```
pip install -r requirements.txt
```
```
python app.py
```




[requirements.txt](https://github.com/user-attachments/files/23736492/requirements.txt)
absl-py==2.3.1
addict==2.4.0
attrs==25.4.0
basicsr==1.4.2
blinker==1.9.0
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.3.0
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.3.3
cycler==0.12.1
facexlib==0.3.0
filelock==3.20.0
filterpy==1.4.5
Flask==3.1.2
flatbuffers==25.9.23
fonttools==4.60.1
fsspec==2025.10.0
future==1.0.0
gfpgan==1.3.8
grpcio==1.76.0
humanfriendly==10.0
idna==3.11
ImageIO==2.37.2
itsdangerous==2.2.0
Jinja2==3.1.6
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
kiwisolver==1.4.9
lazy_loader==0.4
llvmlite==0.45.1
lmdb==1.7.5
Markdown==3.10
MarkupSafe==3.0.3
matplotlib==3.10.7
mpmath==1.3.0
networkx==3.5
numba==0.62.1
numpy==2.2.6
onnxruntime==1.23.2
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
packaging==25.0
pillow==12.0.0
platformdirs==4.5.0
pooch==1.8.2
protobuf==6.33.0
PyMatting==1.1.14
pyparsing==3.2.5
pyreadline3==3.5.4
python-dateutil==2.9.0.post0
PyYAML==6.0.3
realesrgan==0.3.0
referencing==0.37.0
rembg==2.0.67
requests==2.32.5
rpds-py==0.28.0
scikit-image==0.25.2
scipy==1.16.3
six==1.17.0
sympy==1.14.0
tb-nightly==2.21.0a20251023
tensorboard-data-server==0.7.2
tifffile==2025.10.16
torch==2.9.0
torchvision==0.24.0
tqdm==4.67.1
typing_extensions==4.15.0
urllib3==2.5.0
Werkzeug==3.1.3
yapf==0.43.0




[index.html](https://github.com/user-attachments/files/23736495/index.html)
<!DOCTYPE html>
<html>
<head>
  <title>PhotoPro AI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-dark text-light text-center">
  <div class="container py-5">
    <h1 class="mb-4">📸 PhotoPro AI</h1>
    <form action="/process" method="POST" enctype="multipart/form-data" class="border rounded p-4 bg-light text-dark">
      <input type="file" name="file" accept="image/*" required class="form-control mb-3">
      <button class="btn btn-primary">Upload & Enhance</button>
    </form>

    {% if result_img %}
    <div class="mt-4">
      <h4>Enhanced Result:</h4>
      <img src="{{ result_img }}" class="img-fluid rounded shadow">
    </div>
    {% endif %}
  </div>
</body>
</html>

<img width="1024" height="1024" alt="1760893095 8572319" src="https://github.com/user-attachments/assets/875275b6-a4c4-4457-a55d-12b2ecea41bf" /> //Uploaded photo 

<img width="512" height="512" alt="result_1760893095 8572319" src="https://github.com/user-attachments/assets/66e39951-780c-4862-a4b1-20974ab889ca" /> //Enhanced result 






