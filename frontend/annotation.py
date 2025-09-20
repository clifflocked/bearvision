
from flask import Flask, Response, request
from b2sum import b2checksum
from time import time
import cv2, json, os, sys

try:
    video = os.environ["VIDEO"]
except:
    print("No video specified. Using default")
    video = "./samples/sample.mp4"

video = cv2.VideoCapture(video)
currentframe = 113
checksum = ''
frametowrite = []
app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

@app.route("/", methods=['GET'])
def root():
    with open("frontend/static/index.html", 'r') as f:
        contents = f.read()
    return f"{contents}"

@app.route("/frame", methods=['GET'])
def frame():
    global currentframe, checksum, frametowrite
    currentframe += 100
    video.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
    success, newframe = video.read()
    if not success:
        return Response(status=404)
    
    print(success)

    checksum = b2checksum(f"{time()}")

    crop = newframe[20:40, 260:380]
    frametowrite = crop

    success, buf = cv2.imencode('.jpg', crop)
    if not success:
        return Response(status=500)
    
    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route("/framenum", methods=['GET'])
def framenum():
    global currentframe
    return f"{currentframe}"

@app.route("/favicon.ico", methods=['GET'])
def favicom():
    return ""

@app.route("/data", methods=['POST'])
def data():
    data = request.data.decode()
    print(data)
    with open(f"samples/scores/data/{checksum}.json", "a") as f:
        f.write(data)

    cv2.imwrite(f"samples/scores/images/{checksum}.jpg", frametowrite)

    return ""

if __name__ == "__main__":
    try:
        app.run(port=8080, debug=False)
    except:
        with open(json, "w") as f:
            f.write("\n]}")
