
from flask import Flask, Response, request
import cv2, json, os, sys

try:
    video = os.environ["VIDEO"]
except:
    print("No video specified. Using default")
    video = "./samples/sample.mp4"

try:
    csv = os.environ["CSV"]
except:
    print("No CSV specified. Using default.")
    csv = "./samples/sample.json"

open(csv, 'w').close()

with open(csv, "w") as f:
    f.write("{[\n")


video = cv2.VideoCapture(video)
currentframe = 113

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

@app.route("/frame", methods=['GET'])
def frame():
    global currentframe
    currentframe += 100
    video.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
    success, newframe = video.read()
    if not success:
        return Response(status=404)

    success, buf = cv2.imencode('.jpg', newframe)
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
    data = json.loads(request.data)
    f = open(csv, "a")
    f.write(json.dumps(data, sort_keys=True,indent=4))
    return "0"

if __name__ == "__main__":
    try:
        app.run()
    except:
        with open(csv, "w") as f:
            f.write("\n]}")
