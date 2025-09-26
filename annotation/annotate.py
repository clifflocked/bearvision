import os
import cv2
from random import randint
import streamlit as st
from streamlit import session_state as ss
from b2sum import b2checksum
from time import time, sleep

@st.cache_resource
def init(path="./samples/"):
    videos = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f.endswith(".mp4"))]
    print(f"Using the following videos in {path}:\n{videos}")
    return videos

class Session():
    def __init__(self):
        self.has_image = False

    def new_frame(self):
        # Pick random video, get random frame.
        self.video = videos[randint(0, len(videos) - 1)]
        video = cv2.VideoCapture(self.video)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        self.framenum = randint(5 * fps, frames - 30 * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, self.framenum)
        res, frame = video.read()
        self.frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)[20:40, 260:380]


videos = init()

col1, col2 = st.columns([1, 1])
with col2:
    user = st.text_input(" ", placeholder="Your name...", label_visibility="collapsed")

with col1:
    disp = st.empty()

if "session" not in ss:
    ss.session = Session()

if not ss.session.has_image:
    ss.session.has_image = True
    ss.session.new_frame()

image = ss.session.frame

disp.image(image, channels="BGR", width="stretch")

blue, red = st.columns(2)
with blue:
    bluescore = st.number_input("Blue", min_value=0, max_value=1000, step=1, format="%d", width=150)

with red:
    redscore = st.number_input("Red", min_value=0, max_value=1000, step=1, format="%d", width=150)

checksum = b2checksum(f"{time()} {image}")

btcol1, btcol2, debug, btcol3 = st.columns([1, 1, 3, 1])

if btcol1.button("Next frame", type="primary", width="stretch"):
    with open(f"./samples/scores/data/{checksum}.json", "a") as f:
        f.write(f"{{\"bluescore\":\"{int(bluescore)}\",\"redscore\":\"{int(redscore)}\"}}")

    cv2.imwrite(f"./samples/scores/images/{checksum}.jpg", image)
    ss.leaderboard[user] += 1

    ss.session.new_frame()
    image = ss.session.frame
    disp.image(image, channels="BGR", width="stretch")

if btcol2.button("Skip frame", type="tertiary", width="stretch"):
    ss.session.new_frame()
    image = ss.session.frame
    disp.image(image, channels="BGR", width="stretch")

if btcol3.button("Debug info", type="tertiary", width="stretch"):
    with debug:
        st.write(f"""Video: `{ss.session.video}`\n
Frame: `{ss.session.framenum}`\n
Checksum: `{checksum}`""")
