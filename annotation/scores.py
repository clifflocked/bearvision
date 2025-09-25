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
        video = cv2.VideoCapture(videos[randint(0, len(videos) - 1)])
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        self.framenum = randint(5 * fps, frames - 30 * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, self.framenum)
        res, frame = video.read()
        self.frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)[20:40, 260:380]


videos = init()
title = st.empty()
disp = st.empty()

if "session" not in ss:
    ss.session = Session()

if not ss.session.has_image:
    ss.session.has_image = True
    ss.session.new_frame()

image = ss.session.frame

title.write(f"Frame number: {ss.session.framenum}")
disp.image(image, channels="BGR")

bluescore = st.number_input("Blue", value=None)
redscore = st.number_input("Red", value=None)
checksum = b2checksum(f"{time()} {image}")

if st.button("Next frame", type="primary"):
    with open(f"./samples/scores/data/{checksum}.jpg", "a") as f:
        f.write(f"{{\"bluescore\":\"{bluescore}\",\"redscore\":\"{redscore}\"}}")

    cv2.imwrite(f"./samples/scores/images/{checksum}.jpg", image)
    ss.session.new_frame()
    image = ss.session.frame
    title.write(f"Frame number: {ss.session.framenum}")
    disp.image(image, channels="BGR")

if st.button("Skip frame", type="tertiary"):
    ss.session.new_frame()
    image = ss.session.frame
    title.write(f"Frame number: {ss.session.framenum}")
    disp.image(image, channels="BGR")

