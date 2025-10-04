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

with st.form("ColorNumberInputs", clear_on_submit=True):  # Creates a form which holds both input spaces, as well as the buttons
    blue, red = st.columns(2)  # list of columns for the 2 input spaces
    with blue:
        bluescore = st.number_input("Blue", min_value=0, max_value=1000, value=None, step=1, format="%d", width=150)
    with red:
        redscore = st.number_input("Red", min_value=0, max_value=1000, value=None, step=1, format="%d", width=150)

    checksum = b2checksum(f"{bluescore} {redscore}")  # Honestly idk this was u segen
    submit_button, skip_button = st.columns([1, 1])  # List of columns for the 2 buttons under the inputs

    submitted = submit_button.form_submit_button("Next frame", type="primary", key = "submitButton", width="stretch") #creates skip button
    if submitted:  # Runs when skip button pressed
        if bluescore == None or redscore == None:  # Make sure the user didnt leave the boxes blank
            st.write("both score boxes must have a value")
        else:
            with open(f"./samples/scores/data/{checksum}.json", "a") as f:
                f.write(f"{{\"bluescore\":\"{int(bluescore)}\",\"redscore\":\"{int(redscore)}\"}}")

            cv2.imwrite(f"./samples/scores/images/{checksum}.jpg", image)
            ss.leaderboard[user] += 1

            ss.session.new_frame()
            image = ss.session.frame
            disp.image(image, channels="BGR", width="stretch")
    skipped = skip_button.form_submit_button("SkipFrame", type="tertiary", key="skipButton", width="stretch") #runs skip button
    if skipped:  # Function run when skip button pressed
        ss.session.new_frame()
        image = ss.session.frame
        disp.image(image, channels="BGR", width="stretch")

debug_text, debug_button = st.columns([3, 1])  # New row for the debug things

if debug_button.button("Debug info", type="tertiary", width="stretch"):
    with debug_text:
        st.write(f"""Video: `{ss.session.video}`\n
Frame: `{ss.session.framenum}`\n
Checksum: `{checksum}`""")
