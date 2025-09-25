# bearvision
A frontend and backend for automatic scouting of FRC

## Set up environment
Using `pip`:
```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install opencv-python torch torchvision streamlit b2sum
$ mkdir -p samples/scores/{data,images}
```
Using `uv`:
```sh
$ uv venv .
$ source .venv/bin/activate
$ uv pip install opencv-python torch torchvision streamlit b2sum
$ mkdir -p samples/scores/{data,images}
```

## Getting training data
It is recommended to use `yt-dlp` to download videos. Some examples are provided in `vids.txt`:
```sh
$ yt-dlp -a vids.txt
```
Because YouTube is constantly changing their API, `yt-dlp` may fail if using a packaged version. Get the nightly version
 [here](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#installation).


## Run annotator
```sh
streamlit run annotation/scores.py
```
