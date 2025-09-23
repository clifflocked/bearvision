# bearvision
A frontend and backend for automatic scouting of FRC

## Set up environment
Using `pip`:
```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install Flask opencv-python torch torchvision
$ mkdir -p samples/scores/{data,images}
```
Using `uv`:
```sh
$ uv venv .
$ source .venv/bin/activate
$ uv pip install Flask opencv-python torch torchvision
$ mkdir -p samples/scores/{data,images}
```

## Getting training data
It is recommended to use `yt-dlp` to download videos. Some examples are provided in `./samples/vids.txt`:
```sh
$ yt-dlp -a ./samples/vids.txt
```
Because YouTube is constantly changing their API, `yt-dlp` may fail if using a packaged version. Get the nightly version
 [here](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#installation).


## Run annotator
Make sure to set the `VIDEO` environment variable, otherwise it will default to `./samples/sample.mp4`
```sh
VIDEO=yourvideo python3 annotation/annotation.py
```
