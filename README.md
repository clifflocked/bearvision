# bearvision
A frontend and backend for automatic scouting of FRC

## Set up environment
```sh
$ mkdir .venv
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install Flask opencv-python
$ mkdir samples/
$ touch samples/sample.json
```

## Run annotator
Make sure to set the `VIDEO` and `ANNOTATIONS` environment variables accordingly, otherwise they will be set to their defaults.
```sh
$ PYTHONDONTWRITEBYTECODE=1 VIDEO=yourvideo ANNOTATIONS=yourannotations flask --app frontend/annotation.py run
```
