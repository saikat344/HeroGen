"""HeroGen lab: a local, read-only web page that shows one run end to end, including the 3D game body.

`python3 -m herogen.lab serve [--runs runs] [--port 8790]` starts a standard-library HTTP server on 127.0.0.1.
It serves the page in `static/`, a JSON index of the run folders (`/api/runs`, `/api/runs/<id>`) and image,
GLB and JSON files from inside the runs folder only. It never writes to the runs folder.
"""
