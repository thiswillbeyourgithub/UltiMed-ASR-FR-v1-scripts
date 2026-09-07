import json
import os
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory, abort

ROOT = Path(__file__).parent
MANIFEST = Path(os.environ.get("MANIFEST", "dataset/fr/nemo_manifest_ordered.json"))
# Manifest paths look like ./export/fr/audio_wav/X.wav but files live under ./dataset/...
PATH_FROM = os.environ.get("PATH_FROM", "./export/")
PATH_TO = os.environ.get("PATH_TO", "./dataset/")
FLAGGED = Path(os.environ.get("FLAGGED", "flagged.jsonl"))

app = Flask(__name__, static_folder=None)


def load_manifest():
    items = []
    with open(MANIFEST) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"idx": i, **obj})
    return items


def resolve_audio(rel: str) -> Path:
    if rel.startswith(PATH_FROM):
        rel = PATH_TO + rel[len(PATH_FROM):]
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT.resolve())):
        abort(403)
    return p


@app.get("/")
def index():
    return send_file(ROOT / "static" / "index.html")


@app.get("/api/manifest")
def api_manifest():
    return jsonify(load_manifest())


@app.get("/api/flagged")
def api_flagged():
    if not FLAGGED.exists():
        return jsonify([])
    out = []
    with open(FLAGGED) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return jsonify(out)


@app.post("/api/flag")
def api_flag():
    entry = request.get_json(force=True)
    with open(FLAGGED, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


@app.get("/audio")
def audio():
    rel = request.args.get("path", "")
    if not rel:
        abort(400)
    p = resolve_audio(rel)
    if not p.exists():
        abort(404)
    return send_file(p)


if __name__ == "__main__":
    # 0.0.0.0 inside the container is correct and stays: Docker has to reach the
    # process across the bridge network. Containment is done one level up, where
    # docker-compose.yml publishes the port on 127.0.0.1 only. There is no auth
    # here and it serves clinical-derived audio, so it must never be reachable
    # from outside the host.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
