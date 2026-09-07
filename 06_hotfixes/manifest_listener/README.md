# Manifest listener

A small Flask app that serves the clips of a NeMo manifest one at a time, next to their transcripts, so a human can listen and flag the bad ones.

Stage 06's automated loop scores every clip by CER and regenerates what it can. This is the human-in-the-loop complement: it is how you check what the number cannot tell you, whether a clip that scores well actually *sounds* right. `06_hotfixes/03_collect_suspicious.py` does the same job in the other direction, copying the clips the gates already flagged into one folder for offline listening; this one lets you sweep a whole manifest, including the clips nothing flagged.

Vendored from a standalone sibling project, originally written against a Common Voice export layout. Nothing about it is specific to that: point the environment variables at any NeMo manifest.

## Run

```bash
docker compose up --build      # then open http://localhost:5007
```

Flagged clips are appended to `flagged.jsonl`, one JSON object per line, which the compose file bind-mounts read-write. Create it before the first run (`touch flagged.jsonl`) or Docker will make a directory of that name.

## Configuration

Every knob is an environment variable, set in `docker-compose.yml`:

| Variable | Default | Meaning |
|---|---|---|
| `MANIFEST` | `dataset/fr/nemo_manifest_ordered.json` | The NeMo manifest to serve. One JSON object per line. |
| `PATH_FROM` | `./export/` | Path prefix to strip from each row's `audio_filepath`. |
| `PATH_TO` | `./dataset/` | Prefix to substitute in its place, so manifests written against another machine's layout still resolve. |
| `FLAGGED` | `flagged.jsonl` | Where flags are appended. |
| `PORT` | `5000` | Port inside the container. **Read by `app.py` but not set in the compose `environment:` map**, since the default already matches the `EXPOSE` line and the `5007:5000` mapping. If you change it, add it to the compose file too or the port mapping will point at nothing. |

To use it against this project's own manifests, mount the audio directory and point `MANIFEST` at one of `data/NeMO_files/*.jsonl`. Those store `audio_filepath` relative to the manifest, so `PATH_FROM` and `PATH_TO` can usually be set to the same value to disable the rewrite.

## A caution

This app has **no authentication**, binds `0.0.0.0`, and `POST /api/flag` writes whatever JSON it is given straight to disk. It is a local inspection tool. Do not expose it to a network you do not control.

The one thing it does defend is path traversal: `resolve_audio` resolves the requested path and refuses anything that escapes the application root, so a crafted `?path=` cannot read arbitrary files.
