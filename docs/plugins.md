# Output plugins

Every processed file produces one JSON record. Output sinks decide where it goes. Two are built in:

| Sink | Writes | Tracks what's done |
|---|---|---|
| `sidecar` (default) | `<file>.<ext>.json` beside each file (`clip.mp4` → `clip.mp4.json`); falls back to `~/Library/Application Support/VLM Media Tagger/sidecars/<absolute path>` when the folder is read-only | yes |
| `jsonl` | one line per file appended to `OUTPUT_JSONL` (default `data/records.jsonl`) | no |

Choose them with `OUTPUT_SINKS` (in `.env` or the environment) or `--sinks` on the CLI, comma-separated:

```bash
OUTPUT_SINKS=sidecar,jsonl
python src/tagger.py footage/ -r --sinks sidecar,mywebhook:WebhookSink --skip-done
```

## Writing a sink

A sink is a subclass of `sinks.OutputSink` with a no-argument constructor (read settings from the environment)
and a `write(media, record)` method. `module:Class` in `OUTPUT_SINKS` imports it, so the module only needs to be
importable (on `PYTHONPATH`, or installed in the same venv). This one POSTs every record to a URL:

```python
# mywebhook.py
import json
import os
import urllib.request

from sinks import OutputSink


class WebhookSink(OutputSink):
    name = "webhook"

    def __init__(self):
        self.url = os.environ["WEBHOOK_URL"]  # e.g. http://localhost:8000/media
        self.token = os.environ.get("WEBHOOK_TOKEN", "")

    def write(self, media, record):
        if record.get("status") != "ok":
            return  # or forward failures too
        body = json.dumps({"file": str(media), **record}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()

    def close(self):
        pass  # flush buffers / close connections here
```

A sink that raises is logged and skipped for that file; it never stops the run or the other sinks.

**Lifecycle.** One instance lives as long as the app (or one CLI run). The app calls `close()` at the end of
every run and keeps using the same instance, so `write()` can follow `close()`: reopen lazily on the next
write, as `JsonlSink` does. `is_done()` is called from background threads (the app's progress counts) while a
run may be writing, so keep it read-only and thread-safe.

## The record

The keys a sink receives (the sidecar stores the same data, minus the run-only `path`/`status` and plus its
own bookkeeping):

| Key | Meaning |
|---|---|
| `status` | `ok`, or why not: `undecodable`, `image-undecodable`, `resolve-unavailable`, `pipeline-failed`, `redundant-chunk`, `error` |
| `type` | `video` or `image` |
| `tags` | cleaned tags, in the output language |
| `description` | summary (video) or description (image); empty for a tags-only run |
| `segments` | video only: `[{tag, start_sec, end_sec, frame_count}, …]`, when each tag is on screen |
| `duration_sec`, `shot_count` | video only |
| `source` | `native`, or `resolve-proxy` when DaVinci Resolve decoded a RAW file |
| `tags_en`, `description_en` | the English originals, when the output language isn't English |
| `model`, `prompt_set`, `language`, `tool_version`, `tasks` | what produced this record |

## Tracking processed state

To let `--skip-done` (and the app) skip files your store already has, set `tracks_state = True` and implement
`is_done(media, tasks, *, language="en", model=None)`. A file is skipped only when *every* state-tracking sink
says it's done. The sidecar's rule is a good default: done when each requested task (`tags`, `description`) is
present, in that language, and the file's size and modification time still match. With `model` set, output from
a different model doesn't count; that's the opt-in "redo files from another model" setting, never automatic.
