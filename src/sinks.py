"""
Output sinks: where each processed file's JSON record goes.

The pipeline produces one record per file (tagger.process_file); a sink
decides where it lands. Built in:

  sidecar  <file>.<ext>.json beside the original (the default). The full
           filename is kept, so clip.mp4 and clip.mov never collide. Also
           the processed-state store: is_done() reads it back.
  jsonl    every record appended to one file (easy to load into pandas or
           a database).

Anything else is a plugin: OUTPUT_SINKS="sidecar,mypkg.webhook:WebhookSink"
imports mypkg.webhook and instantiates WebhookSink() with no arguments.
docs/plugins.md has a worked example.

A sink that tracks processed state sets tracks_state = True and implements
is_done(); the library scanner skips a file when every state-tracking sink
reports it done. The output is JSON only: loading it into a database or
search index is left to whoever consumes it (the jsonl sink makes that
easy).
"""
import contextlib
import importlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config

# Per-task fields: a tags-only run keeps an earlier run's description
# fields, and vice versa.
TASK_FIELDS = {
    "tags": ("tags", "tags_en", "segments"),
    "description": ("description", "description_en"),
}
# Record fields that describe this run, not the file; never copied into a
# sidecar (the absolute path is not portable when a share is remounted).
_RUN_ONLY_FIELDS = {"path", "status", "tasks", "model", "prompt_set", "language",
                    "tool_version"}


class OutputSink:
    """Base class. Subclass, set `name`, implement write(); override
    is_done() and set tracks_state = True if the sink can answer "is this
    file already processed"."""
    name = "base"
    tracks_state = False

    def is_done(self, media: Path, tasks: set[str], *, language: str = "en",
                model: str | None = None) -> bool:
        """True when `media` already has every task in `tasks`, in
        `language`, for its current content. With `model`, only output
        produced by that model counts (the "re-run files tagged by an older
        model" setting); without it any model counts."""
        return False

    def write(self, media: Path, record: dict) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _source_stat(media: Path) -> dict:
    st = media.stat()
    return {"source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns}


# Read once: os.umask can only be queried by setting it, which would race
# file creation on other threads.
_UMASK = os.umask(0)
os.umask(_UMASK)


def _atomic_write_json(path: Path, data: dict) -> None:
    """tmp file in the same directory + os.replace, so a reader (or a crash
    mid-write) never sees a half-written sidecar. Keeps an existing file's
    permissions, else uses the normal umask (mkstemp's 0600 would hide
    sidecars from other users of a shared volume)."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        mode = 0o666 & ~_UMASK
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with contextlib.suppress(OSError):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class SidecarSink(OutputSink):
    """<file>.<ext>.json beside the original. When the folder is read-only
    (a locked share, a mounted card), the sidecar goes to a mirror under
    APP_SUPPORT_DIR/sidecars/<absolute path> instead, and that is logged."""
    name = "sidecar"
    tracks_state = True

    def __init__(self, mirror_root: Path | None = None):
        self.mirror_root = Path(mirror_root or config.APP_SUPPORT_DIR / "sidecars")

    def sidecar_path(self, media: Path) -> Path:
        return media.with_name(media.name + ".json")

    def mirror_path(self, media: Path) -> Path:
        absolute = media.resolve()
        return self.mirror_root / absolute.relative_to(absolute.anchor) \
            .with_name(absolute.name + ".json")

    def _candidates(self, media: Path):
        """Well-formed sidecars for `media`: beside it, then the mirror. The
        file is user-writable, so its shape is checked, not trusted: a
        corrupt or hand-mangled sidecar is skipped (the file is redone)."""
        for p in (self.sidecar_path(media), self.mirror_path(media)):
            try:
                data = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            tasks = data.get("tasks", {}) if isinstance(data, dict) else None
            if isinstance(tasks, dict) and all(isinstance(m, dict) for m in tasks.values()):
                yield data

    def read(self, media: Path) -> dict | None:
        """The sidecar for `media` (beside it, else the mirror), or None."""
        return next(self._candidates(media), None)

    def _current(self, media: Path, language: str) -> dict | None:
        """The sidecar that still describes this exact file content in this
        language. Both locations are checked: a stale sidecar beside the file
        (left from before the volume went read-only) must not hide a current
        mirror, or the file would be redone on every run."""
        try:
            source = _source_stat(media)
        except OSError:
            return None
        for data in self._candidates(media):
            if all(data.get(k) == v for k, v in source.items()) and \
                    data.get("language", "en") == language:
                return data
        return None

    def is_done(self, media, tasks, *, language="en", model=None):
        data = self._current(Path(media), language)
        if not data:
            return False
        have = data.get("tasks") or {}
        for task in tasks:
            if task not in have:
                return False
            if model and have[task].get("model") != model:
                return False
        return True

    def write(self, media, record):
        media = Path(media)
        if record.get("status") != "ok":
            return  # a failure stays a candidate; it is logged elsewhere
        language = record.get("language", "en")
        now = datetime.now(timezone.utc).isoformat()
        new_tasks = {t: {"model": record.get("model"), "prompt_set": record.get("prompt_set"),
                         "generated_at": now}
                     for t in record.get("tasks", ())}

        data = {"tool": "vlm-media-tagger", "tool_version": record.get("tool_version"),
                "file": media.name, **_source_stat(media), "language": language,
                "generated_at": now, "tasks": {}}
        previous = self._current(media, language)
        if previous:
            for task, meta in (previous.get("tasks") or {}).items():
                if task in new_tasks:
                    continue
                data["tasks"][task] = meta
                for field in TASK_FIELDS.get(task, ()):
                    if field in previous:
                        data[field] = previous[field]
        data["tasks"].update(new_tasks)
        task_fields = {f for fields in TASK_FIELDS.values() for f in fields}
        for k, v in record.items():
            if k in _RUN_ONLY_FIELDS:
                continue
            if k in task_fields and not any(k in TASK_FIELDS[t] for t in new_tasks):
                continue  # e.g. the empty description of a tags-only run
            data[k] = v

        target = self.sidecar_path(media)
        try:
            _atomic_write_json(target, data)
            return
        except PermissionError:
            pass
        except OSError as e:
            if e.errno not in (13, 30):  # EACCES, EROFS
                raise
        mirror = self.mirror_path(media)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(mirror, data)
        print(f"  NOTE: {media.parent} is read-only; sidecar written to {mirror}",
              file=sys.stderr, flush=True)


class JsonlSink(OutputSink):
    """Append every record as one JSON line. No processed state."""
    name = "jsonl"

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.OUTPUT_JSONL)
        self._f = None

    def write(self, media, record):
        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a")
        line = {**record, "path": str(Path(media)),
                "written_at": datetime.now(timezone.utc).isoformat()}
        self._f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


BUILTIN_SINKS = {"sidecar": SidecarSink, "jsonl": JsonlSink}


def load_sinks(spec: str | None = None) -> list[OutputSink]:
    """Comma-separated sink names (BUILTIN_SINKS) or module:Class specs.
    Raises ValueError on an unknown name or a class that isn't a sink, so a
    typo fails at start, not after an hour of processing."""
    spec = config.OUTPUT_SINKS if spec is None else spec
    sinks = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        if item in BUILTIN_SINKS:
            sinks.append(BUILTIN_SINKS[item]())
            continue
        if ":" not in item:
            raise ValueError(f"unknown output sink {item!r} (built in: "
                             f"{', '.join(BUILTIN_SINKS)}; plugins use module:Class)")
        module_name, _, class_name = item.partition(":")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"cannot load output sink {item!r}: {e}") from e
        if not (isinstance(cls, type) and issubclass(cls, OutputSink)):
            raise ValueError(f"{item!r} is not an OutputSink subclass")
        sinks.append(cls())
    return sinks


def is_done(sinks: list[OutputSink], media: Path, tasks: set[str], **kw) -> bool:
    """Done when every state-tracking sink says so; with none, never."""
    tracking = [s for s in sinks if s.tracks_state]
    return bool(tracking) and all(s.is_done(media, tasks, **kw) for s in tracking)
