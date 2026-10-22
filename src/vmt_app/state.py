"""
Persisted app settings: one JSON file in the app's Application Support
folder. Parsed as data only (json), never executed.
"""
import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

SETTINGS_FILENAME = "settings.json"
# Saves come from several threads (pywebview runs each UI call on its own).
_save_lock = threading.Lock()

DEFAULT_SETTINGS = {
    # Output language for tags and descriptions: "en" is a no-op, anything
    # else an entry in translation_models/manifest.json (downloaded on demand).
    "language": "en",
    # What to analyze: absolute path -> {"selected": bool}.
    # A path is a folder (scanned recursively) or a single file.
    "sources": {},
    # Network shares the app has seen sources on: mount point -> share URL
    # (smb://host/share). Used to reconnect them at launch (through macOS's
    # own mount, with Keychain credentials) and, opt-in, unmount on quit.
    "shares": {},
    "reconnect_shares": True,
    "unmount_on_quit": False,
    # Tasks: tags and/or a description (video summary / image description).
    "task_tags": True,
    "task_description": True,
    # Opt-in: redo files whose sidecar came from a different model. Never
    # automatic, so a model upgrade doesn't silently reprocess a library.
    "redo_other_model": False,
    # Keep DaVinci Resolve Studio running headless for the session (RAW
    # camera files). Off by default: only useful with RAW footage.
    "resolve_enabled": False,
    # Recurring analysis: re-run over the selected sources every N minutes
    # to pick up new files. Manual Analyze stays the primary mode.
    "auto_analyze": False,
    "auto_analyze_interval_minutes": 60,
    # Daily window during which a scheduled run won't START (manual runs
    # ignore it). start > end wraps midnight.
    "auto_pause_enabled": False,
    "auto_pause_start": "09:00",
    "auto_pause_end": "18:00",
    # Drop or redact profanity genuinely present in the media (off: it's
    # real content). See profanity.py.
    "filter_profanity": False,
    # batch_*.log files older than this are pruned at the start of a run.
    "log_retention_days": 30,
}


def load_settings(config_dir: Path) -> dict:
    path = config_dir / SETTINGS_FILENAME
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_SETTINGS))  # deep copy
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # Say so: silently falling back would look like every setting reset
        # for no reason.
        print(f"WARNING: could not read {path} ({e.__class__.__name__}: {e}) "
              f"— starting from default settings")
        _keep_corrupt(path)
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    if not isinstance(data, dict):
        print(f"WARNING: {path} does not contain a settings object "
              f"— starting from default settings")
        _keep_corrupt(path)
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    for key, value in data.items():
        if key not in DEFAULT_SETTINGS:
            continue
        default = DEFAULT_SETTINGS[key]
        # The file is user-writable: a value of the wrong type (null, a
        # number for a time string) would crash the UI later, so it falls
        # back to the default. bool is an int subclass: kept apart.
        if isinstance(value, type(default)) and \
                (isinstance(value, bool) == isinstance(default, bool)):
            merged[key] = value
        else:
            print(f"WARNING: settings: ignoring {key}={value!r} (expected "
                  f"{type(default).__name__})")
    # Only "selected" is kept: an older build's "hidden" flag is dropped, so
    # a source hidden there just shows again (unticked, as hiding left it).
    merged["sources"] = {k: {"selected": bool(v.get("selected"))}
                         for k, v in merged["sources"].items()
                         if isinstance(k, str) and isinstance(v, dict)}
    merged["shares"] = {k: v for k, v in merged["shares"].items()
                        if isinstance(k, str) and isinstance(v, str)}
    return merged


def _keep_corrupt(path: Path) -> None:
    """Move an unreadable settings file aside before defaults overwrite it,
    so sources can be recovered by hand."""
    backup = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    with contextlib.suppress(OSError):
        path.replace(backup)
        print(f"WARNING: kept the unreadable settings as {backup}")


def save_settings(config_dir: Path, settings: dict) -> None:
    """Best-effort and atomic: a failure is reported, never raised (callers
    are UI handlers, where an OSError would just look like a dead control,
    and the in-memory setting already applies for this session)."""
    path = config_dir / SETTINGS_FILENAME
    try:
        with _save_lock:
            text = json.dumps(settings, indent=2)  # snapshot under the lock
            config_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".settings.", suffix=".tmp", dir=config_dir)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(text)
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
    except (OSError, TypeError, ValueError, RuntimeError) as e:
        print(f"WARNING: could not save settings to {path} "
              f"({e.__class__.__name__}: {e}) — this session's changes will "
              f"be lost on quit")


def apply_to_config(settings: dict) -> None:
    """Push settings into the shared `config` module, which the pipeline
    reads at call time."""
    import config

    config.OUTPUT_LANGUAGE = settings.get("language", "en")
    config.FILTER_PROFANITY = bool(settings.get("filter_profanity", False))
