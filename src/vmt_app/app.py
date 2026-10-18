"""
VLM Media Tagger — the macOS app (pywebview / WKWebView + an HTML front end
in resources/webui/).

Screens:
  Analyze        Sources (folders, scanned recursively, and single files,
                 local or on a network share) with "N / M analyzed" per
                 source, task toggles, Analyze/Stop, progress.
  Log            The live run log.
  Skipped Files  Files that can't be decoded here (undecodable.jsonl).
  Settings       General / Output / Automation / Integrations.
Plus the first-launch setup overlay (model download + engine start).

Results go through the output sinks (src/sinks.py): by default a JSON
sidecar beside each file, which is also how a later run knows what is
already done.

Threading model:
  - pywebview owns the macOS main thread (Cocoa run loop).
  - One background thread runs an asyncio loop hosting the long-running
    logic (setup, batch runs, the automation timer, Resolve lifecycle).
  - pywebview calls Api methods on its own worker threads. Everything that
    reads or changes settings runs on the backend loop (Api.call / submit),
    so there is exactly one writer; long operations report back through
    pushed UI events, not return values.
  - Source counts (a recursive scan, minutes on a big share) run on their
    own small thread pool, so they can never starve the default pool that
    engine start-up and Resolve's Quit() use.
  - Python → JS pushes go through Backend.push() → window.evaluate_js;
    events queue until JS calls init(), so startup pushes can't race the
    page load.

JS ↔ Python protocol: resources/webui/app.js (UI.dispatch event types and
pywebview.api method names; keep both sides in lockstep).
"""
import asyncio
import concurrent.futures
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import webview

import config
import library
import llama_manager
import resolve_proxy
import sinks as sinks_mod
import tagger
import translation_models

from . import batch, state, volumes

APP_NAME = config.APP_NAME
CONFIG_DIR = config.APP_SUPPORT_DIR
LOGS_DIR = Path.home() / "Library" / "Logs" / APP_NAME
WEBUI_DIR = Path(__file__).parent / "resources" / "webui"


def _begin_no_nap():
    """Keep macOS App Nap from throttling a long batch. A batch almost
    always runs in the background while the user does something else, and
    App Nap silently demotes a backgrounded process's scheduling partway
    through. Returns an activity token for _end_no_nap, or None."""
    try:
        from Foundation import NSProcessInfo
        NSActivityUserInitiated = 0x00FFFFFF
        NSActivityLatencyCritical = 0xFF00000000
        return NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
            NSActivityUserInitiated | NSActivityLatencyCritical, "Analyzing media")
    except Exception:
        return None


def _end_no_nap(token) -> None:
    if token is None:
        return
    with contextlib.suppress(Exception):
        from Foundation import NSProcessInfo
        NSProcessInfo.processInfo().endActivity_(token)


def _notify(title: str, message: str, subtitle: str | None = None) -> None:
    """Best-effort macOS banner (a run finishing is otherwise only visible in
    a log nobody is watching). osascript `display notification` never shows
    a blocking dialog, so it can't hang an unattended run. Argv-form, and
    backslash/quote escaping is what keeps the text from ending the
    AppleScript string; newlines are flattened because a literal can't span
    lines. Never raises."""

    def esc(text: str) -> str:
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        return " ".join(text.splitlines())

    script = f'display notification "{esc(message)}" with title "{esc(title)}"'
    if subtitle:
        script += f' subtitle "{esc(subtitle)}"'
    with contextlib.suppress(OSError):
        subprocess.Popen(["osascript", "-e", script],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _short(text, limit: int = 160) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _LineLogger:
    """stdout replacement during a run: buffers partial writes into whole
    lines, hands each to a callback (pushed to the UI) and mirrors it to the
    run's log file."""

    def __init__(self, on_line, log_file=None):
        self._on_line = on_line
        self._log_file = log_file
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(text)

    def _emit(self, line: str) -> None:
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()
        self._on_line(line)

    def flush(self) -> None:
        # Emit a partial trailing line (an in-place progress line like
        # resolve_proxy's "Rendering... ") rather than dropping it.
        if self._buf:
            line, self._buf = self._buf, ""
            self._emit(line)


def _available_languages() -> list[dict]:
    """English plus every language in the translation manifest."""
    langs = [{"code": "en", "name": "English"}]
    for code in sorted(translation_models._manifest().get("languages", {})):
        langs.append({"code": code, "name": translation_models.language_name(code) or code})
    return langs


class Backend:
    """App state and logic, no widget code. Long work runs on a dedicated
    asyncio loop thread; UI updates are pushed with evaluate_js."""

    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.settings = state.load_settings(CONFIG_DIR)
        codes = {lang["code"] for lang in _available_languages()}
        if self.settings.get("language") not in codes:
            self.settings["language"] = "en"
        state.apply_to_config(self.settings)
        state.save_settings(CONFIG_DIR, self.settings)

        self.window: "webview.Window | None" = None
        self.loop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

        self._ui_ready = False
        self._pending_events: list[dict] = []
        self._events_lock = threading.Lock()

        self._engine_ready = False
        # True while a batch (manual or scheduled) is in flight; only touched
        # on the loop thread, so a plain bool is race-free.
        self._batch_running = False
        self._stop_event: threading.Event | None = None
        # The automation loop task + the wall-clock time the next run is due.
        # Comparing wall-clock time each tick (instead of one long sleep)
        # gives a natural catch-up run after the Mac wakes from sleep.
        self._auto_task: asyncio.Task | None = None
        self._next_auto_run: datetime | None = None
        # (resolve_instance, ours_to_quit) once running, else None.
        self._resolve_handle: tuple | None = None
        # Serializes _start_resolve/_stop_resolve: the startup auto-start can
        # otherwise race a manual toggle and leak a Resolve process on quit.
        self._resolve_lock = asyncio.Lock()
        # The sidebar light: off | busy | ok | error. Enabled-but-not-started
        # yet (Resolve waits for the engine setup) already reads as busy.
        self._resolve_light = "busy" if self.settings.get("resolve_enabled") else "off"
        self._resolve_watch_task: asyncio.Task | None = None

        # Per-source (done, total), in memory only: recomputed on demand.
        # _progress_gen bumps whenever what "done" means changes (tasks,
        # language, model filter), so a count started before is dropped.
        self._source_progress: dict[str, tuple[int, int]] = {}
        self._progress_gen = 0
        self._progress_inflight: set[str] = set()
        self._progress_again: set[str] = set()
        self._scan_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                                thread_name_prefix="scan")
        # Shares this session mounted (the only ones unmount-on-quit touches;
        # a share the user mounted in Finder is left alone).
        self._reconnected: set[str] = set()

        try:
            self.output_sinks = sinks_mod.load_sinks()
        except ValueError as e:
            print(f"WARNING: {e} — falling back to the JSON sidecar")
            self.output_sinks = [sinks_mod.SidecarSink()]

    # -- UI push plumbing --------------------------------------------------

    def push(self, **event) -> None:
        with self._events_lock:
            if not self._ui_ready:
                self._pending_events.append(event)
                return
        self._eval(event)

    def _eval(self, event: dict) -> None:
        if self.window is None:
            return
        with contextlib.suppress(Exception):  # window mid-close
            self.window.evaluate_js(f"window.UI && UI.dispatch({json.dumps(event)})")

    def snapshot(self) -> dict:
        """Initial state for the front end; called once by JS (api.init())
        when the bridge is ready. Also flips the push queue live."""
        with self._events_lock:
            self._ui_ready = True
            pending, self._pending_events = self._pending_events, []
        for event in pending:
            self._eval(event)
        s = self.settings
        return {
            "version": config.APP_VERSION,
            "app_name": APP_NAME,
            "app_support_dir": str(config.APP_SUPPORT_DIR),
            "logs_dir": str(LOGS_DIR),
            "engine": self._engine_description(),
            "sinks": [sink.name for sink in self.output_sinks],
            "tasks": {"tags": bool(s["task_tags"]), "description": bool(s["task_description"])},
            "auto": {"enabled": bool(s["auto_analyze"]),
                     "minutes": int(s["auto_analyze_interval_minutes"])},
            "pause": {"enabled": bool(s["auto_pause_enabled"]),
                      "start": s["auto_pause_start"], "end": s["auto_pause_end"]},
            "resolve_enabled": bool(s["resolve_enabled"]),
            "resolve_light": self._resolve_light,
            "reconnect_shares": bool(s["reconnect_shares"]),
            "unmount_on_quit": bool(s["unmount_on_quit"]),
            "filter_profanity": bool(s["filter_profanity"]),
            "redo_other_model": bool(s["redo_other_model"]),
            "log_retention_days": int(s["log_retention_days"]),
            "language": s["language"],
            "languages": _available_languages(),
            **self._sources_payload(),
            "skipped": self._skipped_payload(),
        }

    @staticmethod
    def _engine_description() -> str:
        if config.VLM_BACKEND == "llama":
            where = "managed llama-server" if config.LLAMA_MANAGED else config.LLAMA_HOST
            return f"MiniCPM-V 4.7 ({where})"
        return f"{config.VISION_MODEL} (Ollama at {config.OLLAMA_HOST})"

    # -- lifecycle ---------------------------------------------------------

    def attach(self, window: "webview.Window") -> None:
        self.window = window

    def on_gui_started(self) -> None:
        """webview.start() callback, on a background thread once the Cocoa
        run loop is live. Starts the asyncio loop and the startup flow."""
        self._loop_thread = threading.Thread(target=self._run_loop, name="backend-loop",
                                             daemon=True)
        self._loop_thread.start()
        self.submit(self._startup_flow())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, fn, *args) -> None:
        """Run a plain method on the loop thread (the single settings writer)."""
        def _guarded():
            try:
                fn(*args)
            except Exception:
                traceback.print_exc()
        self.loop.call_soon_threadsafe(_guarded)

    def snapshot_on_loop(self) -> dict:
        async def _snap():
            return self.snapshot()
        return asyncio.run_coroutine_threadsafe(_snap(), self.loop).result(timeout=30)

    async def _startup_flow(self) -> None:
        """Reconnect remembered shares, then the setup gate (model + engine),
        then whatever automation was on last session."""
        if self.settings.get("reconnect_shares"):
            # In parallel with setup: an offline NAS must not delay the engine.
            asyncio.ensure_future(self._reconnect_then_count())
        else:
            self._spawn_progress_refresh_all()
        if not await self._first_launch_setup():
            return  # the overlay (with Retry) owns the window for now
        await self._post_setup_startup()

    async def _reconnect_then_count(self) -> None:
        await self._reconnect_shares()
        self._spawn_progress_refresh_all()

    async def _post_setup_startup(self) -> None:
        if self.settings.get("auto_analyze"):
            # First run a full interval from now, not at launch.
            self._start_auto_analyze()
        if self.settings.get("resolve_enabled"):
            await self._start_resolve()

    def shutdown(self) -> None:
        """Cleanup on quit: stop the batch cooperatively, quit Resolve (only
        if this session should), stop the managed engine (~23 GB resident),
        and unmount shares if opted in. Every step is bounded, so a wedged
        child can't hang Cmd+Q, and idempotent: it runs from the window's
        closing event and again after webview.start() returns."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._loop_thread is not None and self._loop_thread.is_alive():
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._stop_resolve(), self.loop).result(timeout=70)

        engine = threading.Thread(target=llama_manager.shutdown, daemon=True)
        engine.start()
        engine.join(timeout=15)

        self._scan_pool.shutdown(wait=False, cancel_futures=True)
        if self.settings.get("unmount_on_quit"):
            def _unmount_all():
                for point in list(self._reconnected):
                    volumes.unmount(point)
            t = threading.Thread(target=_unmount_all, daemon=True)
            t.start()
            t.join(timeout=30)

    # -- first-launch setup (model download + engine start) ------------------

    async def _first_launch_setup(self) -> bool:
        """Make sure the model files exist (downloading ~22 GB once, or
        linking them from a local Ollama store) and the engine is up. True
        when ready; False leaves the overlay showing an error and Retry."""
        managed_llama = config.VLM_BACKEND == "llama" and config.LLAMA_MANAGED
        if managed_llama and not await asyncio.to_thread(llama_manager.models_installed):
            self.push(type="setup", mode="progress", phase="Preparing…")

            def _progress(done: int, total: int, phase: str) -> None:
                self.push(type="setup", mode="progress", done=done, total=total, phase=phase)

            ok, msg = await asyncio.to_thread(llama_manager.fetch_models, _progress)
            if not ok:
                self.push(type="setup", mode="error", message=_short(msg, 300))
                return False
            _notify(APP_NAME, "AI model downloaded — ready to use.",
                    subtitle="First-time setup complete")

        self.push(type="engine", state="busy", text="Loading the AI model…")
        if managed_llama:
            # A cold load reads ~22 GB from disk: it can take a few minutes.
            self.push(type="setup", mode="progress", phase="Loading the AI model (can take a few minutes)…")
            ok, msg = await asyncio.to_thread(llama_manager.ensure_started)
        else:
            ok, msg = await asyncio.to_thread(config.vlm_status)
        if not ok:
            self.push(type="engine", state="error", text=f"✗ {_short(msg)}")
            self.push(type="setup", mode="error", message=_short(msg, 300))
            return False
        self._engine_ready = True
        self.push(type="engine", state="ok", text="✓ AI engine ready")
        self.push(type="setup", mode="hidden")
        return True

    async def setup_retry(self) -> None:
        if await self._first_launch_setup():
            await self._post_setup_startup()

    # -- sources (folders, files, network shares) -------------------------------

    def _source_row(self, path: str) -> dict:
        entry = self.settings["sources"].get(path, {})
        p = Path(path)
        if p.is_dir():
            kind = "folder"
        elif p.is_file():
            kind = "file"
        else:
            kind = "missing"
        share = self.settings["shares"]
        on_share = next((url for point, url in share.items()
                         if path == point or path.startswith(point.rstrip("/") + "/")), None)
        row = {"path": path, "name": p.name or path, "location": str(p.parent),
               "kind": kind, "selected": bool(entry.get("selected")),
               "share": on_share or ""}
        progress = self._source_progress.get(path)
        if progress is not None:
            row["progress"] = {"done": progress[0], "total": progress[1]}
        return row

    def _sources_payload(self) -> dict:
        names = sorted(self.settings["sources"], key=str.lower)
        hidden = [n for n in names if self.settings["sources"][n].get("hidden")]
        visible = [n for n in names if n not in hidden]
        selected = sum(1 for n in visible if self.settings["sources"][n].get("selected"))
        if not names:
            status = "Add a folder or files to analyze."
        else:
            status = f"{len(visible)} source(s) / {selected} selected"
            if hidden:
                status += f" / {len(hidden)} hidden"
        return {"sources": [self._source_row(n) for n in visible],
                "sources_hidden": [self._source_row(n) for n in hidden],
                "sources_status": status}

    def _push_sources(self) -> None:
        self.push(type="sources", **self._sources_payload())

    def _remember_share(self, path: str) -> None:
        """If `path` lives on a network share, remember the share's URL so
        it can be reconnected at the next launch. (One `mount` call.)"""
        found = volumes.share_for_path(path)
        if found:
            point, url = found
            self.settings["shares"][point] = url

    def add_sources(self, paths: list[str]) -> None:
        added, rejected = [], []
        for raw in paths or []:
            p = Path(raw).expanduser().resolve()
            path = str(p)
            if p.is_file() and p.suffix.lower() not in library.MEDIA_EXTS:
                rejected.append(p.name)
                continue
            if path not in self.settings["sources"]:
                self.settings["sources"][path] = {"selected": True, "hidden": False}
                added.append(path)
            self._remember_share(path)
        state.save_settings(CONFIG_DIR, self.settings)
        self._push_sources()
        if rejected:
            self.push(type="sources_status",
                      message=f"✗ not a supported media file: {_short(', '.join(rejected), 120)}")
        for path in added:
            self._spawn_progress_refresh(path)

    def pick_sources(self, folders: bool) -> None:
        """Runs on a pywebview worker thread: the dialog is modal to the
        window and pywebview marshals it onto the main thread."""
        if self.window is None:
            return
        kind = webview.FileDialog.FOLDER if folders else webview.FileDialog.OPEN
        exts = sorted(e.lstrip(".") for e in library.MEDIA_EXTS)
        file_types = () if folders else (
            "Media files (" + ";".join(f"*.{e};*.{e.upper()}" for e in exts) + ")",
            "All files (*.*)")
        try:
            chosen = self.window.create_file_dialog(kind, allow_multiple=True,
                                                    file_types=file_types)
        except Exception as e:
            self.push(type="sources_status", message=f"✗ could not open the dialog: {_short(e)}")
            return
        if chosen:
            self.call(self.add_sources, list(chosen))

    def remove_source(self, path: str) -> None:
        self.settings["sources"].pop(path, None)
        self._source_progress.pop(path, None)
        # Forget a share no remaining source lives on.
        for point in list(self.settings["shares"]):
            if not any(p == point or p.startswith(point.rstrip("/") + "/")
                       for p in self.settings["sources"]):
                del self.settings["shares"][point]
        state.save_settings(CONFIG_DIR, self.settings)
        self._push_sources()

    def set_source_selected(self, path: str, selected: bool) -> None:
        entry = self.settings["sources"].get(path)
        if entry is None:
            return
        entry["selected"] = bool(selected)
        state.save_settings(CONFIG_DIR, self.settings)
        self.push(type="sources_status", message=self._sources_payload()["sources_status"])

    def select_all_sources(self, value: bool) -> None:
        """Hidden sources are excluded: Select all must never sweep one back in."""
        for entry in self.settings["sources"].values():
            if not entry.get("hidden"):
                entry["selected"] = bool(value)
        state.save_settings(CONFIG_DIR, self.settings)
        self._push_sources()

    def set_source_hidden(self, path: str, hidden: bool) -> None:
        entry = self.settings["sources"].get(path)
        if entry is None:
            return
        entry["hidden"] = bool(hidden)
        if hidden:
            entry["selected"] = False
        state.save_settings(CONFIG_DIR, self.settings)
        self._push_sources()

    async def refresh_sources(self) -> None:
        """Reconnect missing shares (if enabled), then recount every source."""
        if self.settings.get("reconnect_shares"):
            await self._reconnect_shares()
        mounts = await asyncio.to_thread(volumes.network_mounts)
        for path in list(self.settings["sources"]):
            found = volumes.share_for_path(path, mounts)
            if found:
                self.settings["shares"][found[0]] = found[1]
        state.save_settings(CONFIG_DIR, self.settings)
        self._push_sources()
        self._spawn_progress_refresh_all()

    async def _reconnect_shares(self) -> None:
        """Ask macOS to remount remembered shares a selected source needs.
        Credentials come from the Keychain; the app never sees them."""
        mounts = await asyncio.to_thread(volumes.network_mounts)
        for point, url in list(self.settings["shares"].items()):
            if point in mounts:
                continue
            needed = any(p.startswith(point.rstrip("/") + "/") or p == point
                         for p, e in self.settings["sources"].items()
                         if e.get("selected") and not e.get("hidden"))
            if not needed:
                continue
            self.push(type="sources_status", message=f"Reconnecting {url}…")
            if await asyncio.to_thread(volumes.reconnect, url, point):
                self._reconnected.add(point)
            else:
                self.push(type="sources_status",
                          message=f"✗ could not reconnect {url} — connect it in Finder, then Refresh")
        self._push_sources()

    def connect_server(self, url: str) -> None:
        if not volumes.connect_dialog(url):
            self.push(type="sources_status",
                      message="✗ enter a server address like smb://nas.local")
            return
        self.push(type="sources_status",
                  message="Connecting… once the volume is mounted, use Add Folder to pick a folder on it.")

    def _spawn_progress_refresh(self, path: str) -> None:
        """Recount one source in the background. One scan per source at a
        time: a request while one is running queues exactly one rerun."""
        def _start():
            if path in self._progress_inflight:
                self._progress_again.add(path)
                return
            self._progress_inflight.add(path)
            asyncio.ensure_future(self._refresh_source_progress(path))
        self.loop.call_soon_threadsafe(_start)

    def _spawn_progress_refresh_all(self) -> None:
        for path, entry in list(self.settings["sources"].items()):
            if not entry.get("hidden"):
                self._spawn_progress_refresh(path)

    def _invalidate_progress(self) -> None:
        """What "done" means changed: drop counts (and any in flight)."""
        self._progress_gen += 1
        self._source_progress.clear()
        self._push_sources()
        self._spawn_progress_refresh_all()

    def _tasks(self) -> set[str]:
        tasks = set()
        if self.settings.get("task_tags"):
            tasks.add("tags")
        if self.settings.get("task_description"):
            tasks.add("description")
        return tasks

    async def _refresh_source_progress(self, path: str) -> None:
        """"N / M analyzed" for one source, for the current tasks. Runs on the
        scan pool; a result from an older generation is dropped."""
        gen = self._progress_gen
        tasks = self._tasks()
        language = self.settings["language"]
        model = config.VISION_MODEL if self.settings.get("redo_other_model") else None

        def _count():
            p = Path(path)
            if p.is_dir():
                r = library.scan(p, tasks, self.output_sinks, language=language, model=model)
                return r.done, r.total
            if p.is_file():
                done = sinks_mod.is_done(self.output_sinks, p, tasks, language=language, model=model)
                return int(done), 1
            return None

        result = None
        try:
            if tasks:
                result = await self.loop.run_in_executor(self._scan_pool, _count)
        except Exception:
            result = None
        finally:
            self._progress_inflight.discard(path)
        if result is not None and gen == self._progress_gen and path in self.settings["sources"]:
            self._source_progress[path] = result
            self.push(type="source_progress", path=path, done=result[0], total=result[1])
        if path in self._progress_again or gen != self._progress_gen:
            self._progress_again.discard(path)
            if path in self.settings["sources"]:
                self._spawn_progress_refresh(path)

    # -- tasks / automation settings ------------------------------------------

    def set_tasks(self, tags: bool, description: bool) -> None:
        changed = (bool(tags), bool(description)) != (
            bool(self.settings["task_tags"]), bool(self.settings["task_description"]))
        self.settings["task_tags"] = bool(tags)
        self.settings["task_description"] = bool(description)
        state.save_settings(CONFIG_DIR, self.settings)
        if changed:
            self._invalidate_progress()

    async def set_auto(self, enabled: bool) -> None:
        self.settings["auto_analyze"] = bool(enabled)
        state.save_settings(CONFIG_DIR, self.settings)
        if enabled:
            self._start_auto_analyze()
        else:
            # Not task.cancel(): the task may be awaiting an in-flight run.
            # The loop re-reads the setting each tick and exits on its own.
            self._next_auto_run = None
            self._update_auto_status()

    async def set_auto_interval(self, minutes: int) -> None:
        self.settings["auto_analyze_interval_minutes"] = max(1, int(minutes))
        state.save_settings(CONFIG_DIR, self.settings)
        if self.settings.get("auto_analyze"):
            self._next_auto_run = datetime.now() + self._auto_interval()
            self._update_auto_status()

    async def set_pause(self, enabled: bool, start: str, end: str) -> None:
        """A daily window during which a SCHEDULED run won't start. Manual
        Analyze ignores it, and a run already going when the window opens
        finishes. A due time inside the window fires on the first tick
        after it closes."""
        self.settings["auto_pause_enabled"] = bool(enabled)
        self.settings["auto_pause_start"] = self._valid_hhmm(start, "09:00")
        self.settings["auto_pause_end"] = self._valid_hhmm(end, "18:00")
        state.save_settings(CONFIG_DIR, self.settings)
        self._update_auto_status()

    @staticmethod
    def _valid_hhmm(value: str, fallback: str) -> str:
        try:
            hours, minutes = str(value).split(":")
            if 0 <= int(hours) < 24 and 0 <= int(minutes) < 60:
                return f"{int(hours):02d}:{int(minutes):02d}"
        except (ValueError, AttributeError):
            pass
        return fallback

    @staticmethod
    def _hhmm_to_minutes(value: str) -> int:
        hours, minutes = value.split(":")
        return int(hours) * 60 + int(minutes)

    def _in_pause_window(self, when: datetime) -> bool:
        if not self.settings.get("auto_pause_enabled"):
            return False
        start = self._hhmm_to_minutes(self.settings.get("auto_pause_start", "09:00"))
        end = self._hhmm_to_minutes(self.settings.get("auto_pause_end", "18:00"))
        if start == end:
            return False  # zero-length window, not "always paused"
        current = when.hour * 60 + when.minute
        if start < end:
            return start <= current < end
        return current >= start or current < end  # wraps midnight

    def _auto_interval(self) -> timedelta:
        return timedelta(minutes=int(self.settings.get("auto_analyze_interval_minutes", 60)))

    def _update_auto_status(self) -> None:
        if self.settings.get("auto_analyze") and self._next_auto_run is not None:
            text = f"Next automatic run: {self._next_auto_run.strftime('%a %H:%M')}"
            if self._in_pause_window(self._next_auto_run):
                text += (f" — inside the pause window, so it will start after "
                         f"{self.settings.get('auto_pause_end')}")
            self.push(type="auto", text=text)
        else:
            self.push(type="auto", text="")

    def _start_auto_analyze(self) -> None:
        """Next run a full interval out; make sure the loop task is alive.
        Idempotent."""
        self._next_auto_run = datetime.now() + self._auto_interval()
        if self._auto_task is None or self._auto_task.done():
            self._auto_task = asyncio.ensure_future(self._auto_analyze_loop())
        self._update_auto_status()

    async def _auto_analyze_loop(self) -> None:
        """Ticks every 60 s; fires a scheduled run once the due time has
        passed and nothing else is running. Short ticks + a wall-clock
        comparison survive Mac sleep (asyncio timers don't fire during
        sleep; the first tick after wake notices the missed time)."""
        while self.settings.get("auto_analyze"):
            await asyncio.sleep(60)
            if not self.settings.get("auto_analyze"):
                break
            due = self._next_auto_run
            if due is None or datetime.now() < due:
                continue
            if self._in_pause_window(datetime.now()) or self._batch_running:
                continue
            # Set before the run, so a run longer than the interval can't
            # queue an immediate second one behind it.
            self._next_auto_run = datetime.now() + self._auto_interval()
            self._update_auto_status()
            await self._run_analyze(scheduled=True)

    # -- the Analyze run -------------------------------------------------------

    _LOG_COUNT_CEILING = 500

    @classmethod
    def _prune_batch_logs(cls, log_dir: Path, retention_days: int = 30) -> None:
        """Delete batch_*.log files older than `retention_days`, plus a count
        ceiling in case a very short automation interval is configured."""
        try:
            logs = sorted(log_dir.glob("batch_*.log"))
        except OSError:
            return
        cutoff = time.time() - retention_days * 86400
        for old in logs[:-cls._LOG_COUNT_CEILING]:
            with contextlib.suppress(OSError):
                old.unlink()
        for log in logs[-cls._LOG_COUNT_CEILING:]:
            with contextlib.suppress(OSError):
                if log.stat().st_mtime < cutoff:
                    log.unlink()

    def _on_log_line(self, line: str) -> None:
        self.push(type="log", t=datetime.now().strftime("%H:%M:%S"), text=line)
        phase = batch.phase_for_line(line)
        if phase:
            self.push(type="phase", text=phase)

    async def analyze(self) -> None:
        await self._run_analyze(scheduled=False)

    async def _run_analyze(self, scheduled: bool) -> None:
        """One run, from the button or the timer. The only difference: a
        scheduled run that processed nothing doesn't notify (an hourly poll
        finding nothing new must not ping every hour)."""
        if self._batch_running:
            return
        sources = [p for p, e in self.settings["sources"].items()
                   if e.get("selected") and not e.get("hidden")]
        if not sources:
            self.push(type="run_status", cls="accent", text="Select at least one source first.")
            return
        tasks = self._tasks()
        if not tasks:
            self.push(type="run_status", cls="accent",
                      text="Select at least one task (tags or descriptions) first.")
            return

        self._batch_running = True
        # Created now, so Stop works during the engine check too.
        self._stop_event = threading.Event()
        # Revive the engine if it died mid-session (cheap when healthy).
        self.push(type="run_status", cls="", text="Checking the AI engine…")
        ok, msg = await asyncio.to_thread(self._engine_check)
        if ok and self._stop_event.is_set():
            ok, msg = False, "stopped"
        if not ok:
            self.push(type="run_status", cls="err", text=f"✗ {_short(msg)}")
            self.push(type="engine", state="error", text=f"✗ {_short(msg)}")
            self._batch_running = False
            return
        self.push(type="engine", state="ok", text="✓ AI engine ready")

        state.apply_to_config(self.settings)
        self.push(type="log_clear")
        self.push(type="run_state", running=True, stopping=False)
        self.push(type="progress", indeterminate=True, label="Scanning sources…")

        self._prune_batch_logs(LOGS_DIR, int(self.settings.get("log_retention_days", 30)))
        started_at = datetime.now()
        log_path = LOGS_DIR / f"batch_{started_at.strftime('%Y%m%d_%H%M%S')}.log"
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            # Tolerant encoding: a filename's odd bytes must not crash a run.
            log_file = open(log_path, "w", encoding="utf-8", errors="backslashreplace")
        except OSError as e:
            self._batch_running = False
            self.push(type="run_state", running=False, stopping=False)
            self.push(type="run_status", cls="err", text=f"✗ cannot write the run log: {_short(e)}")
            return
        log_file.write(f"Run started {started_at.isoformat()} "
                       f"({'scheduled' if scheduled else 'manual'})\n")
        log_file.write(f"Engine: {self._engine_description()}\n")
        log_file.write("Sources:\n" + "".join(f"  {s}\n" for s in sources) + "\n")
        logger = _LineLogger(self._on_log_line, log_file)

        def _on_progress(completed: int, total: int) -> None:
            self.push(type="progress", completed=completed, total=total)

        def _on_file(source: str, path: Path) -> None:
            try:
                shown = str(path.relative_to(source)) if Path(source).is_dir() else path.name
            except ValueError:
                shown = path.name
            self.push(type="progress_file",
                      text=_short(f"{Path(source).name}: {shown}", 100))

        def _on_source_progress(source: str, done: int, total: int) -> None:
            self._source_progress[source] = (done, total)
            self.push(type="source_progress", path=source, done=done, total=total)

        # Captured now: flipping the toggle mid-run shouldn't change an
        # in-flight run's RAW policy.
        resolve_enabled = bool(self.settings.get("resolve_enabled"))
        # Share the keep-warm Resolve connection with tagger's RAW fallback,
        # so it doesn't open (and later re-quit) a second handle. This object
        # stays the only owner of the Quit() decision (ours_to_quit False).
        if self._resolve_handle is not None:
            tagger._resolve_state = {"resolve": self._resolve_handle[0], "ours_to_quit": False}

        def _run() -> dict[str, int]:
            with contextlib.redirect_stdout(logger):
                return batch.run_batch(
                    sources, tasks=tasks, output_sinks=self.output_sinks,
                    language=self.settings["language"],
                    redo_other_model=bool(self.settings.get("redo_other_model")),
                    resolve_enabled=resolve_enabled, stop_event=self._stop_event,
                    on_progress=_on_progress, on_file=_on_file,
                    on_source_progress=_on_source_progress)

        stats: dict[str, int] = {}
        failure: str | None = None
        subtitle = "Analysis complete"
        nap_token = _begin_no_nap()
        try:
            stats = await asyncio.to_thread(_run)
        except batch.BatchAbort as e:
            failure = f"Aborted: {_short(e)}"
            subtitle = "Analysis aborted"
        except (Exception, SystemExit) as e:
            # SystemExit too: a pre-flight helper may sys.exit, and it would
            # otherwise vanish here with no error shown.
            self._on_log_line(f"ERROR: {e}")
            failure = f"Failed: {_short(e)}"
            subtitle = "Analysis failed"
        finally:
            _end_no_nap(nap_token)
            with contextlib.suppress(Exception):
                for sink in self.output_sinks:
                    if sink.name != "sidecar":
                        sink.close()
            self.push(type="run_state", running=False, stopping=False)
            self.push(type="phase", text="")
            self.push(type="progress_file", text="")
            if failure:
                summary = failure
                self.push(type="progress_reset")
            elif stats:
                summary = "Done: " + ", ".join(f"{v} {k}" for k, v in sorted(stats.items()))
            else:
                summary = "Done — nothing new to analyze."

            n_resolve = stats.get("resolve-unavailable", 0)
            if n_resolve:
                self._on_log_line("")
                self._on_log_line(
                    f"NOTE: {n_resolve} RAW file(s) skipped because DaVinci Resolve Studio "
                    f"wasn't available (off, not installed, or failed to launch) — check "
                    f"Settings → Integrations, then run again.")

            elapsed = datetime.now() - started_at
            elapsed_str = str(elapsed - timedelta(microseconds=elapsed.microseconds))
            log_file.write(f"\n{summary}\nTook {elapsed_str}\n")
            log_file.close()
            self._on_log_line("")
            self._on_log_line(f"Log saved to {log_path} (took {elapsed_str})")
            # A scheduled run notifies only when it did something new (not
            # just re-met files it can't decode); a manual run always does.
            news = any(k not in batch.UNDECODABLE_STATUSES for k in stats) or failure
            if news or not scheduled:
                _notify(APP_NAME, summary, subtitle=subtitle)
            self._batch_running = False
            self.push(type="run_status", cls="err" if failure else "ok", text=summary)
            self.refresh_skipped()

    def _engine_check(self) -> tuple[bool, str]:
        """Engine up and answering. ensure_started starts the managed server
        (and says yes for an external one), so ask the server itself too."""
        if config.VLM_BACKEND == "llama":
            ok, msg = llama_manager.ensure_started()
            if not ok or config.LLAMA_MANAGED:
                return ok, msg
        return config.vlm_status()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
            self.push(type="run_state", running=True, stopping=True)

    # -- skipped files ---------------------------------------------------------

    def _skipped_payload(self) -> dict:
        rows, bad = batch.read_undecodable()
        out = [{"source": Path(r.get("source", "")).name, "path": r.get("path", ""),
                "reason": _short(r.get("reason", ""), limit=100),
                "logged_at": r.get("logged_at", "")} for r in reversed(rows)]
        if not out and not bad:
            return {"rows": [], "count": 0, "message": "No skipped files."}
        suffix = f" ({bad} unparsable line(s) ignored)" if bad else ""
        return {"rows": out, "count": len(out), "message": f"{len(out)} skipped file(s){suffix}"}

    def refresh_skipped(self) -> None:
        self.push(type="skipped", **self._skipped_payload())

    def retry_skipped(self) -> None:
        """Every skipped file becomes a candidate again on the next run."""
        try:
            batch.clear_all_undecodable()
        except OSError as e:
            self.push(type="sources_status", message=f"✗ {_short(e)}")
        self.refresh_skipped()
        self._invalidate_progress()

    # -- settings ---------------------------------------------------------------

    def _set(self, key: str, value) -> None:
        self.settings[key] = value
        state.save_settings(CONFIG_DIR, self.settings)

    async def set_resolve(self, enabled: bool) -> None:
        self._set("resolve_enabled", bool(enabled))
        if enabled:
            await self._start_resolve()
        else:
            await self._stop_resolve()

    def set_unmount_on_quit(self, enabled: bool) -> None:
        self._set("unmount_on_quit", bool(enabled))

    def set_reconnect_shares(self, enabled: bool) -> None:
        self._set("reconnect_shares", bool(enabled))

    def set_filter_profanity(self, enabled: bool) -> None:
        self._set("filter_profanity", bool(enabled))
        state.apply_to_config(self.settings)

    def set_redo_other_model(self, enabled: bool) -> None:
        self._set("redo_other_model", bool(enabled))
        self._invalidate_progress()

    def set_log_retention_days(self, days: int) -> None:
        self._set("log_retention_days", max(1, int(days)))

    def set_language(self, lang: str) -> None:
        """English, or a language whose models are present, applies at once.
        A language still missing its models downloads first (progress in
        Settings) and applies only on success; a failure leaves it unchanged."""
        if lang not in {entry["code"] for entry in _available_languages()}:
            return
        if lang == "en" or translation_models.is_installed(lang):
            self._apply_language(lang)
            self.push(type="lang_download", mode="applied", lang=lang)
            return
        self.submit(self._download_and_set_language(lang))

    def _apply_language(self, lang: str) -> None:
        self._set("language", lang)
        config.OUTPUT_LANGUAGE = lang
        self._invalidate_progress()  # "done" depends on the language

    async def _download_and_set_language(self, lang: str) -> None:
        label = translation_models.language_name(lang) or lang
        self.push(type="lang_download", mode="progress", lang=lang, done=0, total=1,
                  phase=f"Downloading {label} translation…")

        def _progress(done: int, total: int, phase: str) -> None:
            self.push(type="lang_download", mode="progress", lang=lang,
                      done=done, total=total, phase=phase)

        ok, msg = await asyncio.to_thread(translation_models.download_language, lang, _progress)
        if ok:
            self._apply_language(lang)
            self.push(type="lang_download", mode="done", lang=lang)
            _notify(APP_NAME, f"{label} translation ready.", subtitle="Language pack downloaded")
        else:
            self.push(type="lang_download", mode="error", lang=lang, message=_short(msg, 200))

    # -- DaVinci Resolve Studio lifecycle ---------------------------------------

    def _set_resolve_status(self, text: str, cls: str = "", light: str = "off") -> None:
        self._resolve_light = light
        self.push(type="resolve", text=text, cls=cls, light=light)

    async def _start_resolve(self) -> None:
        async with self._resolve_lock:
            if self._resolve_handle is not None:
                return
            self._set_resolve_status("Starting…", light="busy")
            try:
                resolve, ours_to_quit = await asyncio.to_thread(resolve_proxy.ensure_resolve_running)
            except Exception as e:
                self._set_resolve_status(f"✗ {_short(e)}", "err", light="error")
                return
            self._resolve_handle = (resolve, ours_to_quit)
            # ours_to_quit is False when another live process (the CLI, or
            # another instance of this app) already owns this Resolve.
            self._set_resolve_status(
                "✓ Running" if ours_to_quit else "✓ Running (managed by another process)", "ok",
                light="ok")
            if self._resolve_watch_task is None or self._resolve_watch_task.done():
                self._resolve_watch_task = asyncio.ensure_future(self._watch_resolve())

    async def _watch_resolve(self) -> None:
        """Keeps the light honest: Resolve can crash or be quit by hand while
        we hold a handle. Clears the handle so toggling the option off and on
        starts a fresh one."""
        while self._resolve_handle is not None:
            await asyncio.sleep(config.RESOLVE_WATCH_SECONDS)
            if self._resolve_handle is None:
                return
            if await asyncio.to_thread(resolve_proxy.resolve_running):
                continue
            async with self._resolve_lock:
                if self._resolve_handle is None:  # stopped meanwhile: not a crash
                    return
                _, ours_to_quit = self._resolve_handle
                self._resolve_handle = None
                if ours_to_quit:
                    resolve_proxy.release_ownership()
                self._set_resolve_status(
                    "✗ DaVinci Resolve Studio is no longer running — turn this off and on to restart it",
                    "err", light="error")
            return

    async def _stop_resolve(self) -> None:
        async with self._resolve_lock:
            timed_out = False
            failed = False  # Quit() raised: also "may still be running"
            if self._resolve_handle is not None:
                resolve, ours_to_quit = self._resolve_handle
                self._resolve_handle = None
                if ours_to_quit:
                    self._set_resolve_status("Stopping…", light="busy")
                    started = time.time()
                    try:
                        # Bounded, but awaited: on quit the process is about
                        # to exit, and an unawaited call would be killed
                        # mid-flight, orphaning Resolve.
                        await asyncio.wait_for(asyncio.to_thread(resolve.Quit), timeout=30)
                        print(f"  Resolve Quit() took {time.time() - started:.1f}s")
                    except asyncio.TimeoutError:
                        timed_out = True
                        print(f"  Resolve Quit() timed out after {time.time() - started:.1f}s")
                    except Exception as e:
                        failed = True
                        print(f"  Resolve Quit() failed after {time.time() - started:.1f}s: "
                              f"{e.__class__.__name__}: {e}")
                    # Release either way, so a later session can reclaim a
                    # wedged instance as an orphan.
                    resolve_proxy.release_ownership()

            # Safety net: tagger's RAW fallback may have launched Resolve on
            # its own (a RAW file hit before the keep-warm start finished).
            # A cheap no-op when it never connected.
            if not timed_out:
                try:
                    await asyncio.wait_for(asyncio.to_thread(tagger.shutdown_resolve), timeout=30)
                except asyncio.TimeoutError:
                    timed_out = True
                except Exception as e:
                    failed = True
                    print(f"  tagger.shutdown_resolve failed: {e.__class__.__name__}: {e}")

            if timed_out:
                self._set_resolve_status(
                    "✗ Resolve did not quit within 30s — it may still be running", "err",
                    light="error")
            elif failed:
                self._set_resolve_status(
                    "✗ Resolve did not quit cleanly — it may still be running (see the log)", "err",
                    light="error")
            else:
                self._set_resolve_status("")

    # -- Finder ------------------------------------------------------------------

    @staticmethod
    def _open(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def reveal_app_support_dir(self) -> None:
        self._open(config.APP_SUPPORT_DIR)

    def reveal_logs(self) -> None:
        self._open(LOGS_DIR)

    def reveal_path(self, path: str) -> None:
        """Show a source or skipped file in Finder (only paths the app knows)."""
        known = set(self.settings["sources"]) | {r.get("path") for r in batch.read_undecodable()[0]}
        if path in known and Path(path).exists():
            with contextlib.suppress(OSError):
                subprocess.Popen(["open", "-R", path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Api:
    """The pywebview js_api surface: thin wrappers around Backend. Method
    names must match the api() calls in resources/webui/app.js."""

    def __init__(self, backend: Backend):
        self._backend = backend
    # Methods that touch settings go through Backend.call (the loop thread);
    # reveal/stop/refresh_skipped only read files or set an Event.

    def init(self) -> dict:
        return self._backend.snapshot_on_loop()

    def add_folders(self) -> None:
        self._backend.pick_sources(folders=True)

    def add_files(self) -> None:
        self._backend.pick_sources(folders=False)

    def remove_source(self, path: str) -> None:
        self._backend.call(self._backend.remove_source, path)

    def set_source_selected(self, path: str, selected: bool) -> None:
        self._backend.call(self._backend.set_source_selected, path, selected)

    def select_all_sources(self, value: bool) -> None:
        self._backend.call(self._backend.select_all_sources, value)

    def set_source_hidden(self, path: str, hidden: bool) -> None:
        self._backend.call(self._backend.set_source_hidden, path, hidden)

    def refresh_sources(self) -> None:
        self._backend.submit(self._backend.refresh_sources())

    def connect_server(self, url: str) -> None:
        self._backend.connect_server(url)

    def set_tasks(self, tags: bool, description: bool) -> None:
        self._backend.call(self._backend.set_tasks, tags, description)

    def analyze(self) -> None:
        self._backend.submit(self._backend.analyze())

    def stop(self) -> None:
        self._backend.stop()

    def refresh_skipped(self) -> None:
        self._backend.refresh_skipped()

    def retry_skipped(self) -> None:
        self._backend.call(self._backend.retry_skipped)

    def set_auto(self, enabled: bool) -> None:
        self._backend.submit(self._backend.set_auto(enabled))

    def set_auto_interval(self, minutes: int) -> None:
        self._backend.submit(self._backend.set_auto_interval(minutes))

    def set_pause(self, enabled: bool, start: str, end: str) -> None:
        self._backend.submit(self._backend.set_pause(enabled, start, end))

    def set_resolve(self, enabled: bool) -> None:
        self._backend.submit(self._backend.set_resolve(enabled))

    def set_unmount_on_quit(self, enabled: bool) -> None:
        self._backend.call(self._backend.set_unmount_on_quit, enabled)

    def set_reconnect_shares(self, enabled: bool) -> None:
        self._backend.call(self._backend.set_reconnect_shares, enabled)

    def set_filter_profanity(self, enabled: bool) -> None:
        self._backend.call(self._backend.set_filter_profanity, enabled)

    def set_redo_other_model(self, enabled: bool) -> None:
        self._backend.call(self._backend.set_redo_other_model, enabled)

    def set_log_retention_days(self, days: int) -> None:
        self._backend.call(self._backend.set_log_retention_days, days)

    def set_language(self, lang: str) -> None:
        self._backend.call(self._backend.set_language, lang)

    def reveal_app_support_dir(self) -> None:
        self._backend.reveal_app_support_dir()

    def reveal_logs(self) -> None:
        self._backend.reveal_logs()

    def reveal_path(self, path: str) -> None:
        self._backend.call(self._backend.reveal_path, path)

    def setup_retry(self) -> None:
        self._backend.submit(self._backend.setup_retry())


def _install_crash_log() -> None:
    """Record uncaught exceptions outside a run (which has its own log) in
    LOGS_DIR/crash.log, after the default hook, which still runs. Covers
    background threads too (threading.excepthook)."""
    default_hook = sys.excepthook

    def _hook(exc_type, exc, tb) -> None:
        default_hook(exc_type, exc, tb)
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOGS_DIR / "crash.log", "a") as f:
                f.write(f"\n--- {datetime.now().isoformat()} ---\n")
                traceback.print_exception(exc_type, exc, tb, file=f)
        except Exception:
            pass  # a broken crash logger must never mask the crash

    sys.excepthook = _hook
    threading.excepthook = lambda args: _hook(args.exc_type, args.exc_value, args.exc_traceback)


def _default_window_size() -> tuple[int, int]:
    """The tallest comfortable height the actual display allows (a 13"
    laptop leaves ~790 pt; a 5K display 1300+), capped at 1000. Falls back
    to a static size if screen info is unavailable."""
    width, height = 1120, 840
    with contextlib.suppress(Exception):
        height = max(660, min(1000, int(webview.screens[0].height) - 110))
    return width, height


# Held for the process lifetime; the kernel drops it on any exit (crash and
# SIGKILL included), so a stale lock can't block the next launch.
_instance_lock_fd: int | None = None


def _acquire_single_instance() -> bool:
    """One app per profile: a second instance would adopt the first one's
    llama-server and stop it on quit. False means another instance holds the
    lock; it is brought to the front."""
    global _instance_lock_fd
    import fcntl
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(CONFIG_DIR / "app.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        with contextlib.suppress(Exception):
            other = int(os.read(fd, 32).decode().strip() or 0)
            from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication
            running = NSRunningApplication.runningApplicationWithProcessIdentifier_(other)
            if running is not None:
                running.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        os.close(fd)
        return False
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    _instance_lock_fd = fd
    return True


def main() -> None:
    _install_crash_log()
    if not _acquire_single_instance():
        print(f"{APP_NAME} is already running — brought it to the front.")
        os._exit(0)
    backend = Backend()
    width, height = _default_window_size()
    window = webview.create_window(APP_NAME, url=str(WEBUI_DIR / "index.html"),
                                   js_api=Api(backend), width=width, height=height,
                                   min_size=(940, 660))
    backend.attach(window)
    # Cmd+Q does NOT return control to code after webview.start(): Cocoa's
    # terminate sequence exits at the ObjC level. window.events.closing fires
    # synchronously on both Quit and window-close, so cleanup hooks there.
    # But it runs ON the main thread, and blocking it stalls the run loop
    # DaVinci Resolve's scripting bridge needs to deliver Quit()'s reply. So:
    # cancel this close (return False), run shutdown() on a background
    # thread, then hard-exit.
    quit_state = {"in_progress": False}

    def _on_closing() -> bool:
        if quit_state["in_progress"]:
            return False
        quit_state["in_progress"] = True

        def _shutdown_then_exit() -> None:
            backend.shutdown()
            os._exit(0)

        threading.Thread(target=_shutdown_then_exit, daemon=True).start()
        return False

    window.events.closing += _on_closing
    # Right-click → Inspect for front-end work: VMT_WEBVIEW_DEBUG=1.
    webview.start(backend.on_gui_started, debug=bool(os.environ.get("VMT_WEBVIEW_DEBUG")))
    backend.shutdown()  # safety net if a path ever skips the event above
    # Hard exit: a run mid-VLM-call sits in a non-daemon executor thread that
    # Python would otherwise join, making Quit look hung for minutes.
    os._exit(0)
