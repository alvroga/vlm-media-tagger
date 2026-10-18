"""The app's backend without a window or a model: batch runs (resume, mixed
sources, skipped-files list, circuit breaker, stop), settings persistence,
network-share parsing, and Backend's source management + push queue."""
import json
import threading

import pytest

import config
import sinks
import tagger
from vmt_app import batch, state, volumes


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", d)
    monkeypatch.setattr(config, "vlm_status", lambda model=None: (True, "ok"))
    # run_batch sets this module global; restore it so other test files
    # don't depend on the order they run in.
    monkeypatch.setattr(tagger, "RESOLVE_ALLOWED", tagger.RESOLVE_ALLOWED)
    return d


@pytest.fixture
def fake_process(monkeypatch):
    """tagger.process_file stand-in: status per file name, records calls."""
    calls, statuses = [], {}

    def fake(path, **kw):
        calls.append(path.name)
        status = statuses.get(path.name, "ok")
        rec = {"path": str(path), "type": "video", "status": status, "elapsed_sec": 0.1}
        if status == "ok":
            rec.update(tags=["car"], description="A car." if kw["describe"] else "")
        else:
            rec["error"] = f"{status} for {path.name}"
        return rec
    monkeypatch.setattr(tagger, "process_file", fake)
    return calls, statuses


def _media(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).parent.mkdir(parents=True, exist_ok=True)
        (folder / n).write_bytes(n.encode())
    return folder


def _run(sources, out_sinks, **kw):
    kw.setdefault("tasks", {"tags", "description"})
    return batch.run_batch([str(s) for s in sources], output_sinks=out_sinks, **kw)


def test_run_writes_sidecars_then_resumes(tmp_path, data_dir, fake_process):
    calls, _ = fake_process
    lib = _media(tmp_path / "lib", "a.mp4", "b.mov")
    sink = sinks.SidecarSink(mirror_root=tmp_path / "mirror")
    progress = []
    stats = _run([lib], [sink], on_source_progress=lambda s, d, t: progress.append((d, t)))
    assert stats == {"ok": 2}
    assert sorted(calls) == ["a.mp4", "b.mov"]
    assert progress[0] == (0, 2) and progress[-1] == (2, 2)
    side = json.loads((lib / "a.mp4.json").read_text())
    assert side["tasks"]["tags"]["model"] == config.VISION_MODEL

    calls.clear()
    assert _run([lib], [sink]) == {}
    assert calls == []


def test_files_and_folders_mixed_and_deduplicated(tmp_path, data_dir, fake_process):
    calls, _ = fake_process
    lib = _media(tmp_path / "lib", "a.mp4", "sub/b.mov")
    single = lib / "a.mp4"
    _run([lib, single, lib / "sub"], [sinks.SidecarSink(mirror_root=tmp_path / "m")])
    assert sorted(calls) == ["a.mp4", "b.mov"]  # each file once


def test_missing_source_is_skipped_not_fatal(tmp_path, data_dir, fake_process, capsys):
    calls, _ = fake_process
    lib = _media(tmp_path / "lib", "a.mp4")
    _run([tmp_path / "unplugged", lib], [sinks.SidecarSink(mirror_root=tmp_path / "m")])
    assert calls == ["a.mp4"]
    assert "not available" in capsys.readouterr().out


def test_skipped_list_logs_holds_back_and_clears(tmp_path, data_dir, fake_process):
    calls, statuses = fake_process
    lib = _media(tmp_path / "lib", "clip.braw", "ok.mp4")
    statuses["clip.braw"] = "resolve-unavailable"
    sink = sinks.SidecarSink(mirror_root=tmp_path / "m")
    _run([lib], [sink])
    rows, bad = batch.read_undecodable()
    assert bad == 0 and [r["path"] for r in rows] == [str(lib / "clip.braw")]
    assert rows[0]["source"] == str(lib)
    assert not (lib / "clip.braw.json").exists()

    calls.clear()
    assert _run([lib], [sink]) == {}              # unchanged: not retried
    assert calls == []

    (lib / "clip.braw").write_bytes(b"re-exported")  # changed: tried again
    assert _run([lib], [sink]) == {"resolve-unavailable": 1}
    assert len(batch.read_undecodable()[0]) == 1      # still one row per file

    batch.clear_all_undecodable()                     # the Retry button
    statuses["clip.braw"] = "ok"
    assert _run([lib], [sink]) == {"ok": 1}
    assert batch.read_undecodable()[0] == []


def test_corrupt_skipped_log_does_not_break_anything(tmp_path, data_dir, fake_process):
    (data_dir / batch.UNDECODABLE_LOG).write_bytes(b'\xff\xfe not json\n{"path": "/x"}\n')
    rows, bad = batch.read_undecodable()
    assert bad == 1 and rows == [{"path": "/x"}]
    lib = _media(tmp_path / "lib", "a.mp4")
    assert _run([lib], [sinks.SidecarSink(mirror_root=tmp_path / "m")]) == {"ok": 1}


def test_circuit_breaker_stops_a_systemic_failure(tmp_path, data_dir, fake_process, capsys):
    calls, statuses = fake_process
    names = [f"{i}.mp4" for i in range(6)]
    lib = _media(tmp_path / "lib", *names)
    for n in names:
        statuses[n] = "pipeline-failed"
    stats = _run([lib], [sinks.SidecarSink(mirror_root=tmp_path / "m")])
    assert stats == {"pipeline-failed": 3}
    assert "Aborting" in capsys.readouterr().out


def test_isolated_failures_do_not_trip_the_breaker(tmp_path, data_dir, fake_process):
    calls, statuses = fake_process
    names = [f"{i}.mp4" for i in range(7)]
    lib = _media(tmp_path / "lib", *names)
    for n in ("0.mp4", "1.mp4", "3.mp4", "4.mp4", "6.mp4"):  # never 3 in a row
        statuses[n] = "pipeline-failed"
    stats = _run([lib], [sinks.SidecarSink(mirror_root=tmp_path / "m")])
    assert sum(stats.values()) == 7 and stats["ok"] == 2


def test_failed_file_workdir_is_swept(tmp_path, data_dir, monkeypatch):
    lib = _media(tmp_path / "lib", "a.mp4")
    work = data_dir / config.video_id_from_path(str(lib / "a.mp4"), unique=True)

    def boom(path, **kw):
        work.mkdir()
        (work / "cuts.json").write_text("{}")
        raise RuntimeError("crashed mid-file")
    monkeypatch.setattr(tagger, "process_file", boom)
    _run([lib], [])
    assert not work.exists()


def test_systemexit_is_contained_and_feeds_the_breaker(tmp_path, data_dir, monkeypatch):
    lib = _media(tmp_path / "lib", *[f"{i}.jpg" for i in range(5)])
    calls = []

    def exits(path, **kw):
        calls.append(path.name)
        raise SystemExit("engine went away")
    monkeypatch.setattr(tagger, "process_file", exits)
    stats = _run([lib], [])
    assert stats == {"error": 3} and len(calls) == 3


def test_unexpected_exception_is_contained(tmp_path, data_dir, monkeypatch):
    lib = _media(tmp_path / "lib", "a.mp4", "b.mp4")

    def boom(path, **kw):
        if path.name == "a.mp4":
            raise RuntimeError("decoder blew up")
        return {"path": str(path), "status": "ok", "tags": [], "description": ""}
    monkeypatch.setattr(tagger, "process_file", boom)
    stats = _run([lib], [sinks.SidecarSink(mirror_root=tmp_path / "m")])
    assert stats == {"error": 1, "ok": 1}


def test_stop_between_files(tmp_path, data_dir, fake_process):
    calls, _ = fake_process
    lib = _media(tmp_path / "lib", "a.mp4", "b.mp4", "c.mp4")
    stop = threading.Event()
    _run([lib], [sinks.SidecarSink(mirror_root=tmp_path / "m")], stop_event=stop,
         on_file=lambda s, p: stop.set())
    assert calls == ["a.mp4"]  # the file in flight finishes; nothing new starts


def test_failing_sink_does_not_lose_the_run(tmp_path, data_dir, fake_process, capsys):
    class Broken(sinks.OutputSink):
        name = "broken"

        def write(self, media, record):
            raise OSError("disk full")
    lib = _media(tmp_path / "lib", "a.mp4")
    good = sinks.SidecarSink(mirror_root=tmp_path / "m")
    assert _run([lib], [Broken(), good]) == {"ok": 1}
    assert (lib / "a.mp4.json").exists()
    assert "broken output failed" in capsys.readouterr().out


def test_engine_not_ready_aborts_before_touching_anything(tmp_path, data_dir, fake_process, monkeypatch):
    calls, _ = fake_process
    monkeypatch.setattr(config, "vlm_status", lambda model=None: (False, "engine down"))
    (data_dir / "orphan").mkdir()
    with pytest.raises(batch.BatchAbort):
        _run([_media(tmp_path / "lib", "a.mp4")], [])
    assert calls == [] and (data_dir / "orphan").exists()


def test_sweep_spares_files_and_symlinks(tmp_path, data_dir):
    (data_dir / "stale").mkdir()
    (data_dir / "undecodable.jsonl").write_text("")
    target = tmp_path / "keep"
    target.mkdir()
    (data_dir / "link").symlink_to(target)
    assert batch.sweep_orphan_workdirs() == 1
    assert not (data_dir / "stale").exists()
    assert (data_dir / "undecodable.jsonl").exists() and target.exists()


def test_phase_markers():
    assert batch.phase_for_line("[1/3] Detecting shot cuts in a.mp4 …") == "Detecting shot cuts…"
    assert batch.phase_for_line("[still] analyzing image") == "Analyzing still image…"
    assert batch.phase_for_line("[3/12] /Volumes/x/a.mp4") is None  # file header, not a phase


# -- settings --------------------------------------------------------------------

def test_settings_roundtrip_and_unknown_keys(tmp_path):
    s = state.load_settings(tmp_path)
    assert s == state.DEFAULT_SETTINGS and s is not state.DEFAULT_SETTINGS
    s["sources"]["/x"] = {"selected": True, "hidden": False}
    state.save_settings(tmp_path, s)
    raw = json.loads((tmp_path / "settings.json").read_text())
    raw["sb_host"] = "legacy-key"
    (tmp_path / "settings.json").write_text(json.dumps(raw))
    loaded = state.load_settings(tmp_path)
    assert loaded["sources"] == {"/x": {"selected": True, "hidden": False}}
    assert "sb_host" not in loaded
    assert state.DEFAULT_SETTINGS["sources"] == {}  # defaults never mutated


def test_corrupt_settings_fall_back_with_warning(tmp_path, capsys):
    (tmp_path / "settings.json").write_text("{nope")
    assert state.load_settings(tmp_path) == state.DEFAULT_SETTINGS
    assert "WARNING" in capsys.readouterr().out


# -- network shares ------------------------------------------------------------------

MOUNT_OUTPUT = """\
/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
//alvaro@nas.local/Footage on /Volumes/Footage (smbfs, nodev, nosuid, mounted by user)
//GUEST:@10.0.0.5/Public%20Share on /Volumes/Public Share (smbfs, nodev, nosuid, mounted by user)
afp_0TQ8@server.local/Archive on /Volumes/Archive (afpfs, nodev, nosuid, mounted by user)
nas.local:/export/media on /Volumes/media (nfs, nodev, nosuid, mounted by user)
/dev/disk5s1 on /Volumes/Card (msdos, local, nodev, nosuid, noowners)
"""


def test_parse_mount_output_keeps_only_network_volumes():
    m = volumes.parse_mount_output(MOUNT_OUTPUT)
    assert set(m) == {"/Volumes/Footage", "/Volumes/Public Share", "/Volumes/Archive", "/Volumes/media"}
    assert m["/Volumes/Footage"]["url"] == "smb://alvaro@nas.local/Footage"
    assert m["/Volumes/media"]["url"] == "nfs://nas.local/export/media"


def test_share_for_path_picks_the_right_volume():
    m = volumes.parse_mount_output(MOUNT_OUTPUT)
    assert volumes.share_for_path("/Volumes/Footage/2026/a.mov", m) == (
        "/Volumes/Footage", "smb://alvaro@nas.local/Footage")
    assert volumes.share_for_path("/Volumes/Footage2/x", m) is None  # prefix, not a parent
    assert volumes.share_for_path("/Volumes/Card/DCIM", m) is None   # local disk


def test_unmount_refuses_non_network_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(volumes, "network_mounts", lambda: {"/Volumes/Footage": {"url": "smb://x/y"}})
    monkeypatch.setattr(volumes.subprocess, "run", lambda *a, **k: calls.append(a) or None)
    assert volumes.unmount("/Volumes/Card") is False
    assert volumes.unmount("/") is False
    assert calls == []


def test_connect_dialog_only_accepts_server_urls(monkeypatch):
    calls = []
    monkeypatch.setattr(volumes.subprocess, "run", lambda args, **k: calls.append(args))
    assert volumes.connect_dialog("smb://nas.local") is True
    for bad in ("file:///etc", "https://example.com", "smb://", "nas.local", "smb://a b"):
        assert volumes.connect_dialog(bad) is False
    assert calls == [["open", "smb://nas.local"]]


# -- Backend (no window) -------------------------------------------------------------

@pytest.fixture
def backend(tmp_path, monkeypatch, data_dir):
    pytest.importorskip("webview")
    from vmt_app import app
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path / "support")
    monkeypatch.setattr(app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(volumes, "network_mounts", lambda: {})
    b = app.Backend()
    b.output_sinks = [sinks.SidecarSink(mirror_root=tmp_path / "m")]
    b._spawn_progress_refresh = lambda path: None
    yield b


def test_push_queue_flushes_on_init(backend):
    sent = []

    class FakeWindow:
        def evaluate_js(self, js):
            sent.append(js)
    backend.attach(FakeWindow())
    backend.push(type="log", t="00:00:00", text="early")
    assert sent == []                       # queued until the page is ready
    snap = backend.snapshot()
    assert len(sent) == 1 and '"early"' in sent[0]
    assert snap["tasks"] == {"tags": True, "description": True}
    assert snap["languages"][0] == {"code": "en", "name": "English"}
    backend.push(type="log", t="00:00:01", text="live")
    assert len(sent) == 2


def test_add_select_hide_remove_sources(backend, tmp_path):
    lib = _media(tmp_path / "lib", "a.mp4")
    clip = lib / "a.mp4"
    backend.add_sources([str(lib), str(clip), str(lib)])
    assert set(backend.settings["sources"]) == {str(lib.resolve()), str(clip.resolve())}
    rows = {r["path"]: r for r in backend._sources_payload()["sources"]}
    assert rows[str(lib.resolve())]["kind"] == "folder"
    assert rows[str(clip.resolve())]["kind"] == "file"

    backend.set_source_hidden(str(clip.resolve()), True)
    backend.select_all_sources(True)
    assert backend.settings["sources"][str(clip.resolve())]["selected"] is False  # hidden stays out
    backend.remove_source(str(lib.resolve()))
    assert set(backend.settings["sources"]) == {str(clip.resolve())}
    saved = json.loads((tmp_path / "support" / "settings.json").read_text())
    assert set(saved["sources"]) == {str(clip.resolve())}


def test_missing_source_row(backend, tmp_path):
    backend.settings["sources"]["/Volumes/Gone/footage"] = {"selected": True, "hidden": False}
    backend.settings["shares"]["/Volumes/Gone"] = "smb://nas.local/Gone"
    row = backend._source_row("/Volumes/Gone/footage")
    assert row["kind"] == "missing" and row["share"] == "smb://nas.local/Gone"


def test_pause_window(backend):
    from datetime import datetime
    backend.settings.update(auto_pause_enabled=True, auto_pause_start="22:00", auto_pause_end="06:00")
    assert backend._in_pause_window(datetime(2026, 1, 1, 23, 0))
    assert backend._in_pause_window(datetime(2026, 1, 1, 5, 59))
    assert not backend._in_pause_window(datetime(2026, 1, 1, 12, 0))
    backend.settings.update(auto_pause_start="10:00", auto_pause_end="10:00")
    assert not backend._in_pause_window(datetime(2026, 1, 1, 10, 0))  # zero-length
    assert backend._valid_hhmm("25:00", "09:00") == "09:00"
    assert backend._valid_hhmm("7:5", "09:00") == "07:05"


def test_prune_batch_logs(tmp_path):
    pytest.importorskip("webview")
    import os
    import time

    from vmt_app.app import Backend
    old = tmp_path / "batch_20200101_000000.log"
    new = tmp_path / "batch_20990101_000000.log"
    old.write_text("x")
    new.write_text("y")
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    Backend._prune_batch_logs(tmp_path, retention_days=30)
    assert not old.exists() and new.exists()


def test_reveal_path_only_known_paths(backend, tmp_path, monkeypatch):
    opened = []
    from vmt_app import app
    monkeypatch.setattr(app.subprocess, "Popen", lambda args, **k: opened.append(args))
    lib = _media(tmp_path / "lib", "a.mp4")
    backend.reveal_path(str(lib))
    assert opened == []
    backend.add_sources([str(lib)])
    backend.reveal_path(str(lib.resolve()))
    assert opened == [["open", "-R", str(lib.resolve())]]


def test_concurrent_settings_saves_never_corrupt(tmp_path):
    import json as _json
    settings = state.load_settings(tmp_path)
    settings["sources"] = {f"/v/{i}": {"selected": True, "hidden": False} for i in range(500)}
    errors = []

    def writer():
        for _ in range(40):
            state.save_settings(tmp_path, settings)

    def reader():
        for _ in range(200):
            try:
                _json.loads((tmp_path / "settings.json").read_text())
            except FileNotFoundError:
                pass
            except ValueError as e:
                errors.append(e)
    threads = [threading.Thread(target=writer) for _ in range(4)] + [threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(state.load_settings(tmp_path)["sources"]) == 500
    assert not list(tmp_path.glob(".settings.*.tmp"))


def test_shutdown_interrupts_a_startup_wait(monkeypatch):
    import time as _time

    import llama_manager
    monkeypatch.setattr(llama_manager, "server_alive", lambda timeout=1.0: False)
    monkeypatch.setattr(llama_manager, "_proc", None)
    monkeypatch.setattr(llama_manager, "_adopted_pid", None)
    llama_manager._stopping.clear()
    result = {}
    t = threading.Thread(target=lambda: result.update(
        r=llama_manager._wait_ready(_time.monotonic() + 600)))
    t.start()
    _time.sleep(0.3)
    started = _time.monotonic()
    llama_manager.shutdown(grace_seconds=0.1)
    t.join(timeout=5)
    assert not t.is_alive() and _time.monotonic() - started < 3
    assert result["r"] == (False, "shutting down")
    llama_manager._stopping.clear()


def test_stale_progress_counts_are_dropped(backend, tmp_path, monkeypatch):
    import asyncio

    import library
    lib = _media(tmp_path / "lib", "a.mp4")
    real_scan = library.scan

    def scan_while_tasks_change(*a, **k):
        backend._progress_gen += 1   # the user toggled a task mid-scan
        return real_scan(*a, **k)
    monkeypatch.setattr(library, "scan", scan_while_tasks_change)
    backend.settings["sources"][str(lib)] = {"selected": True, "hidden": False}
    loop = asyncio.new_event_loop()
    backend.loop = loop
    backend._spawn_progress_refresh = lambda path: None
    pushed = []
    backend.push = lambda **ev: pushed.append(ev)

    async def go():
        backend._progress_inflight.add(str(lib))
        await backend._refresh_source_progress(str(lib))
    loop.run_until_complete(go())
    loop.close()
    assert str(lib) not in backend._source_progress
    assert not [e for e in pushed if e.get("type") == "source_progress"]


def test_add_sources_rejects_non_media_files(backend, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    pushed = []
    backend.push = lambda **ev: pushed.append(ev)
    backend.add_sources([str(notes)])
    assert backend.settings["sources"] == {}
    assert any("not a supported media file" in e.get("message", "") for e in pushed)


def test_reconnect_refuses_non_share_urls(monkeypatch):
    calls = []
    monkeypatch.setattr(volumes.subprocess, "run", lambda args, **k: calls.append(args))
    monkeypatch.setattr(volumes, "network_mounts", lambda: {})
    assert volumes.reconnect("https://dav.example.com/x", "/Volumes/x", timeout=0) is False
    assert calls == []


def test_run_analyze_end_to_end_through_the_backend(backend, tmp_path, fake_process, monkeypatch):
    """The Analyze button's path: engine check, run, sink close, the
    notification rule, and _batch_running reset."""
    import asyncio

    from vmt_app import app
    calls, statuses = fake_process
    monkeypatch.setattr(backend, "_engine_check", lambda: (True, "ok"))
    notes = []
    monkeypatch.setattr(app, "_notify", lambda title, msg, subtitle=None: notes.append(msg))
    pushed = []
    backend.push = lambda **ev: pushed.append(ev)
    closed = []

    class Recorder(sinks.OutputSink):
        name = "recorder"

        def write(self, media, record):
            pass

        def close(self):
            closed.append(1)
    backend.output_sinks = [sinks.SidecarSink(mirror_root=tmp_path / "m"), Recorder()]
    lib = _media(tmp_path / "lib", "a.mp4", "b.braw")
    statuses["b.braw"] = "undecodable"
    backend.settings["sources"] = {str(lib): {"selected": True, "hidden": False}}

    loop = asyncio.new_event_loop()
    loop.run_until_complete(backend._run_analyze(scheduled=False))
    assert calls == ["a.mp4", "b.braw"] and closed == [1]
    assert backend._batch_running is False
    assert notes and notes[-1].startswith("Done:")
    assert [e for e in pushed if e.get("type") == "run_status"][-1]["cls"] == "ok"

    # Scheduled re-run: b.braw is held back, nothing new, no notification.
    calls.clear()
    notes.clear()
    loop.run_until_complete(backend._run_analyze(scheduled=True))
    loop.close()
    assert calls == [] and notes == []


def test_run_analyze_engine_down(backend, monkeypatch):
    import asyncio
    monkeypatch.setattr(backend, "_engine_check", lambda: (False, "llama-server exited"))
    pushed = []
    backend.push = lambda **ev: pushed.append(ev)
    backend.settings["sources"] = {"/x": {"selected": True, "hidden": False}}
    asyncio.new_event_loop().run_until_complete(backend._run_analyze(scheduled=False))
    assert backend._batch_running is False
    assert any(e.get("type") == "run_status" and e.get("cls") == "err" for e in pushed)


# -- DaVinci Resolve light ------------------------------------------------------------

def test_resolve_light_follows_start_crash_and_stop(backend, monkeypatch):
    import asyncio

    from vmt_app import app
    events = []
    monkeypatch.setattr(backend, "push", lambda **ev: events.append(ev))
    monkeypatch.setattr(config, "RESOLVE_WATCH_SECONDS", 0)
    monkeypatch.setattr(app.resolve_proxy, "ensure_resolve_running", lambda: (object(), False))
    alive = {"v": True}
    monkeypatch.setattr(app.resolve_proxy, "resolve_running", lambda: alive["v"])
    monkeypatch.setattr(app.tagger, "shutdown_resolve", lambda: None)
    assert backend.snapshot()["resolve_light"] == "off"

    async def scenario():
        await backend._start_resolve()
        assert backend._resolve_light == "ok"
        await asyncio.sleep(0.01)                      # watcher sees Resolve alive
        assert backend._resolve_handle is not None
        alive["v"] = False                             # Resolve crashed / quit by hand
        await asyncio.wait_for(backend._resolve_watch_task, 1)
        assert backend._resolve_handle is None and backend._resolve_light == "error"
        alive["v"] = True                              # toggle off and on: fresh start
        await backend._stop_resolve()
        assert backend._resolve_light == "off"
        await backend._start_resolve()
        assert backend._resolve_light == "ok"
        await backend._stop_resolve()
        await asyncio.wait_for(backend._resolve_watch_task, 1)  # exits once stopped
    asyncio.new_event_loop().run_until_complete(scenario())
    lights = [ev["light"] for ev in events if ev.get("type") == "resolve"]
    assert lights == ["busy", "ok", "error", "off", "busy", "ok", "off"]


def test_second_instance_is_refused_until_the_first_exits(tmp_path, monkeypatch):
    pytest.importorskip("webview")
    import subprocess
    import sys

    from vmt_app import app
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(app, "_instance_lock_fd", None)
    holder = subprocess.Popen([sys.executable, "-c",
                               "import fcntl, os, time; "
                               f"fd = os.open({str(tmp_path / 'app.lock')!r}, os.O_RDWR | os.O_CREAT); "
                               "fcntl.flock(fd, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(600)"],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert app._acquire_single_instance() is False
        holder.kill()                                   # even SIGKILL frees it
        holder.wait()
        assert app._acquire_single_instance() is True
        assert (tmp_path / "app.lock").read_text() == str(__import__("os").getpid())
    finally:
        if holder.poll() is None:
            holder.kill()
        if app._instance_lock_fd is not None:
            __import__("os").close(app._instance_lock_fd)
