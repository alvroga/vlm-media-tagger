"""The llama-server watchdog: the engine (~23 GB) must not outlive the
process that started it, even when that process is SIGKILLed."""
import os
import subprocess
import time

import config
import llama_manager


def _sleeper():
    return subprocess.Popen(["sleep", "600"])


def test_watchdog_terminates_server_when_owner_is_killed():
    owner, server = _sleeper(), _sleeper()
    dog = llama_manager.spawn_watchdog(owner.pid, server.pid, poll=0.2, grace=2)
    try:
        time.sleep(0.6)
        assert server.poll() is None          # owner alive: server left alone
        owner.kill()
        owner.wait()
        assert server.wait(timeout=5) != 0    # SIGTERMed by the watchdog
        assert dog.wait(timeout=5) == 0
    finally:
        for p in (owner, server):
            if p.poll() is None:
                p.kill()


def test_watchdog_exits_when_server_stops_first():
    owner, server = _sleeper(), _sleeper()
    dog = llama_manager.spawn_watchdog(owner.pid, server.pid, poll=0.2, grace=2)
    try:
        server.terminate()
        server.wait()
        assert dog.wait(timeout=5) == 0       # never lingers
        assert owner.poll() is None           # and never touches the owner
    finally:
        owner.kill()


def test_server_with_a_live_owner_is_never_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "APP_SUPPORT_DIR", tmp_path)
    server = subprocess.Popen(["/bin/sh", "-c", 'exec -a llama-server sleep 600'])
    owner = _sleeper()
    try:
        time.sleep(0.2)
        (tmp_path / "llama-server.pid").write_text(f"{server.pid} {owner.pid}")
        assert llama_manager._adopt_orphan() is None       # owner alive: hands off
        owner.kill()
        owner.wait()
        assert llama_manager._adopt_orphan() == server.pid  # owner dead: an orphan
        (tmp_path / "llama-server.pid").write_text(f"{server.pid} {os.getpid()}")
        assert llama_manager._adopt_orphan() == server.pid  # our own
    finally:
        for p in (owner, server):
            if p.poll() is None:
                p.kill()


def test_final_shutdown_refuses_a_respawn(monkeypatch):
    monkeypatch.setattr(config, "LLAMA_MANAGED", True)
    monkeypatch.setattr(config, "VLM_BACKEND", "llama")
    monkeypatch.setattr(llama_manager, "_spawn_locked",
                        lambda: (_ for _ in ()).throw(AssertionError("respawned")))
    try:
        llama_manager.shutdown(final=True)
        assert llama_manager.ensure_started(wait=False) == (False, "shutting down")
    finally:
        llama_manager._closed.clear()


def test_short_download_keeps_the_part_file(tmp_path, monkeypatch):
    import io

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(llama_manager.urllib.request, "urlopen", lambda *a, **k: Resp(b"x" * 10))
    dest = tmp_path / "m.gguf"
    import pytest
    with pytest.raises(OSError, match="incomplete download"):
        llama_manager._download("http://x", dest, 100, lambda n: None)
    assert not dest.exists() and dest.with_suffix(".part").stat().st_size == 10
