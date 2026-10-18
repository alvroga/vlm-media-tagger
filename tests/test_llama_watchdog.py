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
