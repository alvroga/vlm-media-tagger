"""The llama-server watchdog: the engine (~23 GB) must not outlive the
process that started it, even when that process is SIGKILLed."""
import subprocess
import time

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
