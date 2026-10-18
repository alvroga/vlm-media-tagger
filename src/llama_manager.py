"""
Managed llama-server — the inference engine for MiniCPM-V 4.7 35B-A3B.

Ollama can't run 4.7 yet (its bundled llama.cpp predates the minicpmv4_7
vision projector), so this manages llama.cpp's own server:

  - the vendored static binary (scripts/build_llama_cpp.sh →
    vendor/llama_cpp/llama-server) runs as a private child on
    config.LLAMA_PORT, bound to 127.0.0.1;
  - the two model files (Q4_K_M weights, 21.2 GB, + the f16 vision
    projector, 1.1 GB) live in config.LLAMA_MODELS_DIR. They come from, in
    order: already there → the local Ollama store, if this exact build was
    pulled through Ollama (same bytes, matched by sha256; hard-linked, so no
    extra disk) → a resumable Hugging Face download, sha256-verified;
  - the loaded model holds ~23 GB RSS; it must not outlive the process
    that started it. A clean quit terminates it (shutdown()); a watchdog
    (/bin/sh, own session) terminates it within seconds when that process
    dies without cleanup (Force Quit, crash, SIGKILL); and an orphan that
    still slips through is adopted via a pidfile and terminated on shutdown.

When LLAMA_HOST is set (config.LLAMA_MANAGED False) every function here is a
no-op and that external server is used as-is.

    python src/llama_manager.py status | fetch | start | stop
"""
import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

HF_REPO = "openbmb/MiniCPM-V-4.7-35B-A3B-gguf"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/"
# (role, filename, bytes, sha256) — pinned: a different file is a different
# model and needs a harness run before use.
MODEL_FILES = (
    ("model", "MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf", 21166255008,
     "bf3d1c67f7843193e4483aebd26c5dea6b6d1f6a06ffa03c2387a6f3cc6d7df5"),
    ("mmproj", "mmproj-model-f16.gguf", 1119575840,
     "bff3b688e14c300eca273092c37c50a0ddfcc2054e079af1a0480edeed5d20f1"),
)
OLLAMA_BLOBS = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models") / "blobs"
# Loading 22 GB of weights into Metal takes a while on a cold disk cache.
STARTUP_TIMEOUT = 600
# Watchdog: how often it checks that the owning process is alive, and how
# long llama-server gets after SIGTERM before SIGKILL.
WATCHDOG_POLL_SECONDS = 2
WATCHDOG_GRACE_SECONDS = 10
# Exits on its own as soon as the server is gone; never outlives it.
_WATCHDOG_SH = (
    'p=$1; c=$2; poll=$3; grace=$4\n'
    'while kill -0 "$p" 2>/dev/null && kill -0 "$c" 2>/dev/null; do sleep "$poll"; done\n'
    'kill -0 "$c" 2>/dev/null || exit 0\n'
    'kill -TERM "$c" 2>/dev/null\n'
    'i=0; while [ "$i" -lt "$grace" ] && kill -0 "$c" 2>/dev/null; do sleep 1; i=$((i+1)); done\n'
    'kill -KILL "$c" 2>/dev/null; exit 0\n'
)

_lock = threading.Lock()
# Set by shutdown(): a startup wait in progress gives up at once instead of
# blocking quit for up to STARTUP_TIMEOUT.
_stopping = threading.Event()
# Set by shutdown(final=True) at app quit: nothing may respawn the engine
# afterwards (a click or a timer while the window lingers during cleanup).
_closed = threading.Event()
_proc: "subprocess.Popen | None" = None
_adopted_pid: "int | None" = None


def _pidfile() -> Path:
    return config.APP_SUPPORT_DIR / "llama-server.pid"


def log_path() -> Path:
    return config.APP_SUPPORT_DIR / "logs" / "llama-server.log"


def model_path(role: str) -> Path:
    name = next(f for r, f, _, _ in MODEL_FILES if r == role)
    return config.LLAMA_MODELS_DIR / name


def find_binary() -> "Path | None":
    """Vendored binary: checkout = vendor/llama_cpp/, frozen app =
    Resources/app/llama_cpp/ (briefcase lands vendor dirs flat under app/).
    LLAMA_SERVER_BIN overrides; PATH last."""
    override = os.environ.get("LLAMA_SERVER_BIN")
    candidates = ([Path(override)] if override else []) + [
        config.PROJECT_ROOT / "vendor" / "llama_cpp" / "llama-server",
        config.PROJECT_ROOT / "app" / "llama_cpp" / "llama-server",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    import shutil
    which = shutil.which("llama-server")
    return Path(which) if which else None


# -- model files ----------------------------------------------------------------

def _complete(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def models_installed() -> bool:
    return all(_complete(model_path(r), size) for r, _, size, _ in MODEL_FILES)


def missing_bytes() -> int:
    """Bytes still to fetch (0 once installed). A file the Ollama store can
    supply costs nothing."""
    total = 0
    for role, _, size, sha in MODEL_FILES:
        if _complete(model_path(role), size) or _complete(OLLAMA_BLOBS / f"sha256-{sha}", size):
            continue
        part = model_path(role).with_suffix(".part")
        total += size - (part.stat().st_size if part.exists() else 0)
    return total


def _sha256(path: Path, on_bytes=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
            if on_bytes:
                on_bytes(len(chunk))
    return h.hexdigest()


def _link_from_ollama(dest: Path, size: int, sha: str) -> bool:
    """Reuse an identical blob from the local Ollama store. Ollama names
    blobs by sha256, so a size match on the expected name is the same file;
    hard link (no extra disk, survives `ollama rm`), symlink across volumes."""
    blob = OLLAMA_BLOBS / f"sha256-{sha}"
    if not _complete(blob, size):
        return False
    try:
        os.link(blob, dest)
    except OSError:
        try:
            dest.symlink_to(blob)
        except OSError:
            return False
    return True


def _download(url: str, dest: Path, size: int, on_bytes) -> None:
    """Resumable download to dest.part (HTTP Range), renamed when complete."""
    part = dest.with_suffix(".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have:
        on_bytes(have)
    if have < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
            if have and r.status != 206:  # server ignored Range: start over
                f.truncate(0)
                on_bytes(-have)
            while chunk := r.read(1 << 20):
                f.write(chunk)
                on_bytes(len(chunk))
    # read() returns b"" on a dropped connection instead of raising: keep the
    # .part (the retry resumes) rather than hash and delete a short file.
    got = part.stat().st_size
    if got != size:
        raise OSError(f"incomplete download ({got} of {size} bytes)")
    part.replace(dest)


def fetch_models(progress_cb=None) -> tuple[bool, str]:
    """Make both model files present and verified. progress_cb(done, total,
    phase) is called from this thread. Never raises; (ok, message)."""
    config.LLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    total = missing_bytes()
    done = [0]

    def on_bytes(n):
        done[0] += n
        if progress_cb:
            progress_cb(max(0, done[0]), max(total, 1), phase[0])

    phase = ["preparing…"]
    try:
        for role, name, size, sha in MODEL_FILES:
            dest = model_path(role)
            if _complete(dest, size):
                continue
            if dest.is_symlink() or dest.exists():
                dest.unlink()  # wrong size: stale or partial
            if _link_from_ollama(dest, size, sha):
                print(f"  reused {name} from the local Ollama store")
                continue
            phase[0] = f"downloading {'vision model' if role == 'model' else 'vision projector'}…"
            _download(HF_BASE + name, dest, size, on_bytes)
            phase[0] = "verifying download…"
            if progress_cb:
                progress_cb(done[0], max(total, 1), phase[0])
            if _sha256(dest) != sha:
                dest.unlink(missing_ok=True)
                return False, f"{name} failed its checksum — deleted, retry to download again"
    except (urllib.error.URLError, OSError) as e:
        return False, f"download failed: {e} (retry resumes where it stopped)"
    if progress_cb:
        progress_cb(max(total, 1), max(total, 1), "done")
    return True, "model files ready"


# -- server lifecycle -------------------------------------------------------------

def server_alive(timeout: float = 2.0) -> bool:
    """Ready to serve (200 on /health; 503 while loading counts as not yet)."""
    try:
        with urllib.request.urlopen(config.LLAMA_HOST.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _port_open() -> bool:
    """Anything listening (including a server still loading)."""
    try:
        urllib.request.urlopen(config.LLAMA_HOST.rstrip("/") + "/health", timeout=2)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _adopt_orphan() -> "int | None":
    """PID recorded by a previous session, verified with ps so a recycled PID
    of an unrelated process is never adopted (we'd SIGTERM it). The pidfile
    also names the owner ("<server> <owner>"): a server whose owner is still
    alive (another app instance, a `llama_manager.py start` in a terminal) is
    used as-is but never adopted, so our shutdown can't pull it from under
    its owner."""
    try:
        fields = _pidfile().read_text().split()
        pid = int(fields[0])
        owner = int(fields[1]) if len(fields) > 1 else None
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None
    if "llama-server" not in out:
        return None
    if owner is not None and owner != os.getpid() and _pid_alive(owner):
        return None
    return pid


def _wait_ready(deadline: float) -> tuple[bool, str]:
    """Called WITHOUT _lock held, so shutdown() can run while a cold model
    load (minutes) is being waited on."""
    while time.monotonic() < deadline:
        if _stopping.is_set():
            return False, "shutting down"
        proc = _proc
        if proc is not None and proc.poll() is not None:
            return False, f"llama-server exited (code {proc.returncode}) — see {log_path()}"
        if server_alive(timeout=1.0):
            return True, f"ready at {config.LLAMA_HOST}"
        time.sleep(0.5)
    return False, f"llama-server did not become ready in {STARTUP_TIMEOUT}s — see {log_path()}"


def ensure_started(wait: bool = True) -> tuple[bool, str]:
    """Managed mode: make sure the private server is up (spawning it if
    needed). Cheap when already running. Never raises; (ok, message)."""
    global _adopted_pid
    if not config.LLAMA_MANAGED or config.VLM_BACKEND != "llama":
        return True, "unmanaged — using the configured server"
    if _closed.is_set():
        return False, "shutting down"
    _stopping.clear()
    with _lock:
        if _proc is not None and _proc.poll() is None:
            spawned = False  # ours, still loading (port not open yet): never a second one
        elif _port_open():
            if _proc is None and _adopted_pid is None:
                _adopted_pid = _adopt_orphan()
                if _adopted_pid is not None:
                    # Adopted orphans get the same guarantees as our own.
                    with contextlib.suppress(OSError):
                        _pidfile().write_text(f"{_adopted_pid} {os.getpid()}")
                    spawn_watchdog(os.getpid(), _adopted_pid)
            spawned = False
        else:
            ok, msg = _spawn_locked()
            if not ok:
                return False, msg
            spawned = True
    if not wait:
        return True, "starting" if spawned else "already running"
    return _wait_ready(time.monotonic() + STARTUP_TIMEOUT)


def _spawn_locked() -> tuple[bool, str]:
    """Start the server process. Caller holds _lock."""
    global _proc, _adopted_pid
    if _proc is not None and _proc.poll() is not None:
        _proc = None
    _adopted_pid = None

    binary = find_binary()
    if binary is None:
        return False, ("llama-server not found — run scripts/build_llama_cpp.sh "
                       "(or set LLAMA_SERVER_BIN)")
    if not models_installed():
        return False, "model files missing — run the first-time download"
    log_path().parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary),
           "-m", str(model_path("model")), "--mmproj", str(model_path("mmproj")),
           "--host", "127.0.0.1", "--port", str(config.LLAMA_PORT),
           "-ngl", "99", "-fa", "on", "--jinja",
           "--parallel", str(config.PARALLEL_WORKERS),
           "-c", str(config.LLAMA_CTX * config.PARALLEL_WORKERS),
           "--alias", config.LLAMA_MODEL_NAME]
    try:
        with open(log_path(), "w") as log_file:
            _proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                     stdout=log_file, stderr=subprocess.STDOUT)
    except OSError as e:
        return False, f"failed to launch llama-server: {e}"
    try:
        _pidfile().write_text(f"{_proc.pid} {os.getpid()}")
    except OSError:
        pass
    spawn_watchdog(os.getpid(), _proc.pid)
    return True, "spawned"


def spawn_watchdog(owner_pid: int, server_pid: int, poll: float = WATCHDOG_POLL_SECONDS,
                   grace: int = WATCHDOG_GRACE_SECONDS) -> "subprocess.Popen | None":
    """Terminate server_pid once owner_pid is gone. In its own session, so a
    signal aimed at the owner (or its process group) doesn't take it down
    too; macOS has no PR_SET_PDEATHSIG."""
    try:
        return subprocess.Popen(
            ["/bin/sh", "-c", _WATCHDOG_SH, "vmt-llama-watchdog",
             str(owner_pid), str(server_pid), str(poll), str(int(grace))],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    except OSError as e:
        print(f"WARNING: llama-server watchdog not started ({e})", file=sys.stderr)
        return None


def shutdown(grace_seconds: float = 10.0, final: bool = False) -> None:
    """Terminate the managed server (ours or adopted). Safe to call always.
    final=True (app quit) also refuses any later ensure_started()."""
    global _proc, _adopted_pid
    if final:
        _closed.set()
    _stopping.set()
    with _lock:
        if _proc is not None:
            _proc.terminate()
            try:
                _proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                _proc.kill()
            _proc = None
        elif _adopted_pid is not None:
            try:
                os.kill(_adopted_pid, signal.SIGTERM)
                deadline = time.monotonic() + grace_seconds
                while time.monotonic() < deadline:
                    os.kill(_adopted_pid, 0)  # raises once it's gone
                    time.sleep(0.25)
                os.kill(_adopted_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _adopted_pid = None
        try:
            _pidfile().unlink(missing_ok=True)
        except OSError:
            pass


def _main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(f"binary:  {find_binary()}")
        for role, name, size, _ in MODEL_FILES:
            p = model_path(role)
            print(f"{role:7}  {p}  {'OK' if _complete(p, size) else 'missing'}")
        print(f"to fetch: {missing_bytes() / 1e9:.1f} GB")
        print(f"server:  {config.LLAMA_HOST} {'ready' if server_alive() else 'not running'}")
    elif cmd == "fetch":
        last = [0.0]

        def cb(done, total, phase):
            if time.monotonic() - last[0] > 2:
                last[0] = time.monotonic()
                print(f"  {phase} {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
        print(fetch_models(cb))
    elif cmd == "start":
        ok, msg = ensure_started()
        print(msg)
        if ok:
            print("running in the foreground — Ctrl-C to stop")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                shutdown()
    elif cmd == "stop":
        _adopted = _adopt_orphan()
        if _adopted:
            os.kill(_adopted, signal.SIGTERM)
            print(f"sent SIGTERM to {_adopted}")
        else:
            print("no managed llama-server running")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    _main()
