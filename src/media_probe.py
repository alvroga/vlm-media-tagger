"""
Quick "can this machine decode this file?" probe.

RAW camera formats (BRAW, R3D, ARRIRAW, X-OCN …) can't be decoded by
PyAV/ffmpeg — those files need the DaVinci Resolve Studio proxy path.

Three hard-won rules baked in here:
  - The probe must DECODE one frame, not just open the container: an ARRIRAW
    .mxf opens fine (MXF is a standard container) but its essence can't be
    decoded.
  - The decode runs in a SUBPROCESS: ARRIRAW .mxf doesn't make libav raise —
    it makes it segfault (measured: exit 139), which would kill the whole
    process mid-batch. A crashed child is just "undecodable".
  - The child must not LOOK crashed to macOS (2026-07-14): a subprocess dying
    of SIGSEGV triggers ReportCrash — a "Python quit unexpectedly" dialog
plus an .ips report in DiagnosticReports, every
    single time an ARRIRAW/X-OCN .mxf is probed, even though the pipeline
    handles it fine. _trap_fatal_signals() installs libc's _exit as the
    C-level handler for the fatal signals, so the child exits cleanly with
    code == signum instead of terminating abnormally — no abnormal exit, no
    crash reporter. It must be a pure-C handler (signal(sig, _exit) — the
    kernel calls handler(signum), and _exit's signature matches): a Python
    `signal.signal` handler only runs later on the interpreter loop, which
    never resumes after a segfault in C code.

Path is passed via an env var, not argv: in a packaged app the child process
can be a launcher stub that ignores argv. Harmless for the plain CLI.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

PROBE_TIMEOUT = 60  # seconds — decoding ONE frame should be near-instant
PROBE_ENV_VAR = "VMT_PROBE_PATH"

# Signals a broken decoder can die of. The child traps these (see module
# docstring) and exits with code == signum; the parent maps those exit codes
# back to "decoder crashed".
_FATAL_SIGNALS = (signal.SIGILL, signal.SIGABRT, signal.SIGFPE,
                  signal.SIGBUS, signal.SIGSEGV)


class UndecodableMediaError(Exception):
    """The file can't be decoded by PyAV — needs a Resolve proxy."""


def probe_video(path: str | Path) -> None:
    """Raise UndecodableMediaError unless one video frame decodes cleanly."""
    name = Path(path).name
    env = {**os.environ, PROBE_ENV_VAR: str(path)}
    try:
        proc = subprocess.run(
            [sys.executable, __file__],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        raise UndecodableMediaError(
            f"{name}: decode probe timed out after {PROBE_TIMEOUT}s") from None
    if proc.returncode == 0:
        return
    # Trapped fatal signal → clean exit with code == signum (see module
    # docstring); < 0 kept as a fallback for an untrapped/real signal kill.
    if proc.returncode in _FATAL_SIGNALS or proc.returncode < 0:
        raise UndecodableMediaError(
            f"{name}: decoder crashed (signal {abs(proc.returncode)})")
    detail = (proc.stderr or "").strip().splitlines()
    raise UndecodableMediaError(
        f"{name}: {detail[-1] if detail else f'probe exited {proc.returncode}'}")


def _decode_one_frame(path: str) -> str | None:
    """Child-side: return None if a frame decodes, else an error string."""
    import av
    try:
        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return "no video stream"
            for _ in container.decode(stream):
                return None  # one decoded frame is proof enough
        return "video stream has no decodable frames"
    except Exception as e:  # av.error.* — InvalidDataError, DecoderNotFound, …
        return str(e)


def _trap_fatal_signals() -> None:
    """Child-side: exit(signum) on fatal signals instead of crashing, so a
    segfaulting decoder doesn't summon the macOS crash-reporter dialog."""
    import ctypes
    libc = ctypes.CDLL(None)
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.signal.restype = ctypes.c_void_p
    exit_addr = ctypes.cast(libc._exit, ctypes.c_void_p).value
    for sig in _FATAL_SIGNALS:
        libc.signal(int(sig), exit_addr)


def _run_as_child() -> None:
    """Entry point when invoked as the probe subprocess (see probe_video())."""
    try:
        _trap_fatal_signals()
    except Exception:
        pass  # worst case: old behavior (crash dialog), probe still works
    path = os.environ[PROBE_ENV_VAR]
    error = _decode_one_frame(path)
    if error:
        print(error, file=sys.stderr)
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    _run_as_child()
