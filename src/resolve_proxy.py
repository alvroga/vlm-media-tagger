"""
resolve_proxy.py — Headless DaVinci Resolve Studio proxy generator

Standalone module (from the resolve-proxy-noGUI project). Given an input media path
(including RAW formats Resolve natively decodes — BRAW, RED R3D, Sony X-OCN,
ARRI, etc.), produces a proxy next to it (or at a given output path).
Defaults to H.264 .mp4; pass codec="prores422hq" (etc.) for a ProRes .mov.

Requires: DaVinci Resolve Studio installed locally, with scripting enabled.

Usage as a library (single file):
    from resolve_proxy import transcode
    transcode("/path/to/clip.braw")

Usage as a library (queue of files — launches/quits Resolve only once):
    from resolve_proxy import transcode_batch
    transcode_batch([
        "/path/to/clip1.r3d",
        "/path/to/clip2.mxf",
        "/path/to/clip3.mxf",
    ])

Usage from the command line (one or more files):
    python3 resolve_proxy.py /path/to/clip.braw
    python3 resolve_proxy.py /path/to/clip.braw -o /path/to/output.mp4
    python3 resolve_proxy.py /path/to/clip1.r3d /path/to/clip2.mxf --output-dir /proxies
    python3 resolve_proxy.py --help
"""

import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# R3D chunk filtering
# ---------------------------------------------------------------------------
# R3D ships as numbered chunks per clip (..._001.R3D, ..._002.R3D,
# ..._003.R3D, ...) — the first chunk's clip metadata already references
# and plays through the rest as one continuous clip, so handing Resolve a
# later chunk as if it were an independent file is redundant at best
# (Resolve either re-derives the same full clip from a later chunk, or
# errors on a fragment). Any code that builds a Resolve transcode batch must
# filter to chunk 001 per R3D clip first — this is that filter, standalone so
# it can be tested on its own.
_R3D_CHUNK_RE = re.compile(r"^(?P<clip>.+)_(?P<chunk>\d{3})\.r3d$", re.IGNORECASE)


def filter_r3d_chunks(paths):
    """Keep only the first chunk (_001.R3D) per R3D clip; drop
    _002.R3D/_003.R3D/etc. Non-R3D paths (or R3D paths that don't match the
    _NNN chunk-numbering convention at all) pass through unchanged — this
    only ever REMOVES redundant later chunks, never alters anything else.
    Preserves input order."""
    kept = []
    for path in paths:
        m = _R3D_CHUNK_RE.match(os.path.basename(path))
        if not m:
            kept.append(path)
            continue
        if m.group("chunk") == "001":
            kept.append(path)
        else:
            print(f"  skipping {os.path.basename(path)} — not the first chunk "
                  f"of clip '{m.group('clip')}' (already covered by _001.R3D)")
    return kept


# ---------------------------------------------------------------------------
# Resolve — paths and startup
# ---------------------------------------------------------------------------
RESOLVE_SCRIPT_PATH = (
    "/Library/Application Support/Blackmagic Design"
    "/DaVinci Resolve/Developer/Scripting/Modules"
)
RESOLVE_BIN             = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/MacOS/Resolve"
RESOLVE_STARTUP_TIMEOUT = 120  # seconds

# Ceiling on a single clip's render, matching the startup timeout's pattern:
# a wedged Resolve must never spin _render_clip's IsRenderingInProgress()
# poll loop forever (that loop runs INSIDE tagger.py's per-file batch loop —
# a hang there stalls an unattended overnight run indefinitely, with no
# circuit breaker able to notice, since nothing ever fails). Generous on
# purpose: this is real RAW transcoding of a possibly long clip on possibly
# modest hardware, so it must only ever fire on a genuine hang. Reaching it
# fails just that clip (returns None) — tagger.py reports the file as
# undecodable and the batch moves on.
RESOLVE_RENDER_TIMEOUT = 3600  # seconds (1 h per clip)

# Ownership marker — which live process currently manages the headless
# Resolve instance's lifecycle (i.e. is responsible for Quit()ing it).
# Path hardcoded rather than imported from config: this module is
# deliberately standalone (see module docstring), importable with no other
# part of this project present.
_OWNER_MARKER = os.path.join(
    os.path.expanduser("~"), ".cache", "vlm-media-tagger", "resolve_owner.pid")

# ---------------------------------------------------------------------------
# Default proxy settings — override per-call via function args, or per-run
# via CLI flags (--width, --height, --codec, --suffix, --output-dir). Change
# these if you want new defaults everywhere.
# ---------------------------------------------------------------------------
DEFAULT_WIDTH   = 1920
DEFAULT_HEIGHT  = 1080
DEFAULT_CODEC   = "h264"

# Codec choices -> Resolve's SetCurrentRenderFormatAndCodec(format, codec)
# arguments, plus the container extension used to build the default output
# suffix ("_proxy" + extension) when no explicit suffix/output path is given.
CODEC_PRESETS = {
    "h264":           {"format": "mp4", "codec": "H264",           "extension": ".mp4"},
    "prores422proxy": {"format": "mov", "codec": "ProRes422Proxy", "extension": ".mov"},
    "prores422lt":    {"format": "mov", "codec": "ProRes422LT",    "extension": ".mov"},
    "prores422":      {"format": "mov", "codec": "ProRes422",      "extension": ".mov"},
    "prores422hq":    {"format": "mov", "codec": "ProRes422HQ",    "extension": ".mov"},
    "prores4444":     {"format": "mov", "codec": "ProRes4444",     "extension": ".mov"},
    "prores4444xq":   {"format": "mov", "codec": "ProRes4444XQ",   "extension": ".mov"},
}

if RESOLVE_SCRIPT_PATH not in sys.path:
    sys.path.insert(0, RESOLVE_SCRIPT_PATH)

try:
    import DaVinciResolveScript as dvr
except ImportError:
    # Deliberately NOT sys.exit here — tagger.py imports this module on every
    # run, and that import must succeed even when Resolve isn't installed
    # (only a RAW file needs it). ensure_resolve_running() raises
    # RuntimeError lazily instead.
    dvr = None


def _running_resolve_pids() -> list[str]:
    result = subprocess.run(["pgrep", "-x", "Resolve"], capture_output=True, text=True)
    return result.stdout.split() if result.returncode == 0 else []


def resolve_running() -> bool:
    """True while any Resolve process (headless or GUI) is alive."""
    return bool(_running_resolve_pids())


def _wait_for_scripting_api(timeout=RESOLVE_STARTUP_TIMEOUT):
    """Poll dvr.scriptapp("Resolve") until it returns non-None or `timeout`
    elapses. Returns the resolve instance, or None on timeout."""
    print(f"  Waiting for scripting API (up to {timeout}s)...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        resolve = dvr.scriptapp("Resolve")
        if resolve is not None:
            print(" ready.")
            return resolve
        time.sleep(3)
        print(".", end="", flush=True)
    print()
    return None


def _find_gui_resolve_pid() -> str | None:
    """Return the pid of a running Resolve process launched WITHOUT
    -nogui (a real GUI session — someone left it open) — the ONLY way
    this codebase ever launches it is with -nogui, so anything else found
    running must be a human's session, not ours. None if every running
    instance is headless. Checked before this module ever tries to claim
    or use a running Resolve for anything (see ensure_resolve_running):
    a scripted transcode's CreateProject/SetCurrentTimeline calls really
    do switch whatever project is open in a live GUI, so reusing one for
    background work would visibly disrupt an actual editing session."""
    for pid in _running_resolve_pids():
        cmd = subprocess.run(
            ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
        ).stdout
        if "-nogui" not in cmd:
            return pid
    return None


def _kill_headless_resolve() -> None:
    """kill -9 every running Resolve process launched with -nogui — used
    only to clear a confirmed-ours, confirmed-wedged orphan (see
    ensure_resolve_running). Never touches a real GUI session (no -nogui
    in its command line) — belt-and-suspenders: by the time this runs,
    ensure_resolve_running() has already refused to proceed at all if a
    GUI session was found, but this is a kill -9, so it re-checks anyway."""
    for pid in _running_resolve_pids():
        cmd = subprocess.run(
            ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
        ).stdout
        if "-nogui" in cmd:
            subprocess.run(["kill", "-9", pid])


def _read_owner_pid() -> int | None:
    try:
        # `with`, not a bare open(): this is called on every
        # ensure_resolve_running(), and the leaked handle showed up as a real
        # ResourceWarning in a long batch run (2026-07-30).
        with open(_OWNER_MARKER) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _claim_ownership() -> None:
    """Record THIS process as the one responsible for Quit()ing the
    headless Resolve instance — see ensure_resolve_running()."""
    try:
        os.makedirs(os.path.dirname(_OWNER_MARKER), exist_ok=True)
        with open(_OWNER_MARKER, "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass  # best-effort: worst case reverts to the old any-headless-is-ours behavior


def release_ownership() -> None:
    """Call after actually Quit()ing a headless instance this process
    claimed via ensure_resolve_running() (ours_to_quit=True) —
    tagger.shutdown_resolve and transcode_batch's own cleanup both do this. Only clears the marker if it still names THIS
    process, so a race with a newer owner can't erase their claim."""
    if _read_owner_pid() == os.getpid():
        try:
            os.remove(_OWNER_MARKER)
        except OSError:
            pass


def owned_by_this_process() -> bool:
    """This process holds the Quit() responsibility for a headless Resolve."""
    return _read_owner_pid() == os.getpid()


def kill_headless_if_lingering(grace: float = 10.0) -> None:
    """Quit-time fallback for an instance this process owned: give a Quit()
    in flight `grace` seconds, then kill whatever headless Resolve is left
    (never a GUI session; see _kill_headless_resolve)."""
    deadline = time.time() + grace
    while time.time() < deadline:
        if not _running_resolve_pids():
            return
        time.sleep(0.5)
    _kill_headless_resolve()
    release_ownership()


def ensure_resolve_running():
    """
    Connect to Resolve, launching it headless (-nogui) if it isn't already
    running, then wait for the scripting API to come up.

    NOTE: -rr (remote rendering) does NOT work for scripting-driven renders —
    StartRendering queues the job into a "waiting for remote controller" queue
    and no file is ever written. -nogui is the mode that actually renders.

    Refuses outright (raises RuntimeError) if Resolve is found running
    with a real GUI (someone left it open) — see _find_gui_resolve_pid.
    This project never uses a GUI session for scripted transcoding: a proxy
    render's CreateProject/SetCurrentTimeline calls would visibly switch
    whatever project is open, disrupting real work. The caller (tagger.py's
    _get_resolve/_process_raw_via_resolve) already surfaces any exception here as a normal error status/skip
    reason, so this just needs a clear message telling the user to close
    Resolve — no separate plumbing.

    Returns (resolve_instance, ours_to_quit) — ours_to_quit is True when
    it's this call's responsibility to Quit() this instance later: either
    this call launched it, or it was already running headless with no
    OTHER live process currently claiming ownership (see the _OWNER_MARKER
    PID file, checked below) — most likely an orphan left by a previous
    ungraceful exit. False only when a different live process already owns
    it — found 2026-07-24: a standalone verification script's
    ensure_resolve_running() call happily claimed another live process's
    already-running headless Resolve as "ours to quit" (nothing else was
    tracking ownership, so the old logic figured any pre-existing headless
    instance had to be a stale orphan) and quit it out from under that
    still-running process on exit. The marker records which PID actually owns
    the Quit() responsibility so a second, unrelated process reuses the
    connection without claiming it. Callers that DO get ours_to_quit=True
    must call release_ownership() after actually calling Quit()
    (tagger.shutdown_resolve and transcode_batch already do).

    Raises RuntimeError if the DaVinciResolveScript module isn't
    importable (Resolve not installed), a GUI session is open (see above),
    or the scripting API never comes up within RESOLVE_STARTUP_TIMEOUT
    (after one self-heal retry — see below).
    """
    if dvr is None:
        raise RuntimeError(
            f"Could not import DaVinciResolveScript (expected at "
            f"{RESOLVE_SCRIPT_PATH}) — make sure DaVinci Resolve Studio "
            f"is installed with scripting enabled."
        )

    gui_pid = _find_gui_resolve_pid()
    if gui_pid is not None:
        raise RuntimeError(
            "DaVinci Resolve Studio is open — close it so a background instance "
            "can be started."
        )

    already_running = bool(_running_resolve_pids())

    if not already_running:
        print("  Resolve not running — launching headless (-nogui)...")
        subprocess.Popen(
            [RESOLVE_BIN, "-nogui"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        ours_to_quit = True
        _claim_ownership()
    else:
        owner_pid = _read_owner_pid()
        other_live_owner = (owner_pid is not None and owner_pid != os.getpid()
                            and _pid_alive(owner_pid))
        if other_live_owner:
            ours_to_quit = False
            print(f"  Resolve already running headless, owned by another "
                  f"process (pid {owner_pid}) — reusing it.")
        else:
            ours_to_quit = True
            _claim_ownership()
            print("  Resolve already running headless — claiming it.")

    resolve = _wait_for_scripting_api()
    if resolve is not None:
        return resolve, ours_to_quit

    # Self-heal (found 2026-07-22, a real 7+ hour orphan from a prior
    # ungraceful exit — a killed, never-Quit() instance):
    # an ALREADY-RUNNING instance we own that never answers the scripting
    # API is a wedged orphan, not "still starting" — a fresh launch we
    # started ourselves this call already got the full
    # RESOLVE_STARTUP_TIMEOUT to come up on its own, so a second full wait
    # would only be worth it for a PRE-EXISTING instance we didn't just
    # launch. Without this, a wedged orphan blocks Resolve forever: no
    # future ensure_resolve_running() call can ever tell it apart from one
    # that's merely slow, and the normal Quit() path can't clear it
    # either (Quit() needs the very scripting connection that never came
    # up). Kill it and try ONE fresh launch before giving up.
    if already_running and ours_to_quit:
        print("  Scripting API never responded on an existing instance — "
              "treating it as a wedged orphan, killing and retrying once...")
        _kill_headless_resolve()
        time.sleep(2)
        subprocess.Popen(
            [RESOLVE_BIN, "-nogui"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        resolve = _wait_for_scripting_api()
        if resolve is not None:
            return resolve, True

    if ours_to_quit:
        # Launched (or claimed) by us and never answered: don't leave a
        # headless Resolve (GBs of RAM + GPU) running behind the error.
        _kill_headless_resolve()
        release_ownership()
    raise RuntimeError(
        f"Resolve did not become ready within {RESOLVE_STARTUP_TIMEOUT}s."
    )


def _render_clip(resolve, input_path, output_path, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                  codec=DEFAULT_CODEC):
    """
    Render a single clip within an already-connected Resolve session.
    Creates and tears down its own scratch project so clips don't interfere
    with each other. Returns the output path on success, None on failure.
    """
    pm = resolve.GetProjectManager()
    output_dir  = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(output_path))[0]

    # --- Create a fresh project ---
    # On a cold Resolve launch, scriptapp("Resolve") can return before the
    # Project Manager backend is actually ready — CreateProject silently
    # returns None for the first few seconds. Retry briefly before giving up.
    project_name = f"PROXY_{os.path.basename(input_path)}"
    existing = pm.LoadProject(project_name)
    if existing:
        pm.CloseProject(existing)
        pm.DeleteProject(project_name)

    project = None
    deadline = time.time() + 30
    while time.time() < deadline:
        project = pm.CreateProject(project_name)
        if project is not None:
            break
        time.sleep(2)

    if project is None:
        print(f"ERROR: Could not create Resolve project '{project_name}'.")
        return None

    # --- Import clip ---
    media_pool    = project.GetMediaPool()
    media_storage = resolve.GetMediaStorage()
    clips = media_storage.AddItemListToMediaPool([input_path])
    if not clips:
        print(f"ERROR: Resolve could not import: {input_path}")
        pm.DeleteProject(project_name)
        return None

    # --- Build timeline ---
    timeline = media_pool.CreateEmptyTimeline("Transcode")
    media_pool.AppendToTimeline(clips)
    project.SetCurrentTimeline(timeline)
    duration = timeline.GetEndFrame() - timeline.GetStartFrame()
    print(f"  Timeline: {duration} frames")

    # --- Render settings: load a matching preset base, enforce format/codec ---
    render_format = CODEC_PRESETS[codec]["format"]
    render_codec  = CODEC_PRESETS[codec]["codec"]
    if render_format == "mp4":
        presets = project.GetRenderPresets()
        for name in (list(presets.values()) if presets else []):
            if "H.264" in name or "H264" in name or "MP4" in name or "YouTube" in name:
                if project.LoadRenderPreset(name):
                    break
    # Must come AFTER loading a preset — presets override the format/codec otherwise
    project.SetCurrentRenderFormatAndCodec(render_format, render_codec)
    project.SetRenderSettings({
        "TargetDir":    output_dir,
        "CustomName":   output_name,
        "FormatWidth":  width,
        "FormatHeight": height,
        "ExportVideo":  True,
        "ExportAudio":  True,
    })

    # --- Render ---
    job_id = project.AddRenderJob()
    if not job_id:
        print("ERROR: Could not queue render job.")
        pm.DeleteProject(project_name)
        return None

    project.StartRendering([job_id])
    print("  Rendering... ", end="", flush=True)
    render_deadline = time.time() + RESOLVE_RENDER_TIMEOUT
    while project.IsRenderingInProgress():
        if time.time() > render_deadline:
            print("")
            print(f"ERROR: render still in progress after "
                  f"{RESOLVE_RENDER_TIMEOUT}s — giving up on this clip.")
            try:
                project.StopRendering()
            except Exception as e:
                print(f"  (StopRendering failed: {e.__class__.__name__}: {e})")
            pm.CloseProject(project)
            pm.DeleteProject(project_name)
            return None
        time.sleep(2)
        print(".", end="", flush=True)
    print(" done.")

    # --- Cleanup this clip's scratch project ---
    # Must close it first — DeleteProject silently fails on the currently-open project.
    pm.CloseProject(project)
    pm.DeleteProject(project_name)

    # --- Confirm output ---
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved: {output_path}  ({size_mb:.1f} MB)")
        return output_path
    else:
        print(f"WARNING: Output file not found at expected path: {output_path}")
        return None


def transcode(input_path, output_path=None, output_dir=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
              codec=DEFAULT_CODEC, suffix=None):
    """
    Transcode a single input_path to a proxy via headless DaVinci Resolve Studio.
    Connects, renders, and quits Resolve (if this call launched it) — a
    convenience wrapper around transcode_batch() for one file.

    output_dir: optional directory to place the output in (ignored if
                output_path is given). Created automatically if it doesn't
                already exist.

    Returns the output path on success, or None on failure.
    """
    results = transcode_batch([input_path], output_paths=[output_path] if output_path else None,
                               output_dir=output_dir, width=width, height=height, codec=codec, suffix=suffix)
    return results[input_path]


def transcode_batch(input_paths, output_paths=None, output_dir=None,
                     width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, codec=DEFAULT_CODEC, suffix=None):
    """
    Transcode a queue of clips to proxies, launching Resolve only once for
    the whole batch (much faster than one launch per file).

    input_paths:  list of source media paths (mixed formats OK — BRAW, R3D, MXF, etc.)
    output_paths: optional list of output paths, same length/order as input_paths.
                  If omitted, defaults to '<input><suffix>' next to each source
                  (or inside output_dir, if given).
    output_dir:   optional directory to place all outputs in (ignored per-file
                  if output_paths is given). Defaults to next to each source file.
                  Created automatically if it doesn't already exist.
    width/height: proxy resolution (defaults to 1920x1080)
    codec:        one of CODEC_PRESETS' keys — "h264" (default), "prores422proxy",
                  "prores422lt", "prores422", "prores422hq", "prores4444", "prores4444xq"
    suffix:       appended to the input's basename when no output path is given
                  (defaults to "_proxy" + the chosen codec's container extension,
                  e.g. "_proxy.mp4" for h264 or "_proxy.mov" for any ProRes flavor)

    Returns a dict mapping input_path -> output_path (or None on failure).
    """
    if codec not in CODEC_PRESETS:
        raise ValueError(f"Unknown codec '{codec}'. Choices: {', '.join(CODEC_PRESETS)}")
    if suffix is None:
        suffix = "_proxy" + CODEC_PRESETS[codec]["extension"]

    if output_paths is None:
        output_paths = [None] * len(input_paths)
    if len(output_paths) != len(input_paths):
        raise ValueError("output_paths must be the same length as input_paths")

    resolved_outputs = []
    for input_path, output_path in zip(input_paths, output_paths, strict=True):
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            if output_dir:
                output_path = os.path.join(output_dir, os.path.basename(base) + suffix)
            else:
                output_path = base + suffix
        resolved_outputs.append(output_path)

    # Create each output's destination directory if it doesn't exist yet
    # (covers --output-dir, an explicit -o/--output, and default output_paths).
    for output_path in resolved_outputs:
        dest_dir = os.path.dirname(output_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

    print("")
    print("DaVinci Resolve Studio")
    resolve, we_launched = ensure_resolve_running()
    print(f"  Connected: {resolve.GetVersionString()}")

    results = {}
    total = len(input_paths)
    for i, (input_path, output_path) in enumerate(
            zip(input_paths, resolved_outputs, strict=True), start=1):
        print("")
        print(f"[{i}/{total}] {os.path.basename(input_path)}")
        print(f"  Output: {output_path}")
        results[input_path] = _render_clip(resolve, input_path, output_path, width, height, codec)

    if we_launched:
        # release_ownership() must run even if Quit() raises — otherwise the
        # marker keeps naming this (about-to-exit) PID and the next session's
        # ensure_resolve_running() can't reclaim the instance as an orphan.
        # A failed Quit() also must not swallow the batch summary printed
        # below — the render results are the point of the run.
        quit_started = time.time()
        try:
            resolve.Quit()
            print(f"  Resolve Quit() took {time.time() - quit_started:.1f}s")
        except Exception as e:
            print(f"WARNING: Resolve Quit() failed after "
                  f"{time.time() - quit_started:.1f}s "
                  f"({e.__class__.__name__}: {e}) — it may still be running")
        finally:
            release_ownership()

    print("")
    print("Batch summary")
    for input_path in input_paths:
        status = "OK" if results[input_path] else "FAILED"
        print(f"  [{status}] {input_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Headless DaVinci Resolve Studio proxy generator (single file or batch)."
    )
    parser.add_argument("inputs", nargs="+", help="one or more source media files")
    parser.add_argument("-o", "--output", help="explicit output path (single-file only)")
    parser.add_argument("--output-dir", help="directory to place all outputs in "
                                              "(created automatically if it doesn't exist)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"proxy width (default {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"proxy height (default {DEFAULT_HEIGHT})")
    parser.add_argument("--codec", choices=sorted(CODEC_PRESETS), default=DEFAULT_CODEC,
                         help=f"render codec (default '{DEFAULT_CODEC}')")
    parser.add_argument("--suffix", default=None,
                         help="filename suffix when no explicit output is given "
                              "(default '_proxy' + the codec's container extension, "
                              "e.g. '_proxy.mp4' for h264 or '_proxy.mov' for ProRes)")
    args = parser.parse_args()

    if args.output and len(args.inputs) > 1:
        parser.error("--output can only be used with a single input file")

    print("")
    print("=" * 45)
    print("  Resolve Proxy Generator")
    print("=" * 45)

    try:
        if len(args.inputs) == 1:
            print(f"  Input: {args.inputs[0]}")
            transcode(args.inputs[0], args.output, output_dir=args.output_dir, width=args.width, height=args.height,
                      codec=args.codec, suffix=args.suffix)
        else:
            print(f"  Queue: {len(args.inputs)} file(s)")
            transcode_batch(args.inputs, output_dir=args.output_dir,
                             width=args.width, height=args.height, codec=args.codec, suffix=args.suffix)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
