"""
Batch runs for the app's Analyze button and the scheduled timer: scan the
selected sources (folders recursively, or single files), skip what the
output sinks already have, run tagger.process_file on the rest and hand
each record to the sinks.

Progress goes to stdout as print() lines (the app captures them for its
live log and phase label) plus the on_* callbacks.
"""
import json
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import config
import library
import sinks as sinks_mod
import tagger

UNDECODABLE_LOG = "undecodable.jsonl"

# Statuses that mean "this file can't be read here" (not a transient
# failure): listed on the Skipped Files screen, and not retried while the
# file is unchanged (or until the user presses Retry), so a scheduled run
# doesn't re-attempt the same RAW clip every hour.
UNDECODABLE_STATUSES = {"undecodable", "image-undecodable", "resolve-unavailable"}
# Consecutive failures of these mean something systemic (engine died, share
# dropped), not a bad file.
_FAILURE_STATUSES = {"error", "pipeline-failed"}
_BREAKER_LIMIT = 3


class BatchAbort(RuntimeError):
    """A pre-flight check failed and nothing was processed (engine not
    ready). Distinct from a healthy run that found nothing to do, which
    also returns empty stats."""


# Stable substrings of pipeline/tagger print lines -> the Analyze screen's
# phase label. Keep in lockstep with those prints. Full "[1/3] <phrase>"
# prefixes: the per-file header also starts with "[i/total]".
PHASE_MARKERS = (
    ("[1/3] Detecting shot cuts", "Detecting shot cuts…"),
    ("[1/3] cuts.json exists", "Detecting shot cuts…"),
    ("[2/3] Extracting frames", "Extracting frames…"),
    ("[2/3] frames/index.json exists", "Extracting frames…"),
    ("[3/3] Detecting objects", "Detecting objects…"),
    ("proxy via DaVinci Resolve Studio", "Transcoding via DaVinci Resolve Studio…"),
    ("[still]", "Analyzing still image…"),
)


def phase_for_line(line: str) -> str | None:
    for marker, label in PHASE_MARKERS:
        if marker in line:
            return label
    return None


@dataclass
class Candidate:
    source: str   # the selected folder or file it came from
    path: Path


def sweep_orphan_workdirs() -> int:
    """Delete data/<video_id>/ working folders stranded by an interrupted
    run (quit mid-file, crash, power loss). A finished or stopped file
    always cleans up after itself, so at batch start any directory left is
    an orphan. Only directories directly under DATA_DIR; never symlinks or
    mount points, never plain files (undecodable.jsonl, records.jsonl)."""
    try:
        children = list(config.DATA_DIR.iterdir())
    except OSError:
        return 0
    swept = 0
    for child in children:
        if child.is_symlink() or not child.is_dir() or os.path.ismount(child):
            continue
        shutil.rmtree(child, ignore_errors=True)
        swept += 1
    if swept:
        print(f"Cleaned up {swept} leftover working folder(s) from an interrupted run.")
    return swept


# -- the Skipped Files list -------------------------------------------------------

def _undecodable_path() -> Path:
    return config.DATA_DIR / UNDECODABLE_LOG


def read_undecodable() -> tuple[list[dict], int]:
    """(entries oldest first, unparsable line count)."""
    try:
        lines = _undecodable_path().read_text(errors="replace").splitlines()
    except OSError:
        return [], 0
    rows, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(entry, dict):
            rows.append(entry)
        else:
            bad += 1
    return rows, bad


def _write_undecodable(rows: list[dict]) -> None:
    path = _undecodable_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _stat(path: Path) -> dict:
    try:
        st = path.stat()
        return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return {}


def log_undecodable(path: Path, source: str, reason: str) -> None:
    """One row per file: a repeat failure replaces the earlier row. The
    file's size + mtime are kept so a changed file is tried again."""
    rows, _ = read_undecodable()
    rows = [r for r in rows if r.get("path") != str(path)]
    rows.append({"path": str(path), "source": source, "reason": reason, **_stat(path),
                 "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    _write_undecodable(rows)


def clear_all_undecodable() -> None:
    """The user's Retry: every skipped file is a candidate again."""
    try:
        _undecodable_path().unlink()
    except FileNotFoundError:
        pass


def _known_undecodable() -> dict[str, dict]:
    """path -> the stat it had when it failed (only rows that recorded one)."""
    rows, _ = read_undecodable()
    return {r["path"]: {"size": r["size"], "mtime_ns": r["mtime_ns"]}
            for r in rows if "path" in r and "size" in r and "mtime_ns" in r}


def clear_undecodable(path: Path) -> None:
    """A file that now processes is no longer skipped."""
    rows, _ = read_undecodable()
    kept = [r for r in rows if r.get("path") != str(path)]
    if len(kept) != len(rows):
        _write_undecodable(kept)


# -- scanning -------------------------------------------------------------------

def collect_candidates(sources: list[str], tasks: set[str], output_sinks: list, *,
                       language: str, model: str | None
                       ) -> tuple[list[Candidate], dict[str, tuple[int, int]]]:
    """Candidates across all sources, plus per-source (done, total) before
    the run. A source that disappeared (share not mounted) is reported and
    skipped, not fatal."""
    candidates: list[Candidate] = []
    progress: dict[str, tuple[int, int]] = {}
    seen: set[Path] = set()
    skipped = _known_undecodable()
    held_back = 0
    for source in sources:
        p = Path(source)
        if p.is_dir():
            result = library.scan(p, tasks, output_sinks, language=language, model=model)
            todo = result.candidates
            progress[source] = (result.done, result.total)
        elif p.is_file() and p.suffix.lower() in library.MEDIA_EXTS:
            done = sinks_mod.is_done(output_sinks, p, tasks, language=language, model=model)
            todo = [] if done else [p]
            progress[source] = (1 if done else 0, 1)
        else:
            print(f"  NOTE: {source} is not available (unplugged, or a network volume that "
                  f"isn't mounted) — skipped this run")
            continue
        kept = []
        for path in todo:
            known = skipped.get(str(path))
            if known and _stat(path) == known:
                held_back += 1  # failed before, unchanged since
                continue
            kept.append(path)
        print(f"{source}: {len(kept)} to process")
        for path in kept:
            key = path.resolve()
            if key not in seen:  # a file inside two selected folders runs once
                seen.add(key)
                candidates.append(Candidate(source, path))
    if held_back:
        print(f"{held_back} file(s) that couldn't be decoded before are unchanged — not retried "
              f"(Skipped Files → Retry all to try them again)")
    return candidates, progress


def run_batch(
    sources: list[str],
    *,
    tasks: set[str],
    output_sinks: list,
    language: str = "en",
    redo_other_model: bool = False,
    resolve_enabled: bool = False,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_file: Callable[[str, Path], None] | None = None,
    on_source_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    """Process every not-yet-done file under `sources`. Returns
    status -> count. Raises BatchAbort when the engine isn't ready.

    stop_event is checked between files: the file in flight finishes.
    on_progress(completed, total); on_file(source, path) as each file
    starts; on_source_progress(source, done, total) as each file finishes.
    """
    model = config.VISION_MODEL
    ok, msg = config.vlm_status(model)
    if not ok:
        print(f"ERROR: {msg}")
        raise BatchAbort(msg)
    tagger.RESOLVE_ALLOWED = bool(resolve_enabled)
    sweep_orphan_workdirs()

    describe = "description" in tasks
    run_meta = {"model": model, "prompt_set": config.PROMPT_SET, "language": language,
                "tool_version": tagger.TOOL_VERSION, "tasks": sorted(tasks)}

    candidates, source_progress = collect_candidates(
        sources, tasks, output_sinks, language=language,
        model=model if redo_other_model else None)
    total = len(candidates)
    if on_progress:
        on_progress(0, total)
    if on_source_progress:
        for source, (done, count) in source_progress.items():
            on_source_progress(source, done, count)

    stats: dict[str, int] = {}
    consecutive_failures = 0
    for i, cand in enumerate(candidates, 1):
        if stop_event is not None and stop_event.is_set():
            print("\nStopped.")
            break
        path = cand.path
        try:
            # Inside the try: a filename that isn't valid UTF-8 (possible on
            # NFS) must fail this file, not the run.
            print(f"\n[{i}/{total}] {path}")
            if path.suffix.lower() in tagger.IMAGE_EXTS:
                print("[still] analyzing image")
            if on_file:
                on_file(cand.source, path)
            rec = tagger.process_file(path, model=model, lang=language, describe=describe,
                                      force=False, keep_workdir=False)
        except (Exception, SystemExit) as e:
            # One bad file must not end a long unattended run; the breaker
            # below catches a systemic cause. No sidecar is written, so the
            # file is retried next run.
            print(f"    ERROR: unexpected failure ({e.__class__.__name__}: {e})")
            rec = {"path": str(path), "status": "error", "error": str(e)}
            work = config.DATA_DIR / config.video_id_from_path(str(path), unique=True)
            if work.is_dir() and not work.is_symlink():
                shutil.rmtree(work, ignore_errors=True)
        status = rec["status"]
        print(f"    → {status}" + (f" ({rec['elapsed_sec']}s)" if "elapsed_sec" in rec else ""))
        stats[status] = stats.get(status, 0) + 1

        for sink in output_sinks:
            try:
                sink.write(path, {**rec, **run_meta})
            except Exception as e:  # a failing sink must not lose the run
                print(f"    WARNING: {sink.name} output failed for {path.name}: {e}")
        try:
            if status in UNDECODABLE_STATUSES:
                log_undecodable(path, cand.source, rec.get("error") or status)
            elif status == "ok":
                clear_undecodable(path)
        except (OSError, ValueError) as e:
            print(f"    WARNING: could not update the skipped-files list: {e}")

        if on_source_progress and status in ("ok", "redundant-chunk"):
            done, count = source_progress.get(cand.source, (0, 0))
            source_progress[cand.source] = (min(done + 1, count), count)
            on_source_progress(cand.source, *source_progress[cand.source])

        if status in _FAILURE_STATUSES:
            consecutive_failures += 1
            if consecutive_failures >= _BREAKER_LIMIT and i < total:
                print(f"\nAborting: {consecutive_failures} files in a row failed — this "
                      "looks systemic (the AI engine or a network volume went away), not a bad "
                      "file. Finished files are saved; fix the cause and run again to "
                      "continue where this left off.")
                break
        else:
            consecutive_failures = 0
        if on_progress:
            on_progress(i, total)
    return stats
