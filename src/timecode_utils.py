"""SMPTE timecode helpers."""
from timecode import Timecode


def seconds_to_tc(seconds: float, fps: float, drop_frame: bool) -> str | None:
    """Wall-clock seconds -> "HH:MM:SS:FF" (";" before frames for drop-frame).

    The `+ 1` is required, not an off-by-one: the `timecode` library counts
    frames 1-based — `Timecode(fps, frames=0)` raises ("should be a positive
    integer bigger than zero") and `frames=1` is what renders as
    "00:00:00:00" (verified against timecode 1.4.x, 2026-07-30). So our
    0-based frame number has to be shifted up by one on the way in.

    The library ALSO already emits ";" itself for genuine drop-frame rates
    (29.97/59.94) — verified: `Timecode(29.97, frames=1)` renders
    "00:00:00;00" unprompted. The `drop_frame` substitution below is
    therefore only a belt-and-suspenders normalizer for a caller that asks
    for drop-frame at a non-drop-frame rate; it is currently unreached in
    production (detect_cuts.py, the only caller, passes drop_frame=False).

    None when the library can't express the rate (it raises
    ZeroDivisionError below 1 fps, e.g. a time-lapse): the timecode is an
    informational field, never worth failing a file over.
    """
    frame_number = max(0, int(seconds * fps))
    try:
        s = str(Timecode(fps, frames=frame_number + 1))
    except (ZeroDivisionError, ValueError, TypeError):
        return None
    if drop_frame:
        # Swap the LAST separator rather than slicing at a fixed index —
        # equivalent for "HH:MM:SS:FF" but doesn't silently corrupt the
        # string if the library's format ever changes (e.g. HH:MM:SS.mmm).
        head, sep, frames = s.rpartition(":")
        if sep:
            s = f"{head};{frames}"
    return s


def add_preroll(seconds: float, preroll_sec: float = 7.0) -> float:
    return max(0.0, seconds - preroll_sec)


def add_postroll(seconds: float, postroll_sec: float, duration_sec: float | None = None) -> float:
    result = seconds + postroll_sec
    if duration_sec is not None:
        result = min(result, duration_sec)
    return result
