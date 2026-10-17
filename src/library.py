"""
Library scanner: which files in a folder still need processing.

A folder is any mounted path: a local disk, an external drive, an SMB/NAS
share. scan() walks it recursively, keeps media the pipeline handles
(tagger.VIDEO_EXTS / IMAGE_EXTS), and asks the output sinks which files are
already done for the requested tasks (sinks.is_done: by default the sidecar
beside each file, matched on size + mtime, so a changed file is redone).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import sinks as sinks_mod
from resolve_proxy import _R3D_CHUNK_RE
from tagger import IMAGE_EXTS, VIDEO_EXTS

MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS

# macOS and NAS housekeeping folders: never media, often unreadable.
_SKIP_DIRS = {".Trashes", ".Spotlight-V100", ".fseventsd", ".TemporaryItems",
              ".DocumentRevisions-V100", "@eaDir", "#recycle", ".AppleDouble"}


@dataclass
class ScanResult:
    total: int = 0                 # media files found
    done: int = 0                  # already processed for these tasks
    candidates: list[Path] = field(default_factory=list)


def iter_media(folder: Path):
    """Media files under `folder`, sorted per directory. Skips hidden
    folders, AppleDouble `._*` files and the later chunks of an R3D clip
    (_001.R3D stands for the whole clip). An unreadable subfolder is
    skipped, not fatal: one locked folder on a share shouldn't stop a scan."""
    for root, dirs, files in os.walk(folder, onerror=lambda e: None):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS)
        for name in sorted(files):
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() not in MEDIA_EXTS:
                continue
            m = _R3D_CHUNK_RE.match(name)
            if m and m.group("chunk") != "001":
                continue
            yield Path(root) / name


def scan(folder: Path, tasks: set[str], output_sinks: list, *, language: str = "en",
         model: str | None = None) -> ScanResult:
    """Everything in `folder` not yet done for `tasks`. `model` set = files
    whose output came from a different model count as not done (the opt-in
    "re-run files tagged by an older model" setting; never automatic)."""
    result = ScanResult()
    for media in iter_media(Path(folder)):
        result.total += 1
        if sinks_mod.is_done(output_sinks, media, tasks, language=language, model=model):
            result.done += 1
        else:
            result.candidates.append(media)
    return result
