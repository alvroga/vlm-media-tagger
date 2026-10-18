"""
Network shares (SMB, AFP, NFS) for the app's sources.

The app never handles share passwords. A share is mounted by macOS itself
(Finder's "Connect to Server", or `open smb://…`, which uses the Keychain),
so from the scanner's point of view it is just a folder under /Volumes.
This module only:
  - tells which mounted volume a path lives on, and that volume's URL,
    from `mount` (so the app can remember it);
  - asks macOS to reconnect a remembered share at launch;
  - unmounts shares on quit, when the user opted in.
"""
import re
import subprocess
import time
from pathlib import Path

NETWORK_FS = {"smbfs", "afpfs", "nfs", "webdav"}

# "//user@host/share on /Volumes/share (smbfs, nodev, nosuid, mounted by user)"
_SHARE_URL_RE = re.compile(r"^(smb|afp|nfs|cifs)://[^\s]+$")
_MOUNT_LINE_RE = re.compile(r"^(?P<src>\S.*?) on (?P<point>/.*?) \((?P<fs>[^,)]+)")


def network_mounts() -> dict[str, dict]:
    """Mounted network volumes: mount point -> {"url", "fs"}. Never raises."""
    try:
        out = subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_mount_output(out)


def parse_mount_output(text: str) -> dict[str, dict]:
    mounts = {}
    for line in text.splitlines():
        m = _MOUNT_LINE_RE.match(line)
        if not m or m.group("fs") not in NETWORK_FS:
            continue
        mounts[m.group("point")] = {"url": _share_url(m.group("src"), m.group("fs")),
                                    "fs": m.group("fs")}
    return mounts


def _share_url(source: str, fs: str) -> str:
    """mount's source column → a URL macOS can reopen. The user name is
    kept (it picks the right Keychain entry); a password never appears in
    `mount` output."""
    scheme = {"smbfs": "smb", "afpfs": "afp", "nfs": "nfs", "webdav": "https"}[fs]
    if fs == "webdav":
        return source
    if fs == "nfs":  # host:/export
        host, _, export = source.partition(":")
        return f"nfs://{host}{export}"
    return f"{scheme}:{source}" if source.startswith("//") else f"{scheme}://{source}"


def share_for_path(path: str, mounts: dict[str, dict] | None = None) -> tuple[str, str] | None:
    """(mount point, share URL) of the network volume holding `path`, or None
    for a local path."""
    mounts = network_mounts() if mounts is None else mounts
    best = None
    for point in mounts:
        if path == point or path.startswith(point.rstrip("/") + "/"):
            if best is None or len(point) > len(best):
                best = point
    return (best, mounts[best]["url"]) if best else None


def reconnect(url: str, mount_point: str, timeout: float = 30.0) -> bool:
    """Ask macOS to mount `url` (Keychain credentials; macOS shows its own
    login prompt if none are saved) and wait for `mount_point` to appear.
    `open -g` keeps Finder in the background. Only file-share URLs: a stray
    https:// (WebDAV) would just open a browser."""
    if not _SHARE_URL_RE.match(url or ""):
        return False
    if Path(mount_point).is_dir() and mount_point in network_mounts():
        return True
    try:
        subprocess.run(["open", "-g", url], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mount_point in network_mounts():
            return True
        time.sleep(1)
    return False


def connect_dialog(url: str) -> bool:
    """Open a server address (smb://nas.local) the way Finder's Connect to
    Server does: macOS shows its share picker and login."""
    if not _SHARE_URL_RE.match(url.strip()):
        return False
    try:
        subprocess.run(["open", url.strip()], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def unmount(mount_point: str) -> bool:
    """Best-effort unmount of a network volume. Refuses anything that isn't
    a currently mounted network share, so a bad settings entry can never
    eject a local disk."""
    if mount_point not in network_mounts():
        return False
    try:
        r = subprocess.run(["diskutil", "unmount", mount_point],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0
