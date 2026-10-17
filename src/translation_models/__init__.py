"""
Download-on-demand for the OPUS-MT translation models used by opus_translate.py.

Each language's ~300 MB of NMT weights is NOT in the repo — it is fetched the
first time that language is requested (`--lang es`, or `python -m
translation_models download es`), into config.translation_model_dir(lang),
then loaded in-process by opus_translate. Adding a language is a data change
(a manifest entry + an upload), not a code change.

A streaming download with a
progress callback, sha256 verification, and an ATOMIC install (a failed or
interrupted download never leaves a half-installed model dir that would later
load as corrupt). Host-agnostic: a bundled manifest maps each language to its
downloadable components (archive URL + sha256 + byte size).

PUBLIC MODEL STORE
The model artifacts live in a public GitHub repo (alvroga/nmt-model-store), so a
fresh machine downloads them with no credentials. VMT_MODELS_TOKEN is honored
as an OPTIONAL bearer token for anyone who points the manifest at a private
mirror.

LICENSING
The models are Helsinki-NLP OPUS-MT weights (tc-big: CC-BY-4.0; base:
Apache-2.0) — both permit redistribution of the format-converted derivative
with attribution. Each component archive carries its own LICENSE + ATTRIBUTION,
and the per-language `attribution` block in the manifest records the credits.
"""
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import config

# Manifest lives inside this package, resolved relative to this module.
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

_manifest_cache = None


def _manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        try:
            _manifest_cache = json.loads(MANIFEST_PATH.read_text())
        except Exception as e:
            print(f"  WARNING: translation manifest unreadable ({e})", flush=True)
            _manifest_cache = {"schema": 1, "base_url": "", "languages": {}}
    return _manifest_cache


def language_entry(lang: str) -> "dict | None":
    return _manifest().get("languages", {}).get(lang)


def language_name(lang: str) -> "str | None":
    e = language_entry(lang)
    return e.get("name") if e else None


def is_downloadable(lang: str) -> bool:
    """A manifest entry with at least one component exists for `lang`."""
    e = language_entry(lang)
    return bool(e and e.get("components"))


def _component_dir(lang: str, comp: dict) -> Path:
    return config.translation_model_dir(lang) / comp["name"]


def _marker(comp_dir: Path) -> Path:
    return comp_dir / ".installed"


def _component_ok(lang: str, comp: dict) -> bool:
    """Installed AND matches the manifest sha256 (a manifest bump forces a
    re-download rather than silently keeping stale weights)."""
    m = _marker(_component_dir(lang, comp))
    try:
        return m.read_text().strip() == comp["sha256"]
    except OSError:
        return False


def missing_components(lang: str) -> list:
    e = language_entry(lang)
    if not e:
        return []
    return [c for c in e.get("components", []) if not _component_ok(lang, c)]


def is_installed(lang: str) -> bool:
    """True when every component this language declares is installed and
    verified. English (no manifest entry, no translation needed) is trivially
    'installed'."""
    e = language_entry(lang)
    if not e:
        return lang == "en"
    return not missing_components(lang)


def download_size(lang: str) -> int:
    """Total bytes still to download for `lang` (0 if fully installed)."""
    return sum(int(c.get("size", 0)) for c in missing_components(lang))


def _auth_headers() -> dict:
    token = os.environ.get("VMT_MODELS_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _component_url(comp: dict) -> str:
    if comp.get("url"):
        return comp["url"]
    return _manifest().get("base_url", "") + comp["archive"]


def _stream_to_file(url: str, dest: Path, on_bytes) -> None:
    """Download `url` -> `dest`, calling on_bytes(n_new_bytes) as it goes.
    file:// URLs are supported for local/dev testing (no server needed)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        src = Path(urllib.parse.unquote(parsed.path))
        with open(src, "rb") as r, open(dest, "wb") as w:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                w.write(chunk)
                on_bytes(len(chunk))
        return
    req = urllib.request.Request(url)
    # Unredirected: GitHub release assets redirect to a storage host, which
    # must never receive the optional token.
    for k, v in _auth_headers().items():
        req.add_unredirected_header(k, v)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as w:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            w.write(chunk)
            on_bytes(len(chunk))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a tar archive to `dest`, refusing any member that would escape
    the target dir (path traversal guard)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        base = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tar.extractall(dest, filter="data")  # + reject symlink/absolute/device members


def _install_component(lang: str, comp: dict, on_bytes) -> None:
    """Download -> verify sha256 -> extract -> atomically swap into place ->
    write marker. Raises on any failure (caller reports it); leaves no
    partial install behind."""
    comp_dir = _component_dir(lang, comp)
    comp_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{comp['name']}-", dir=comp_dir.parent))
    try:
        archive = work / comp["archive"]
        _stream_to_file(_component_url(comp), archive, on_bytes)
        actual = _sha256(archive)
        if actual != comp["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {comp['name']} "
                f"(expected {comp['sha256'][:12]}…, got {actual[:12]}…)")
        staged = work / "extracted"
        _safe_extract(archive, staged)
        # Move the old install aside rather than deleting it outright, so the
        # window where NEITHER version is in place is one os.replace long
        # instead of spanning an rmtree. Both are same-filesystem renames
        # (`work` is created inside comp_dir.parent precisely for this), and
        # the old copy is discarded with `work` in the finally below.
        if comp_dir.exists():
            os.replace(comp_dir, work / "superseded")
        os.replace(staged, comp_dir)
        _marker(comp_dir).write_text(comp["sha256"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def download_language(lang: str, progress_cb=None) -> tuple:
    """Download + install every missing component for `lang`.

    progress_cb(completed_bytes, total_bytes, phase_text) is called from THIS
    thread as the download proceeds. Returns (ok, message). Never raises — a
    failure returns (False, reason) so the caller can fall back to English.
    """
    if not is_downloadable(lang):
        return (lang == "en"), ("no translation needed" if lang == "en"
                                else f"no models available for '{lang}'")
    missing = missing_components(lang)
    if not missing:
        return True, "already installed"

    total = sum(int(c.get("size", 0)) for c in missing) or 1
    done = [0]

    def make_on_bytes(phase):
        def on_bytes(n):
            done[0] += n
            if progress_cb:
                progress_cb(min(done[0], total), total, phase)
        return on_bytes

    for comp in missing:
        phase = f"downloading {comp['name']}…"
        if progress_cb:
            progress_cb(done[0], total, phase)
        try:
            _install_component(lang, comp, make_on_bytes(phase))
        except Exception as e:
            return False, f"{comp['name']}: {e}"
    if progress_cb:
        progress_cb(total, total, "done")
    return True, "installed"


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OPUS-MT model download manager")
    parser.add_argument("command", choices=["status", "download"])
    parser.add_argument("lang")
    args = parser.parse_args()
    if args.command == "status":
        print(f"{args.lang}: downloadable={is_downloadable(args.lang)} "
              f"installed={is_installed(args.lang)} "
              f"missing={[c['name'] for c in missing_components(args.lang)]} "
              f"to_download={download_size(args.lang)/1e6:.0f}MB")
        return

    def bar(done, total, phase):
        pct = 100 * done / total if total else 0
        print(f"\r  {phase:20} {pct:5.1f}% ({done/1e6:.0f}/{total/1e6:.0f}MB)",
              end="", flush=True)
    ok, msg = download_language(args.lang, bar)
    print(f"\n{'OK' if ok else 'FAILED'}: {msg}")


if __name__ == "__main__":
    _main()
