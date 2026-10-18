#!/usr/bin/env python3
"""
End-to-end smoke test of the macOS app in dev mode: opens the real window
against a throwaway data folder (your settings are never touched), drives
the real front end from inside the page, and runs one real analysis.

  ./venv/bin/python scripts/app_smoke.py [media file or folder]
  python3.13 -S scripts/app_smoke.py --bundle "build/.../VLM Media Tagger.app" [media]

--bundle runs the same checks on a built app's own modules and dependencies
(its Resources/app + app_packages, no venv; -S keeps site-packages out).

Default media: tests/golden/entries/poster-bbb/frames/poster.jpg (local
fixture). Needs the AI engine (a model run takes ~15 s once loaded; a cold
model load several minutes). The managed engine is left running on exit.
Exits 0 when every check passes. VMT_SMOKE_FORBIDDEN="word1,word2" adds
words that must not appear anywhere in the UI text.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(tempfile.mkdtemp(prefix="vmt-smoke-"))
os.environ["VMT_APP_SUPPORT_DIR"] = str(WORK / "support")
os.environ["DATA_DIR"] = str(WORK / "data")
if "--bundle" in sys.argv:
    i = sys.argv.index("--bundle")
    bundle = (Path(sys.argv[i + 1]) / "Contents" / "Resources").resolve()
    del sys.argv[i:i + 2]
    sys.path[:0] = [str(bundle / "app"), str(bundle / "app_packages")]
    # The decode probe re-runs sys.executable; in the real app that is the
    # app's own stub, here a bare python that needs the bundle's path.
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path[:2])
    os.environ["VMT_SMOKE_BUNDLE"] = str(bundle)
else:
    sys.path.insert(0, str(ROOT / "src"))

import webview  # noqa: E402

import llama_manager  # noqa: E402
from vmt_app import app  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    RESULTS.append((bool(ok), what))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}", flush=True)


def js(window, expr: str):
    return window.evaluate_js(expr)


def wait_for(window, expr: str, timeout: float, every: float = 0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = js(window, expr)
        except Exception:
            value = None
        if value:
            return value
        time.sleep(every)
    return None


def scenario(window, backend: "app.Backend", media: Path) -> None:
    try:
        # Page loaded and the bridge snapshot applied.
        check(wait_for(window, "document.getElementById('sidebar-version').textContent", 20),
              "page loaded, snapshot applied (version shown)")
        check(js(window, "document.title") == "VLM Media Tagger", "window title")
        nav = js(window, "Array.from(document.querySelectorAll('.nav-item .label')).map(e=>e.textContent)")
        check(nav == ["Analyze", "Log", "Skipped Files", "Settings"], f"nav items {nav}")
        body = js(window, "document.body.innerText").lower()
        extra = [w.strip() for w in os.environ.get("VMT_SMOKE_FORBIDDEN", "").split(",") if w.strip()]
        for word in extra:
            check(word.lower() not in body, f"forbidden word absent: {word}")
        langs = js(window, "Array.from(document.getElementById('sel-language').options).map(o=>o.value)")
        check(langs and langs[0] == "en" and len(langs) >= 2, f"language options {langs}")

        # Engine: the setup overlay hides once the model is loaded.
        ready = wait_for(window, "document.getElementById('setup-overlay').hidden && "
                                 "document.getElementById('engine-text').textContent.indexOf('ready') !== -1",
                         600, every=2)
        check(ready, "AI engine ready, setup overlay hidden")
        # Resolve light: off unless the option is on (a clean profile has it off).
        light = js(window, "document.querySelector('#resolve-pill .dot').className + '|' + "
                           "document.getElementById('resolve-text').textContent")
        check(light == "dot off|DaVinci Resolve off", f"Resolve light shown: {light!r}")

        # Settings tabs switch.
        for tab in ("tab-output", "tab-automation", "tab-integrations", "tab-general"):
            js(window, f"document.querySelector('.tab-btn[data-tab=\"{tab}\"]').click()")
            check(js(window, f"!document.getElementById('{tab}').hidden"), f"settings tab {tab} shows")
        js(window, "document.getElementById('nav-analysis').click()")

        # Add a source (the dialog itself needs a human; this is what it calls).
        backend.add_sources([str(media)])
        rows = wait_for(window, "document.querySelectorAll('#sources-list .source-row').length", 10)
        check(rows == 1, "source row appears")
        counted = wait_for(window, "document.querySelector('#sources-list .progress').textContent", 30)
        check(counted and "analyzed" in counted, f"progress count shown ({counted!r})")

        # Analyze through the real button.
        js(window, "document.getElementById('btn-analyze').click()")
        check(wait_for(window, "document.body.classList.contains('running')", 15), "run started")
        done = wait_for(window, "!document.body.classList.contains('running') && "
                                "document.getElementById('run-status').textContent", 600, every=2)
        check(done and done.startswith("Done"), f"run finished: {done!r}")
        target = media if media.is_file() else next(p for p in media.rglob("*") if p.is_file())
        sidecar = target.with_name(target.name + ".json")
        check(sidecar.exists(), f"sidecar written: {sidecar.name}")
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            check(data.get("tags") and data.get("description"), f"sidecar has tags ({len(data.get('tags', []))}) "
                                                                  "and a description")
        log_lines = js(window, "document.querySelectorAll('#runlog .ln').length")
        check(log_lines and log_lines > 3, f"log screen filled ({log_lines} lines)")
        counted = wait_for(window, "document.querySelector('#sources-list .progress').textContent"
                                   ".indexOf('1 / 1') === 0", 10)
        check(counted, "source shows 1 / 1 analyzed after the run")

        # A second run finds nothing to do.
        js(window, "document.getElementById('btn-analyze').click()")
        time.sleep(1)
        again = wait_for(window, "!document.body.classList.contains('running') && "
                                 "document.getElementById('run-status').textContent.indexOf('Done') === 0 && "
                                 "document.getElementById('run-status').textContent", 120, every=1)
        check(again and "nothing new" in again, f"second run resumes: {again!r}")
    except Exception as e:
        check(False, f"scenario crashed: {e.__class__.__name__}: {e}")
    finally:
        window.destroy()


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "tests/golden/entries/poster-bbb/frames/poster.jpg"
    if not src.exists():
        print(f"no such media: {src}")
        return 2
    media_dir = WORK / "media"
    media_dir.mkdir(parents=True)
    media = media_dir / src.name
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, media)
    print(f"smoke test in {WORK}")
    print(f"app code: {Path(app.__file__).parent}")
    if os.environ.get("VMT_SMOKE_BUNDLE"):
        stray = [m.__file__ for m in list(sys.modules.values())
                 if getattr(m, "__file__", None) and "site-packages" in m.__file__]
        check(not stray, f"nothing loaded from a venv/site-packages ({stray[:2]})")
        check(str(llama_manager.find_binary()).startswith(os.environ["VMT_SMOKE_BUNDLE"]),
              f"bundled llama-server used ({llama_manager.find_binary()})")

    llama_manager.shutdown = lambda *a, **k: None  # leave the engine running for the next run
    app.LOGS_DIR = WORK / "logs"  # never prune or write the real run logs
    backend = app.Backend()
    window = webview.create_window("VLM Media Tagger", url=str(app.WEBUI_DIR / "index.html"),
                                   js_api=app.Api(backend), width=1120, height=840)
    backend.attach(window)

    def started():
        backend.on_gui_started()
        threading.Thread(target=scenario, args=(window, backend, media), daemon=True).start()

    webview.start(started)
    backend.shutdown()
    failed = [w for ok, w in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)
    return 1 if failed or not RESULTS else 0


if __name__ == "__main__":
    code = main()
    os._exit(code)
