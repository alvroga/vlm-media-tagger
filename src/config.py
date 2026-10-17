"""
Central config — every tunable comes from the environment (or a project
`.env`), with the measured defaults below. Import this before `ollama`
anywhere: ollama-python builds its default client from OLLAMA_HOST at
import time.

Two inference backends (VLM_BACKEND):
  llama   (default) MiniCPM-V 4.7 35B-A3B on llama.cpp's llama-server, which
          supports 4.7's vision projector (Ollama doesn't yet). Managed:
          llama_manager.py spawns the vendored binary on a private port and
          fetches the model files. Set LLAMA_HOST to use a server you run.
  ollama  any Ollama vision model (MiniCPM-V 4.5 is the certified one).
"""
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# True inside a built .app (briefcase); the bundle is read-only, so runtime
# data then lives under APP_SUPPORT_DIR instead of the checkout.
IS_FROZEN = ".app/Contents/" in str(PROJECT_ROOT)
APP_NAME = "VLM Media Tagger"
APP_SUPPORT_DIR = Path(os.environ.get("VMT_APP_SUPPORT_DIR")
                       or Path.home() / "Library" / "Application Support" / APP_NAME)


def _int_setting(name: str, default: int, minimum: int = 1) -> int:
    """Integer env setting; a non-numeric value warns and falls back."""
    raw = os.environ.get(name, "")
    try:
        value = int(raw) if raw else default
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a number — using the "
              f"default {default}", file=sys.stderr)
        value = default
    return max(minimum, value)


# Working dirs (cuts.json, frames/, objects.json, Resolve proxies) per file.
DATA_DIR = Path(os.environ.get("DATA_DIR")
                or (APP_SUPPORT_DIR / "data" if IS_FROZEN else PROJECT_ROOT / "data"))

# -- inference backend -------------------------------------------------------
VLM_BACKEND = os.environ.get("VLM_BACKEND", "llama").lower()

# llama-server (MiniCPM-V 4.7). LLAMA_HOST set = an external server you run
# (unmanaged); unset = llama_manager spawns the vendored binary on LLAMA_PORT.
LLAMA_PORT = _int_setting("LLAMA_PORT", 11471)
LLAMA_MANAGED = not os.environ.get("LLAMA_HOST")
LLAMA_HOST = os.environ.get("LLAMA_HOST") or f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_MODELS_DIR = Path(os.environ.get("LLAMA_MODELS_DIR") or APP_SUPPORT_DIR / "models")
# Context per request slot. A 1280px frame is ~1.1K image tokens; 8K was
# measured at ~23.4 GB RSS for the Q4_K_M weights. Keep it modest.
LLAMA_CTX = _int_setting("LLAMA_CTX", 8192, minimum=2048)

# Ollama (MiniCPM-V 4.5 or any other Ollama vision model).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST  # ollama-python reads it at import
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "openbmb/minicpm-v4.5")

LLAMA_MODEL_NAME = "minicpm-v4.7-35b-a3b-q4_k_m"
VISION_MODEL = os.environ.get("VISION_MODEL") or (
    LLAMA_MODEL_NAME if VLM_BACKEND == "llama" else OLLAMA_VISION_MODEL)
# Back-compat alias: pipeline modules historically read this name.
if VLM_BACKEND == "llama":
    OLLAMA_VISION_MODEL = VISION_MODEL

# Which tuned prompt set (src/prompts.py) — follows the model unless forced.
PROMPT_SET = os.environ.get("PROMPT_SET") or (
    "minicpm-v4.7" if "4.7" in VISION_MODEL else "minicpm-v4.5")

# -- output ------------------------------------------------------------------
# "en" is a no-op; any language with OPUS-MT models in
# src/translation_models/manifest.json translates the finished English tags
# and description (see opus_translate.py).
OUTPUT_LANGUAGE = os.environ.get("OUTPUT_LANGUAGE", "en")

# Output plugins (planned, ADR-0003): comma-separated names or `module:Class` specs.
OUTPUT_SINKS = os.environ.get("OUTPUT_SINKS", "sidecar")

# Concurrent VLM requests. 1 is MEASURED, not a guess (Ollama + 4.5: every
# level from 2 to 8 was ~48% slower than serial). Re-measure before raising.
PARALLEL_WORKERS = _int_setting("PARALLEL_WORKERS", 1)

# Drop/redact profanity genuinely present in the media (a sign, a shirt).
# Off by default: it is real content. Unrelated to opus_translate's
# always-on check for profanity a TRANSLATION invents.
FILTER_PROFANITY = os.environ.get("FILTER_PROFANITY", "").lower() == "true"

TRANSLATION_MODELS_DIR = Path(
    os.environ.get("TRANSLATION_MODELS_DIR") or APP_SUPPORT_DIR / "translation_models")


def translation_model_dir(lang: str) -> Path:
    """Install root for one language's OPUS-MT models: base/ (mandatory) and
    optionally tcbig/ (preferred) — see opus_translate.py."""
    return TRANSLATION_MODELS_DIR / lang


def _app_version() -> str:
    """pyproject.toml's [project] version in a checkout; the bundled
    dist-info in a frozen app."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            return tomllib.loads(pyproject.read_text())["project"]["version"]
        except Exception:
            pass
    try:
        from importlib import metadata
        return metadata.version("vlm-media-tagger")
    except Exception:
        return "unknown"


APP_VERSION = _app_version()


# -- helpers -----------------------------------------------------------------

def video_id_from_path(video_path: str, unique: bool = False) -> str:
    """Working-directory name under DATA_DIR.

    unique=True appends a short hash of the absolute path, so same-named
    files in different folders ("A001.mov" from two cards) never share a
    working dir and cross-contaminate cached cuts/frames."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video_path).stem)
    if not unique:
        return stem
    digest = hashlib.sha1(str(Path(video_path).resolve()).encode()).hexdigest()[:8]
    return f"{stem}-{digest}"


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks produced by thinking-mode models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def vlm_extra_kwargs(model: str) -> dict:
    """Per-model kwargs for every OLLAMA chat call. MiniCPM-V 4.x thinks by
    default on Ollama >= 0.31: in detection the whole token budget vanished
    into hidden reasoning and content came back empty. think=False restores
    fast clean output. (The llama backend sends enable_thinking=False
    itself — see vlm.py.)"""
    return {"think": False} if "minicpm-v4" in model.lower() else {}


def _get_json(url: str, timeout: float = 3) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def vlm_status(model: str | None = None) -> tuple[bool, str]:
    """Is the inference backend reachable and the model ready? Never raises."""
    if VLM_BACKEND == "llama":
        try:
            data = _get_json(LLAMA_HOST.rstrip("/") + "/health")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                return False, f"llama-server at {LLAMA_HOST} is still loading the model"
            return False, f"llama-server at {LLAMA_HOST} answered HTTP {e.code}"
        except (urllib.error.URLError, OSError):
            hint = ("run: "
                    "python src/llama_manager.py start") if LLAMA_MANAGED else "start your server"
            return False, f"llama-server is not running at {LLAMA_HOST} ({hint})"
        except ValueError:
            return False, f"Something is answering at {LLAMA_HOST} but it is not llama-server"
        if data.get("status") != "ok":
            return False, f"llama-server at {LLAMA_HOST} reports {data.get('status')!r}"
        return True, f"llama-server running at {LLAMA_HOST} ({VISION_MODEL})"

    model = model or OLLAMA_VISION_MODEL
    try:
        data = _get_json(OLLAMA_HOST.rstrip("/") + "/api/tags")
    except (urllib.error.URLError, OSError):
        return False, f"Ollama is not running at {OLLAMA_HOST}. Start it with: ollama serve"
    except ValueError:
        return False, (f"Something is answering at {OLLAMA_HOST} but it is not "
                       "Ollama — check OLLAMA_HOST")
    available = [m.get("name", "") for m in data.get("models") or []]
    if not any(m.startswith(model.split(":")[0]) for m in available):
        return False, (f"Model '{model}' not found in Ollama. Pull it with: "
                       f"ollama pull {model} (available: {available or '(none)'})")
    return True, f"Ollama running at {OLLAMA_HOST}, model '{model}' available"


# Back-compat name used across the pipeline and tests.
ollama_status = vlm_status


def check_vlm(model: str | None = None) -> None:
    """CLI guard: print an actionable error and exit(1) if the backend isn't ready."""
    ok, msg = vlm_status(model)
    if not ok:
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


check_ollama = check_vlm  # back-compat name
