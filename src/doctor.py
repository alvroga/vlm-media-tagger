#!/usr/bin/env python3
"""
Environment check — verifies what this machine can do and reports what's missing.

Required:  the inference backend (managed llama-server + MiniCPM-V 4.7 by
           default, or Ollama with VLM_BACKEND=ollama) running with its model.
Optional:  DaVinci Resolve Studio with scripting support (RAW-format fallback,
           see resolve_proxy.py). Missing Resolve is reported, never fatal —
           tagger.py reports RAW files as "resolve-unavailable" without it.

Usage:
  python src/doctor.py            # human-readable report, exit 1 if required checks fail
"""
import sys
from pathlib import Path

import config

RESOLVE_APP = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app")
RESOLVE_SCRIPT_MODULES = Path(
    "/Library/Application Support/Blackmagic Design"
    "/DaVinci Resolve/Developer/Scripting/Modules"
)


def check_resolve() -> tuple[bool, str]:
    """DaVinci Resolve Studio installed with the scripting API available?

    Both must be present for the RAW fallback: the application and the
    Developer/Scripting/Modules directory (ships with Resolve; the scripting
    API additionally requires the Studio edition at runtime).
    """
    if not RESOLVE_APP.exists():
        return False, ("DaVinci Resolve Studio not installed — RAW formats "
                       "(BRAW/R3D/X-OCN/ARRI) will be skipped on this machine. "
                       "Install DaVinci Resolve Studio to enable them.")
    if not RESOLVE_SCRIPT_MODULES.exists():
        return False, (f"DaVinci Resolve Studio is installed but the scripting module dir "
                       f"is missing ({RESOLVE_SCRIPT_MODULES}) — RAW formats will "
                       "be skipped. Reinstall Resolve Studio / enable scripting.")
    return True, "DaVinci Resolve Studio installed with scripting modules (RAW fallback available)"


def main() -> int:
    print("vlm-media-tagger environment check\n")

    ok, msg = config.vlm_status()
    print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    required_ok = ok

    ok, msg = check_resolve()
    print(f"  [{'OK' if ok else '--'}] {msg}")

    print()
    if required_ok:
        print("Ready — required checks passed."
              if ok else
              "Ready — required checks passed (Resolve optional, see above).")
        return 0
    print("NOT ready — fix the FAIL items above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
