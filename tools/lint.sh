#!/usr/bin/env bash
# Lint: ruff (pyflakes + import order). Non-zero exit on any finding.
set -euo pipefail
cd "$(dirname "$0")/.."
./venv/bin/ruff check src tests
