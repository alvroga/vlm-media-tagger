#!/usr/bin/env bash
# Unit tests (no VLM, no Resolve: both are mocked). Non-zero exit on failure.
# The golden harness needs Ollama and is run separately (docs/how-to-run.md).
set -euo pipefail
cd "$(dirname "$0")/.."
./venv/bin/python -m pytest -q "$@"
