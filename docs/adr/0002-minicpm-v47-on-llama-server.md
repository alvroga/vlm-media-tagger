# ADR-0002: MiniCPM-V 4.7 on a managed llama-server as the default engine

Status: Accepted
Date: 2026-10-17
Supersedes / related: ADR-0001

## Context
MiniCPM-V 4.5 on Ollama was the certified model. MiniCPM-V 4.7 35B-A3B (released 2026-10-12) measured better
tags, better calibrated naming and ~1.7× faster per frame (`docs/research/minicpm-v4.7-evaluation.md`), but
Ollama 0.32 and 0.40 can't load it (`unknown projector type: minicpmv4_7`). llama.cpp master runs it.

## Decision
1. Default backend `VLM_BACKEND=llama`: llama.cpp `llama-server` pinned at b76b133, built static with Metal
   and no UI/OpenSSL by `scripts/build_llama_cpp.sh` (one ~28 MB binary that links only system frameworks).
   `llama_manager.py` spawns and stops it on a private port and fetches MiniCPM-V 4.7 Q4_K_M + the f16
   projector, sha256-pinned: reused from a local Ollama store when identical, otherwise a resumable Hugging
   Face download. `LLAMA_HOST` points at a server you run yourself instead.
2. Ollama stays selectable (`VLM_BACKEND=ollama`, MiniCPM-V 4.5) through the same `vlm.chat` call.
3. A 4.7 prompt set, `PROMPT_SETS["minicpm-v4.7"]`, derived from the 4.5 set by the study's measured clause
   swaps (V8 + D10 long-form stills); `config.PROMPT_SET` follows the model.
4. The parser salvages truncated JSON, which fixed 4.7's empty tag runs on text-dense images.

## Consequences
+ Better tags and naming at lower latency; long-form still descriptions.
− RAM floor rises to ~32 GB (48 GB comfortable): Q4_K_M holds ~23 GB. 24 GB Macs use the Ollama/4.5 backend.
− We own a llama.cpp build: an engine upgrade means rebuilding, then the harness canary.
− The 4.7 set is certified on the study rig only; the shipped engine needs its own harness run and baselines.
