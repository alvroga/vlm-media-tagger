# ADR-0001: Scope and stack — local VLM tagging to JSON, measured by a regression harness

Status: Accepted
Date: 2026-10-17
Supersedes / related: —

## Context
Searching real footage needs open-vocabulary tags (brands, vehicle models, materials, effects) and readable
descriptions, which closed-set detectors can't produce. Small local vision-language models can, but their raw
output is noisy: copied format examples, person nouns, invented place names, empty answers. The useful part is
the engineering around the model, and every prompt change needs evidence, not eyeballing.

## Decision
1. A Python 3.13 CLI (`src/tagger.py`) that takes files or folders and writes one JSON record per file: tags,
   time-coded segments, a description, status. No network services beyond the local model server.
2. Pipeline: decode probe in a subprocess → PySceneDetect shot cuts → PyAV frames (2 fps, 1280 px, pHash
   dedup) → sharpest frame per shot → VLM tags ×2 passes → per-shot segment merge → deterministic tag cleanup
   (`tag_filter.py`) → caption sample + combine call for the summary → optional profanity filter → optional
   CTranslate2 OPUS-MT translation.
3. RAW camera formats (BRAW, R3D, ARRIRAW, X-OCN, ProRes RAW) go through DaVinci Resolve Studio's scripting API
   for a 720p proxy when it is installed (`resolve_proxy.py`, from resolve-proxy-nogui); otherwise they are
   reported, never fatal.
4. Prompts live in `src/prompts.py` with the measured reason for every clause next to it. A prompt or model
   change is accepted only on a harness run against a saved baseline (`tests/golden/`), plus the COCO recall,
   Places365 false-place and LogoDet-3K false-brand tiers.
5. Tunables come from the environment or `.env` (`src/config.py`); no hard-coded paths or hosts.
6. Fixture images are not redistributed. Expectations, sampling scripts (fixed seeds) and baseline results are
   committed; entries from footage that can't be shared live in a git-ignored `tests/golden/private/`.

## Consequences
+ The pipeline, prompts and measurements can be read, run and challenged on their own; JSON output is easy to
  evaluate or extend.
+ Every behaviour in the prompts has a recorded reason, so regressions are caught by numbers.
− No bounding boxes or tracking; seconds per frame on Apple Silicon.
− A third party can rebuild the breadth tiers from public datasets, but not every curated golden entry.
