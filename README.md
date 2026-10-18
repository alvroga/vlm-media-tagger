# vlm-media-tagger

Open-vocabulary tagging and natural-language description of **video and still images**, running fully
locally with a vision-language model: [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)
on a managed [llama.cpp](https://github.com/ggml-org/llama.cpp) server by default, or MiniCPM-V 4.5 via
[Ollama](https://ollama.com). Point it at a clip, a photo or a folder of footage; get JSON with searchable tags,
time-coded segments and a description. Your media never leaves your machine.

It is a research project into what a small local VLM can reliably detect and understand in real footage, and
what it takes to make that output trustworthy:
- **Measured, not guessed**: every prompt clause in `src/prompts.py` was kept or rejected on regression-harness
  evidence, and the reasons sit next to the clauses. [`docs/findings.md`](docs/findings.md) has what that taught;
  [`docs/research/`](docs/research/) has the model studies.
- **Usable**: a CLI today, and a free macOS app in progress ([ADR-0003](docs/adr/0003-macos-app-and-output-sinks.md))
  that writes a JSON sidecar next to each file, with an output-plugin interface so you can send the metadata
  anywhere.

## Why a VLM instead of a detector

A closed-set detector (COCO's 80 classes) can say `car`, `person`, `cup`. Searching real footage needs
`delorean`, `bokeh`, `volkswagen logo`, `lily pads`, `night skyline` and `wooden railing`. This pipeline asks
a VLM for every visible object, material, effect and readable brand. Then it does the engineering that
makes a small VLM's output trustworthy enough to index: shot-aware sampling, multi-pass detection,
deterministic cleanup, guarded naming, and a regression harness that scores every prompt change.

## Pipeline

```
video ─► decode probe (subprocess) ─► shot cuts (PySceneDetect) ─► frames @2fps, 1280px, pHash dedup
      │        └─ RAW (BRAW/R3D/ARRIRAW/X-OCN/ProRes RAW) ─► 720p proxy via DaVinci Resolve Studio
      ▼
  sharpest frame per shot (+ interior samples for long shots) ─► VLM tags ×2 passes
      ▼
  merge into per-shot segments ─► tag_filter (person-noun ban, colour/junk strip)
      ▼
  captions of ≤10 sampled shots ─► one combine call ─► summary
      ▼
  [profanity filter] ─► [OPUS-MT translation] ─► JSON

image ─► PIL / macOS sips (HEIC, PSD, camera RAW) ─► still prompt ─► tags + one description call
```

| Module | Role |
|---|---|
| `src/tagger.py` | CLI and per-file orchestration, Resolve fallback, JSON output |
| `src/sinks.py`, `src/library.py` | output sinks (JSON sidecar, JSONL, plugins: [`docs/plugins.md`](docs/plugins.md)) and the folder scanner that skips done files |
| `src/pipeline.py` | cuts → frames → detection, with source-aware caching |
| `src/detect_objects_vlm.py` | frame selection, VLM calls, output parsing, segment merge |
| `src/summarize.py` | still descriptions and hierarchical video summaries |
| `src/prompts.py` | the tuned prompts **and the measured reasoning behind each clause** |
| `src/tag_filter.py` | deterministic tag cleanup |
| `src/media_probe.py` | crash-safe "can this be decoded?" probe |
| `src/resolve_proxy.py` | headless DaVinci Resolve Studio proxy renderer (from [resolve-proxy-nogui](https://github.com/alvroga/resolve-proxy-nogui)) |
| `src/opus_translate.py`, `src/translation_models/` | tc-big → base OPUS-MT hybrid, models fetched on demand |

## The app

A macOS app over the same pipeline. Add folders (scanned with all their subfolders), single files, or folders on
a network share; pick tags and/or descriptions; press Analyze. Each file gets its JSON sidecar
(`clip.mp4` → `clip.mp4.json`), and the next run skips everything already done, so a big library can be
processed in as many sittings as you like, or on a schedule (Settings → Automation).

- **Network shares (SMB/AFP/NFS):** connect with Connect to Server (macOS's own dialog; the password stays in
  your Keychain) and add a folder on the share. The app remembers the share, reconnects it at launch and can
  unmount it on quit.
- **First launch** downloads the AI model once (22 GB, or links it from a local Ollama store), then loads it.
  Needs an Apple Silicon Mac with 32 GB of memory or more.
- **RAW camera files** (BRAW, R3D, ARRIRAW, X-OCN) work when DaVinci Resolve Studio is installed and enabled in
  Settings → Integrations.
- **Send results elsewhere:** add an output plugin ([`docs/plugins.md`](docs/plugins.md)).

**Installing a downloaded build:** the app is signed but not notarized, so macOS blocks the first launch. Open
it once, then go to System Settings → Privacy & Security, click **Open Anyway** next to VLM Media Tagger, and
confirm. After that it opens normally.

Run it from a checkout (after the Quick start's engine steps):

```bash
./venv/bin/pip install -r requirements-app.txt
cd src && ../venv/bin/python -m vmt_app
../venv/bin/python ../scripts/app_smoke.py   # optional: end-to-end check that drives the real window
```

## Quick start (CLI)

```bash
python3.13 -m venv venv && ./venv/bin/pip install -r requirements.txt
scripts/build_llama_cpp.sh                     # pinned llama.cpp → vendor/llama_cpp/llama-server (needs cmake)
./venv/bin/python src/llama_manager.py fetch   # 22.3 GB from Hugging Face (0 if Ollama already has this GGUF)
./venv/bin/python src/llama_manager.py start & # MiniCPM-V 4.7 on 127.0.0.1:11471 (~23 GB RAM; 48 GB+ Mac)
./venv/bin/python src/doctor.py                # checks the engine, and Resolve (optional)
# 24 GB Mac: VLM_BACKEND=ollama with `ollama pull openbmb/minicpm-v4.5` instead

./venv/bin/python src/tagger.py clip.mp4                        # JSON to stdout, progress to stderr
./venv/bin/python src/tagger.py footage/ -r -o results.json     # a folder, recursively
./venv/bin/python src/tagger.py photo.heic --lang es            # Spanish output (downloads ~300 MB once)
./venv/bin/python src/tagger.py clip.braw                       # RAW: uses DaVinci Resolve Studio if installed
./venv/bin/python src/tagger.py footage/ -r --sidecar --skip-done # clip.mp4 → clip.mp4.json; reruns skip done files
```

Options: `--model`, `--no-description`, `--filter-profanity`, `--no-resolve`, `--force` (redo cached
cuts/frames), `--keep-workdir`. Configuration can also come from the environment or a `.env` file
(`OLLAMA_HOST`, `OLLAMA_VISION_MODEL`, `DATA_DIR`, …; see `src/config.py`). Exit code 2 means at
least one file didn't process; its record says why.

## Output

```json
{
  "tool": "vlm-media-tagger", "version": "0.1.0", "model": "openbmb/minicpm-v4.5", "language": "en",
  "files": [
    {
      "path": "A001_C001.braw", "type": "video", "status": "ok", "source": "resolve-proxy",
      "duration_sec": "…", "shot_count": 1,
      "tags": ["bridge deck", "cap", "fog", "lily pads", "palm fronds", "shoes", "shorts",
               "signboard", "trees", "water", "wooden railing"],
      "description": "…",
      "segments": [{"tag": "water", "start_sec": 0.0, "end_sec": 0.0, "frame_count": 1}],
      "elapsed_sec": 49.3
    }
  ]
}
```

`status` is one of:
- `ok`
- `redundant-chunk` (a non-first R3D chunk)
- `undecodable`
- `resolve-unavailable` (RAW without Resolve Studio)
- `image-undecodable`
- `pipeline-failed`

With `--lang`, the record also carries `tags_en` and `description_en`.

## Measuring it

`tests/golden/` is a prompt-regression harness. Each entry is a frozen set of frames guarding one failure
mode that actually happened, scored over N runs:
- `must_have` / `must_not` are hard pass/fail.
- `nice_to_have` and description mentions are hit rates, judged as movement against a saved baseline.

There are breadth tiers too:
- COCO val2017 recall
- Places365 false place-guess rate
- LogoDet-3K false brand-guess rate
- multilingual output (native generation vs translation)

```bash
./venv/bin/python tests/golden/harness.py --canary           # ~1 min environment gate (run first)
./venv/bin/python tests/golden/harness.py --quick
./venv/bin/python tests/golden/harness.py --passes 2 --descriptions --save \
    --baseline tests/golden/results/2026-07-20-openbmb_minicpm-v4.5-p2.json
./venv/bin/python tests/golden/coco_tier.py --save
```

**The images aren't redistributed.** They come from Pexels stock clips, Wikimedia Commons, COCO, Places365 and
LogoDet-3K under their own licences, and they are git-ignored. The expectations (with each entry's source), the
sampling scripts (`*/freeze_sample.py`, fixed seeds) and the baseline results are committed, so the breadth tiers
and most golden entries can be rebuilt from public sources. See [`tests/golden/README.md`](tests/golden/README.md).

Unit tests (no model needed; the VLM and Resolve are mocked): `tools/test.sh`. Lint: `tools/lint.sh`.

## Findings

[`docs/findings.md`](docs/findings.md) has the condensed lessons with their numbers. They include:
- JSON format examples leak into a small VLM's answers.
- Thinking mode silently ate the token budget after an Ollama upgrade.
- 720p beats 1080p for this pipeline.
- Concurrency makes Ollama slower.
- An NMT model beats the VLM itself at translating short tags.
- Confidence-gated naming of places and brands holds a low false-guess rate.

[`docs/research/`](docs/research/) has the longer studies:
- [MiniCPM-V 4.7 evaluation](docs/research/minicpm-v4.7-evaluation.md): 4.7 35B-A3B vs 4.5 on speed, tag
  recall, naming calibration, long-form descriptions, and fictional characters vs real-person face ID.
- [Naming policy validation](docs/research/naming-policy-validation.md): does confidence-gated naming
  invent famous brands and places? (770 images, audited by hand.)
- [Scene-level summaries](docs/research/scene-summary-feasibility.md): embedding-segmented scene lines
  for fast-cut video.
- OPUS-MT reviews for [Chinese](docs/research/opusmt-chinese-review.md) and
  [Portuguese](docs/research/opusmt-portuguese-review.md): what real content found that a synthetic corpus missed.
- [Apple Vision and Foundation Models](docs/research/apple-vision-framework.md) as a replacement or
  supplement (COCO benchmark).

## Limits

- The model is slow: seconds per frame. Detection takes ~1 call per shot ×2 passes, plus ≤10
  caption calls for the summary.
- It gives no bounding boxes and no tracking.
- Names are intentionally cautious: a missed name is cheaper than a wrong one in an index.
- Small or distant figures and very dark scenes are weak spots. Audio isn't analysed.

## Design decisions

Architecture decisions are recorded in [`docs/adr/`](docs/adr/). How to run everything: [`docs/how-to-run.md`](docs/how-to-run.md).

## Licence

Apache-2.0 ([`LICENSE`](LICENSE)): use, modify and redistribute freely, including commercially. Redistributions
must keep the [`NOTICE`](NOTICE) file (the attribution) and mark changed files. Third-party data and models keep
their own licences; see [`docs/licenses.md`](docs/licenses.md).
