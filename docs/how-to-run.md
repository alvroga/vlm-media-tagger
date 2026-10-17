# How to run

Requires macOS on Apple Silicon, Python 3.13, and an engine: the managed llama-server with MiniCPM-V 4.7
(default: `scripts/build_llama_cpp.sh`, `src/llama_manager.py fetch`, `src/llama_manager.py start`; ~23 GB RAM)
or `VLM_BACKEND=ollama` with `openbmb/minicpm-v4.5` pulled. `LLAMA_HOST=http://…` uses a server you run yourself.
DaVinci Resolve Studio is optional (RAW camera formats only).

```bash
python3.13 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python src/doctor.py                 # environment check
./venv/bin/python src/tagger.py <files|dirs> [-r] [-o out.json] [--lang es] [--no-description]
```

| What | Command |
|---|---|
| Lint | `tools/lint.sh` |
| Unit tests (no model) | `tools/test.sh` |
| Environment gate before any measurement | `./venv/bin/python tests/golden/harness.py --canary` |
| Prompt regression, quick | `./venv/bin/python tests/golden/harness.py --quick` |
| Prompt regression, full | `./venv/bin/python tests/golden/harness.py --passes 2 --descriptions --save --baseline tests/golden/results/2026-07-20-openbmb_minicpm-v4.5-p2.json` |
| Breadth tiers | `tests/golden/{coco,landmark,logodet}_tier.py --save` |
| Translation models | `PYTHONPATH=src ./venv/bin/python -m translation_models status es` |
| Single steps | `src/detect_cuts.py`, `src/extract_frames.py`, `src/detect_objects_vlm.py <video_id>`, `src/summarize.py <video_id>`, `src/resolve_proxy.py` |

Working files land in `data/<stem>-<pathhash>/` and are deleted after each file unless `--keep-workdir`.
The harness and tiers need the fixture images, which are git-ignored (see README "Measuring it").
