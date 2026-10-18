# Licence ledger

Every third-party dependency, model and dataset this project uses, with its licence and how it is used.
New ones get a row before they are added.

## Code dependencies (`requirements.txt`, installed by pip, not vendored)
| Package | Licence | Use |
|---|---|---|
| ollama (python client) | MIT | VLM calls |
| av (PyAV) | BSD-3-Clause (wraps FFmpeg, LGPL/GPL builds) | decoding, frame extraction |
| scenedetect | BSD-3-Clause | shot-cut detection |
| opencv-python-headless | Apache-2.0 | scenedetect backend |
| pillow | MIT-CMU (HPND) | image decode/resize |
| imagehash | BSD-2-Clause | near-duplicate frame dedup |
| timecode | MIT | timecode formatting |
| numpy | BSD-3-Clause | |
| python-dotenv | BSD-3-Clause | `.env` config |
| ctranslate2 | MIT | NMT inference |
| sentencepiece | Apache-2.0 | NMT tokenization |
| pytest, ruff | MIT | dev only |
| langdetect (optional) | Apache-2.0 | i18n harness tier only; sentence-level language ID |

App only (`requirements-app.txt`):

| Package | Licence | Use |
|---|---|---|
| pywebview | BSD-3-Clause | the app window (WKWebView) and its JS bridge |
| briefcase | BSD-3-Clause | builds and signs the `.app` (build tool, not shipped inside it) |
| std-nslog | BSD-3-Clause | routes the built app's stdout to the macOS log |
| proxy_tools (pywebview dependency) | MIT | wheel built from the official sdist, `vendor/wheels/` |

## Vendored data (in the repo)
| Item | Licence | Source |
|---|---|---|
| App icon and logo (`icons/`, `src/vmt_app/resources/webui/logo.png`) | Apache-2.0, as the project | in-house artwork by the author |
| `vendor/ldnoobw/*.txt` word lists | CC-BY-4.0 (attribution: LDNOOBW / Shutterstock) | see `vendor/ldnoobw/VENDORED.md` |

## Models (downloaded by the user, never in the repo)
| Model | Licence | Notes |
|---|---|---|
| MiniCPM-V 4.7 35B-A3B GGUF (Q4_K_M + mmproj f16, `openbmb/MiniCPM-V-4.7-35B-A3B-gguf`) | MiniCPM Model License (check the HF model card) | fetched by `llama_manager.py`, sha256-pinned |
| MiniCPM-V 4.5 (`ollama pull openbmb/minicpm-v4.5`) | Apache-2.0 code; weights under the MiniCPM Model License | pulled by the user through Ollama |
| Helsinki-NLP `opus-mt-tc-big-en-*` | CC-BY-4.0 | fetched by `translation_models` from a public release; attribution in each archive and the manifest |
| Helsinki-NLP `opus-mt-en-*` | Apache-2.0 / CC-BY-4.0 | same |
| Argos OpenTech `en_zh` 1.9 (Chinese tc-big layer) | CC-BY-4.0 | same store; attribution in the archive and the manifest |

## Evaluation data (local only, git-ignored)
| Set | Licence | Use |
|---|---|---|
| Golden entries' frames | per source: Pexels licence (stock clips), Wikimedia per-file, COCO/LogoDet-3K terms | regression fixtures, not redistributed; provenance in each `expected.json` |
| Private golden entries (`tests/golden/private/`) | not redistributable | local-only fixtures, never committed |
| COCO val2017 sample | annotations CC-BY-4.0; images per-Flickr licence | recall tier |
| Places365 val sample | research / non-commercial (Places2 terms) | false place-guess tier |
| LogoDet-3K sample | research use | false brand-guess tier |
| Google Landmarks Dataset v2 `index` sample | CC0 / public domain (per the GLDv2 README) | archival-photo tier |
| Wikimedia Commons images (landmark entries) | per-file (CC-BY/CC-BY-SA/PD) | golden entries |
| Big Buck Bunny poster (`poster-bbb`) | CC-BY-3.0, Blender Foundation | golden entry |
| Pexels stock clips (bokeh, creek, skyline, night entries) | Pexels licence | golden entries; frames not redistributed |

## External tools (optional, not distributed)
| Tool | Notes |
|---|---|
| llama.cpp `llama-server` (MIT) | built locally by `scripts/build_llama_cpp.sh`; ships inside the .app with its licence file |
| DaVinci Resolve Studio | proprietary, user-installed; scripting API used for RAW proxies |
| macOS `sips` | system tool; decodes HEIC/PSD/camera RAW stills |
