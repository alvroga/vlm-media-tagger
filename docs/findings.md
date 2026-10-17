# Findings

What building and measuring this pipeline taught, condensed. Each item was found on real footage or a
benchmark set, and most were confirmed by changing one thing and re-measuring. Dates are when it was found.
The harness that produced the numbers is in `tests/golden/`.

## Model and inference

- **MiniCPM-V 4.5 over 2.6 and qwen3-vl** (2026-07-10). On the golden set: 73% vs 60% mean `nice_to_have`
  hit rate, 15–4 by entries, and better descriptions. 4.5 is ~1.5× slower per frame, because it writes more.
- **Temperature 0.2.** At 0 the model refuses to tag vehicles outdoors; at 0.5 the variance is wild.
  Probes at 0 and 0.4 were both worse than 0.2 for 4.5.
- **`think=False` on every call** (2026-07-10). Ollama 0.31 flipped MiniCPM-V 4.x into thinking by default:
  the whole `num_predict` budget went to hidden reasoning, content came back empty and calls were ~7× slower.
  This silently invalidated an afternoon of model comparisons. In descriptions, `</think>` debris and
  runaway sentence loops leaked into the text. `config.vlm_extra_kwargs()` is the single place the flag is
  set. Rule: after any Ollama upgrade, run `harness.py --canary` before trusting a measurement.
- **Verify a model can actually see.** One MLX build declared `vision` capability but had no working image
  projector. Check a model on a known image before measuring it.
- **Concurrency doesn't help** (2026-07-15). Serial: 104 s. 2, 4, 6 or 8 parallel requests: ~154 s,
  flat. Ollama serializes these requests, so the bottleneck isn't memory bandwidth. `PARALLEL_WORKERS=1`.
- **Repeat passes are cheap.** 3× the detection calls cost ~1.55× the wall time: the model is warm and the
  image context is cached, so later passes are mostly generation. Two passes per shot are the default.
- **720p is the resolution sweet spot** (2026-07-08, a 23-shot ad). 360p: 15 classes in 27 s.
  720p: 20 classes in 72 s. 1080p: 20 classes in 79 s, and the difference from 720p is run variance.
  Frames are capped at 1280 px, and RAW proxies are rendered at 720p for the same reason.

## Prompting a small VLM

- **Example values in a JSON format get copied** (2026-07-08). An example list `["car", "bow tie",
  "neon sign", "fence"]` showed up as tags on unrelated frames (a sci-fi hologram with no car). Swapping
  the example words made the hallucination follow them. The fix has three parts: obviously-placeholder
  text (`"example item one"`), an explicit "don't copy these" instruction, and a filter that drops any
  leaked placeholder afterwards.
- **Any scaffolding gets echoed.** "Sentence 1: … Sentence 2: …" came back as literal labels. An
  audience phrase ("for a … library") came back as "The main subject of this … library is…". Structure has to be written as prose, with no audience framing; `clean_description`
  strips label echoes.
- **Category coverage matters.** Without natural elements (water, rock, tree) and blur/bokeh handling
  named in the prompt, nature shots and out-of-focus close-ups returned empty lists.
- **Person nouns have to be banned at word level, not by exact match** (2026-07-10). The model emits "a
  man", "person's arm" and "out-of-focus person's face", which pass an exact-match blocklist.
  Colour-stripping can even manufacture a bare person noun ("red hand" → "hand"). `tag_filter` drops any
  tag that contains a person word. Some false drops ("hand truck") are accepted as the cost.
- **Confidence-gated naming beats a ban** (2026-07-11). A blanket "never name places or brands" rested on
  a single fabricated "Boston" that a full re-test didn't reproduce. It was replaced with "name a
  landmark, skyline or logo only when certain". The re-test was 10 landmark entries (including 2
  deliberately unidentifiable traps, Toledo and Akron) plus 26 LogoDet-3K logos, checked against a
  29-brand false-guess list. Across 21 + 54 adversarial generations there were zero wrong guesses.
  Apple's text-free silhouette was named correctly, and a Chinese corporate logo was OCR'd and translated
  instead of being mis-guessed as a Western brand. At scale (300 Places365 images, 450 LogoDet-3K logos)
  the false-guess rates were 12.3% (places) and 8.7% (brands) for 4.5.
- **Naming a structure is easier than naming its city.** Frames where the Brooklyn Bridge and the Empire
  State Building are named reliably still never produce "new york".
- **Negation trips word matchers.** On pure noise, the model correctly says it sees "no landmarks,
  buildings, person…", and a word-boundary matcher counts that as a mention. Seen twice; pruning the word
  from that entry's `must_not` was the fix both times. A negation-aware matcher is the better fix if it
  happens a third time.
- **The 'night' tag can't be prompted.** An explicit day/night instruction false-tagged golden-hour
  footage. Scene attributes like this belong in a deterministic post-step: a luminance check
  (`detect_objects_vlm.apply_deterministic_tags`).

## Translation (tags and descriptions)

- **Batch size, not wording, decides tag-translation reliability** (2026-07-21). With 20 tags per call,
  46% of calls returned the wrong number of items, silently mis-aligning translations with their tags.
  At 10: 34.5%. At 5: 19.2%. At 1: 5.4%.
- **An LLM can't reliably translate isolated words.** Five framings were all tried and all failed on
  confusable vocabulary (`cello` → `violín`): native generation, batched, one tag per call,
  WordNet-validated, and context-primed. A dedicated NMT model (Helsinki-NLP OPUS-MT via CTranslate2,
  CPU-only) never invented a non-word. Its worst case is leaving a word untranslated, which is the right
  failure for output that gets searched. Flowing prose doesn't show this failure; generating short lists
  in another language does.
- **A round-trip back-translation score is noisy enough to flip a comparison.** 221 of 397 tags got
  identical translations from the two models yet were scored differently across judge runs. Diff the
  candidates' actual outputs before trusting an aggregate.
- **`opus-mt-tc-big` models can't emit `ü`, accented capitals or ~100 symbols** (2026-07-23). They were
  trained with `character_coverage=0.9995` and no byte fallback, so `pingüino` → `ping⁇ino`. The base
  models (`character_coverage=1.0`) never emit `<unk>`. The fix is a hybrid: tc-big first per segment,
  falling back to base on any `<unk>`. The two models' weaknesses don't overlap: ~93% of sentences stay
  on tc-big. Descriptions must be sentence-split before NMT, because whole paragraphs truncate.
- **Exporting a tc-big model to CTranslate2 can loop forever.** The exported config can wrongly say
  `add_source_eos: false`; appending `</s>` to the source fixes it.
- **Brand names that are also English words get translated literally** (`apple logo` → `logo de
  manzana`), so tags ending in `logo` pass through untranslated.
- **NMT can hallucinate profanity** (2026-07-30): "bucket" → Chinese "你这个混蛋" ("you bastard"). The text
  is clean and confident, so no corruption check catches it. `opus_translate` rejects any output that
  contains a profane term its English source didn't have. This check is always on and unrelated to
  `--filter-profanity`.

## Media decoding

- **Probe by decoding, and do it in a subprocess.** ARRIRAW and X-OCN `.mxf` files open fine, then
  segfault libav on decode (exit 139). A child that segfaults also summons the macOS crash reporter, so
  the child installs libc `_exit` as a C-level signal handler: it exits cleanly with code == signal. A
  Python `signal.signal` handler can't do this, because it never runs after a C segfault.
- **Probing one frame doesn't prove the whole file decodes.** Atomos ProRes RAW `.mov` decodes one frame
  in PyAV, then PySceneDetect's OpenCV backend returns NaN timestamps. Every full-file decode stage
  converts its failure to `UndecodableMediaError` so the file reaches the Resolve fallback. VLM inference
  deliberately doesn't, so an Ollama outage is never misreported as bad media.
- **PySceneDetect reports frame 0 as the first cut**; strip it before building shot boundaries.
- **The cache must follow the source.** A proxy and its original share a working dir, so cached cuts and
  frames are invalidated when `cuts.json`'s `source` differs from the file being processed, or when step 1
  re-ran.

## DaVinci Resolve Studio (RAW fallback)

- **`-nogui` renders; `-rr` never does.** A scripted render under `-rr` sits in a "waiting for remote
  controller" queue forever.
- **Launch once per batch.** Startup is ~24 s, and `transcode()` launches and quits Resolve on every call.
  The batch path holds one connection and calls `_render_clip` per file.
- **Never script a GUI session.** `CreateProject`/`SetCurrentTimeline` would switch whatever the editor
  has open, so a running GUI Resolve is a hard refusal.
- **Track who owns a headless instance.** A PID marker records which process must `Quit()` it, so a second
  process reuses the instance without quitting it out from under the first. An instance that is
  pre-existing and never answers is a wedged orphan: kill it (only if `-nogui`) and relaunch once.
- **R3D comes in numbered chunks**; only `_001.R3D` is handed to Resolve, which plays the rest as one clip.
