# Golden set — prompt regression harness

A small, frozen set of images with human-verified expectations. Any prompt
or model change is judged by hit-rate movement against this set, not by
eyeballing. See docs/findings.md for the failure modes the entries guard.

## Layout

```
tests/golden/
  entries/<name>/
    frames/*.jpg      # the FROZEN inputs (local only, see "Fixture images")
    frames/index.json # video-full only: portable frame index (see below)
    cuts.json         # video-full only: full real shot-boundary data
    expected.json     # the human-verified expectations
  private/entries/    # entries whose source media can't be shared (gitignored)
  private/results/    # their share of each saved run (gitignored)
  _incoming/          # drop zone for new source material (gitignored)
  _sources/           # staging for raw sources being processed (gitignored)
```

## Other tiers

Bulk samples scored in aggregate, never hand-curated. Each `*/freeze_sample.py`
rebuilds its frozen sample from the public dataset (fixed seed); the images are
git-ignored. A violation count from a naming tier is a worklist for manual
review, never a number to report on its own
([why](../../docs/research/naming-policy-validation.md#why-the-raw-numbers-overcounted)).

| Tier | Measures | Source |
|---|---|---|
| `coco_tier.py` | object recall | COCO val2017, 200 images |
| `landmark_tier.py` | false place/landmark guesses | Places365 val, 300 images |
| `logodet_tier.py` | false brand guesses | LogoDet-3K, 450 logos |
| `i18n_tier.py` | non-English output: native generation vs English-then-translate, language consistency and content fidelity | a sample of the public entries above |
| `archival/` | old photography (B&W, postcards, engravings); sample only, no runner yet | Google Landmarks v2 `index` split, 300 images |

`i18n_tier.py --dry-run` lists its combos without calling the model. `i18n_lang_id.py` is its pure-text
language detector; `langdetect` (optional) improves sentence-level Spanish detection.

## Fixture images

The frame JPEGs are not redistributed (`.gitignore`): `expected.json` records
each entry's provenance so the frames can be rebuilt from the original source
(Pexels stock clips, Wikimedia Commons, COCO, LogoDet-3K).
Entries built from footage that can't be shared at all live under `private/entries/` with the same layout.
The harness runs both sets; `--save` writes the public entries to `results/`
and the private ones to `private/results/` under the same filename, and a
`--baseline` is merged back from both, so a committed baseline never carries
private output.

## Rules

- **Frames are frozen, not derived.** Video entries freeze the exact JPEGs
  (pipeline-style: 1280 px max dimension). The harness must test PROMPTS
  in isolation — if cut detection or frame selection changes, golden
  results must not move. Source videos stay OUT of the repo;
  `expected.json` records provenance. **`video-full` entries are the
  deliberate exception** — see their own section below.
- **Every entry is a sentinel for a specific failure mode** we actually
  hit (see docs/findings.md). Don't add generic "nice pictures".
- Keep entries small (≈3–8 frames) — the harness runs every frame N times.

## expected.json

```json
{
  "type": "video-frames | still | video-full",
  "source": "where the frames came from (path/URL, date frozen)",
  "sentinel_for": "which failure mode this entry guards",
  "prompt": "video | still",
  "must_have": ["tags whose absence in ANY run is a hard regression"],
  "nice_to_have": ["tags scored as hit-rate % across N runs (confidence-edge)"],
  "must_not": ["tags whose PRESENCE in any run is a hard fail"],
  "description_must_mention": ["(stills, --descriptions runs only) things a
                                correct DESCRIPTION must say — rate-scored"],
  "summary_must_mention": ["(video-full only, --descriptions runs only) same
                           idea as description_must_mention but scored
                           against summarize_video()'s output"],
  "min_segment_counts": {"class-name": "video-full only: hard per-run floor "
                         "on discrete post-merge segments for that class"},
  "notes": "anything a future curator needs"
}
```

Matching is substring-tolerant both ways ("coral reef" satisfies "coral")
after `tag_filter` runs — expectations describe post-filter output, i.e.
what would actually be written to the output JSON. An expectation may list
alternatives with `|` ("rabbit|bunny") — any one of them satisfies it.

Global `must_not` rules apply to EVERY entry (the harness hardcodes
them): person nouns / body parts, prompt-placeholder echoes
("example item one"…), and junk nouns tag_filter must drop ("text",
"background"). Entry-level `must_not` adds entry-specific fails, e.g.
fabricated city names; it applies to description text too, not just tags.

`must_not` encodes the hallucination/junk gotchas: person nouns, prompt
placeholder echoes ("example item one"), junk nouns that tag_filter must
drop ("text", "background"), and per-entry forbidden fabrications
(e.g. a specific city name on an anonymous street scene).

## Description scoring (added 2026-07-11)

With `--descriptions`, still entries also get one DESCRIPTION call per run
(the real pipeline path: `IMAGE_CAPTION_PROMPT` through
`summarize.clean_description`), and each `description_must_mention` item is
scored as a hit-rate across runs — word-boundary, plural-tolerant,
`|`-alternatives, same as tags but matched inside the description text.
Rates are movement-vs-baseline signals, never pass/fail; `must_not` DOES
hard-fail on description text (fabricated city names etc.). Person words
are fine in descriptions — only tags ban person nouns. Description texts
are saved in results files for eyeball review and future recalibration.

## video-full entries (added 2026-07-12)

Every other entry type freezes a hand-picked handful of frames specifically
to isolate the PROMPT from cut detection and frame selection. `video-full`
entries do the opposite on purpose: they freeze a WHOLE video's `cuts.json`
+ every extracted frame (`entries/<name>/cuts.json` +
`entries/<name>/frames/*.jpg` + a portable `frames/index.json` — no baked-in
absolute paths, so the entry works from any checkout), and the harness runs
the REAL `detect_objects_vlm.detect_objects_vlm()` and
`summarize.summarize_video()` **unmodified** against that frozen data (via a
scratch `config.DATA_DIR` swap to a temp dir per run — the frozen frames are
symlinked in, never copied or written to). This is the only place two things
get scored at all:

- **Shot merging** — `min_segment_counts` is a hard per-run floor on how
  many DISCRETE post-merge segments a class must produce. This exists
  because `_merge_segments`'s whole job is to NOT collapse a recurring
  object across many real shots into one giant segment (a 30 s ad showing
  9 different cars must not merge them into "car: 1 segment") — a
  hand-picked frame subset can't exercise real shot boundaries, so this is
  the only test that can catch that specific regression.
- **Video summaries** — `summary_must_mention` (needs `--descriptions`,
  same rate-scoring as `description_must_mention`) scores
  `summarize_video()`'s actual output — length now scales with duration
  (`_target_summary_length`, added 2026-07-18) rather than a flat "2-3
  sentences" — which until this entry type existed was completely
  unmeasured.

Because these run the real pipeline (not a `--passes`-parameterized
re-implementation), `DETECT_PASSES` is whatever `detect_objects_vlm.py`
currently hardcodes — `--passes` is accepted but ignored for these entries.
They're also meaningfully more expensive per entry than a still/video-frames
entry (a full shot list × `DETECT_PASSES` detection calls, plus up to
`MAX_SUMMARY_FRAMES` caption calls + one combine call per run with
`--descriptions`) — factor that in before adding many of these.

## Canary (added 2026-07-11)

`harness.py --canary` is the pre-measurement gate: one reliable entry
(poster-bbb — 1 frame, hard title-text floor, covers detection + description),
one run against the newest saved full-set baseline for the active model,
after a warm-up call. It hard-fails (exit ≠ 0) on must_have misses,
violations, EMPTY outputs (the thinking-mode contamination signature), or
per-call time >2.5× baseline. Run it before ANY measurement session and
after ANY Ollama upgrade — it exists because a silent Ollama default-mode
flip once invalidated an afternoon of model comparisons (see docs/findings.md).
