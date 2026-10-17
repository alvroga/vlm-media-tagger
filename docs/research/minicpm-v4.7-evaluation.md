# MiniCPM-V 4.7 35B-A3B evaluation (2026-10-15 → 2026-10-16)

Status: **adopted** as the default engine ([ADR-0002](../adr/0002-minicpm-v47-on-llama-server.md)), with the
prompt set `PROMPT_SETS["minicpm-v4.7"]` in `src/prompts.py` (variant V8 + D10 below). The numbers here come from
a study rig (llama.cpp built from source + a small Ollama-API shim so the unmodified harness could run against
it); the one-off study scripts are not part of this repo. Re-certification on the shipped engine is pending.

## Facts

- Released 2026-10-12. 35B-A3B mixture-of-experts (the first MoE in the line), plus a 1B. Inputs: image,
  multi-image, video (128 sampled frames by default). Output: text only. **No audio input** (that is MiniCPM-o).
  The model card states no language coverage or translation quality.
- The 4.7 1B and the earlier 4.6 (1.3B) are too weak for this pipeline; 4.6 failed the golden set.
- **Runtime:** Ollama 0.32.1 and 0.40.4 both refuse it (`unknown projector type: minicpmv4_7`; their bundled
  llama.cpp predates it). llama.cpp master (b76b133, Metal) runs it. Other runtimes that do: MLX quants, vLLM,
  SGLang.
- **Memory:** Q4_K_M is 21.2 GB of weights; `llama-server` RSS ~23.4 GB at 8K context. Activity Monitor's GPU tab
  shows only ~2.5 GB, because mmapped weights count as file-backed memory, not process-owned. It doesn't fit a
  24 GB Mac; 32 GB is the floor and 48 GB is comfortable.
- **Speed** (8 golden frames, detection prompt, thinking off, temperature 0.2): 4.7 1.8 s/frame vs 4.5
  3.1–3.2 s. 4.5 is the same on both Ollama versions, so the gap is the model, not the runtime. 4.7's answers are
  longer (~68 vs ~37–54 tokens).
- The harness sends `think=False` only when the model name contains `minicpm-v4`; a test model name must keep
  that substring.
- **Scoring artifact:** `landmark-golden-gate` "violates" its `must_not` `bay bridge` when a model emits the
  separate generic tags `bridge` + `bay`, because the matcher is word-level. 4.7 correctly says
  `golden gate bridge`. Same class as the negation artifact in `docs/findings.md`.

## Golden harness, production 4.5 prompts on 4.7 (2026-10-15)

76/77 entries pass; the one failure is the `bay bridge` artifact above. On the 29 entries shared with an older
4.5 baseline: tag hit rate 78% → 87%, description hit rate 97% → 90%, 4.9 → 3.0 s per frame-run.
- 4.7 gains night, city and brand reads.
- Its poster descriptions favour visible text over pictured content. Variant V1 ("describe what is pictured
  first, quote only the most prominent text") fixes that (a film-poster entry 50% → 100%) without moving the
  controls.
- It calls a stapler a "paper cutter" under every wording; asked directly "what object is this", it answers the
  brand. That is a recognition limit, not a prompt problem.

## Tag translation: 4.7 vs the OPUS-MT hybrid (2026-10-15, not native-reviewed)

- Arm A: English tags, translated by 4.7 one tag per call. Arm C: the shipped OPUS-MT hybrid.
- 396 real tags per language. Identical outputs: es 59%, zh 32%, pt 59%.
- Eyeballed (not a native verdict): 4.7 is clearly better on Chinese, where OPUS-MT produced stapler → 锁定器
  ("locking device"), tableware → 表格软件 ("spreadsheet software") and tennis ball → 网球队 ("tennis team"); 4.7
  gave 订书机, 餐具, 网球. It is better on Portuguese, mixed on Spanish.
- 4.7 leaves more tags untranslated (es 17.4% vs 11.6%, cognates included): the safe failure mode.
- Cost: ~0.27 s/tag on 4.7 vs ~0.04 s/tag for CPU OPUS-MT.
- Arm B, native generation (tags asked for directly in the language, 77 frames): tag counts vs English per frame
  es 1.11×, zh 0.93×, pt 1.61×. Synonym drift was small (es car: only `coche`), far less than 4.5's
  `coche/carro/automóvil/vehículo`. **But** native output contains person tags the English ban can't see
  (`persona`, `mujer`, `pessoa`, `mulher`: ~11 on 77 frames), so native generation would need per-language
  person-noun lists.
- Conclusion: OPUS-MT stays. A 4.7 translation step is worth revisiting with native-speaker review.

## Prompt study (2026-10-15)

Variants were built from the 4.5 prompts by exact clause swaps and weighted toward video (~80% of a typical footage library),
so video results come first.
- **Baselines** (V0 = 4.5 prompts on 4.7, vs 4.5): COCO recall 81.9% vs ~78%; false-place tier 10.3% vs 12.3%;
  false-brand tier 5.3% vs 8.7%. Both naming tiers count some genuinely correct identifications as violations
  (Places365 contains real Hong Kong, Chicago and Burj Khalifa shots), so read them before trusting the rate.
- **Golden video entries (21):** every variant 21/21, zero empty runs, zero violations, nice-to-have
  93.6–94.4%. The prompt changes cost video nothing.
- **Real-video panel** (3-minute excerpts of 8 real videos, not scored): variants that keep names in the summary
  keep on-screen names (logos, signage) without naming real people. Tag churn vs V0 (Jaccard ~0.55) is
  close to the same-prompt run-to-run floor (~0.61–0.70). Every variant, V0 included, tags a distant floating
  pelican as dolphin, whale or turtle: a model trait.
- **Why 4.7 lost the names.** Asked directly, 4.7 answers Barcelona, Pittsburgh, Minneapolis and "the DeLorean
  time machine". Asked directly, it also confidently names the trap cities (Toledo → "Worcester", Akron →
  "Cincinnati"). The "only when genuinely certain" gate is what keeps it calibrated, and 4.7 reads it far more
  strictly than 4.5. Rewording the gate in place changed nothing (cities 0/3).
- **D4 (name first, gated on recognition)** fixed it: positives 6/21 → 17/21 (Barcelona, Pittsburgh,
  Minneapolis 3/3, DeLorean 2/3) with zero traps. On a 48-image Places365 slice, famous names went 12 → 17; of
  the 5 new, 4 were correct and 1 was wrong (an ornate Gothic rooftop called the Chrysler Building).
- **Empty tags on a text-dense screenshot:** a long headline overflowed the token budget, the JSON was truncated
  and parsed to `[]`. Fixed in code by the truncated-JSON salvage in `_parse_vlm_objects`, not by a brevity
  clause alone.

## Long-form still descriptions (D7, 2026-10-16)

D4 naming + 4–8 sentences + quote the title, tagline and credits. On a film poster: 260 words covering the
costume, the car, the gull-wing door, the full tagline and credits, and the logos. On the Pittsburgh skyline it
named the U.S. Steel Tower and PPG Place. Traps stayed clean, and it invented no plot facts that aren't printed on
the poster.

## Fictional characters vs real people (2026-10-16)

8 character images (film posters, public-domain illustrations, a 1931 film still) × 2 runs, and 19 real-person
traps: an official portrait, interview frames and the golden people entries. "Face-ID" means a
real person named from their appearance; a name read off a printed logo is legitimate OCR and not counted.

| Prompt | face-ID /19 | characters /16 |
|---|---|---|
| 4.5 production | **4** | **11** |
| 4.5 long-form D7 | 3 | 11 |
| 4.7 production | 3 | 6 |
| 4.7 long-form D7 | 5 | 6 |
| 4.7 D8 (+ "do not name actors, celebrities, public figures") | 1 | 7 |
| 4.7 D9 (characters anywhere, never the actor) | 4 | 8 |
| 4.7 **D10** (D8 + "even when you recognize a real person… never by name unless printed") | **0** | 6 |

- The 4.5 prompts had no real-person clause and do name real people from their faces.
- 4.7 won't name a film's lead character on its poster under any wording (it does when asked directly); since the
  poster shows a real actor, the real-person clause arguably applies. Frankenstein's Monster is named as the
  character, never as the actor.
- Long-form output fits `num_predict=512` (260 words, sentence-complete), but text-dense posters come close, so
  the still-description budget was raised.

## V8 certification (2026-10-15, with the salvage parser)

| Tier | 4.5 | 4.7 V0 | 4.7 V8 |
|---|---|---|---|
| Full harness (77): pass / nice / desc | 77/77, 75.5%, 89.5% | 76/77, 83.1%, 87.0% | 76/77, 84.2%, **92.2%** |
| Empty tag runs | — | 5 | **0** |
| Landmark tier "violations" | 12.3% | 10.3% | 17.0% |
| Logodet tier "violations" | 8.7% | 5.3% | 6.2% |
| COCO recall (all / prominent) | ~78% / 82% | 81.9% / 84.8% | 81.6% / 84.5% |

- Both 4.7 failures are the `bay bridge` artifact. V8 names Barcelona, Pittsburgh, Minneapolis and Paris 3/3 in
  descriptions (V0 0/5), and a multi-shot car clip's summary names the brand 100% (V0 40%).
- V8's losses: an "aerial" description probe (it says "high-angle"; a matcher artifact), Eiffel's `paris` moved
  from tags to the description, and a screenshot headline is no longer quoted (D1's intended trade).
- **The landmark tier's +20 "violations" were hand-checked**, all new: 15 correct (Watts Towers, Burj Khalifa,
  Hong Kong ×2, Charles Bridge, Tokyo Metropolitan Government Building, Grand Palace, San Francisco, a ballpark
  with its URL on the scoreboard, Manhattan, the 9/11 Memorial, Amsterdam Centraal, the Rockefeller tree, Temple
  of Heaven, Summer Palace); 3 right city, wrong landmark (Süleymaniye → "Hagia Sophia", St James's Park →
  "Serpentine", Angkor ruins → "Angkor Wat"); 2 wrong (the Gothic rooftop → "Chrysler Building", Beijing's CBD →
  "Seoul"). Net: ~0.7% of hard architecture images get a wrong city.
- **Logodet +3, hand-checked:** all correct (Coco Pops → Kellogg's, Raisinets → Nestlé, a Del Monte word match).

## Why native video input isn't used yet

Through Ollama it wasn't available: the chat API takes a list of still images. Earlier multi-frame tests showed 4
frames in one request work but dozens of shots don't, hence per-frame captions plus a text combine step
(`summarize.py`). Both are runtime and budget limits, not bans. 4.7 on llama.cpp accepts real video, so it is
now testable.

## Open questions, in priority order

1. **Full-scope tag translation test** (es/zh/pt): 4.7 translate vs native generation vs OPUS-MT, gated on
   native-speaker review. Lessons to apply: diff actual outputs before trusting aggregates, translate one tag
   per call, measure synonym drift and tag-count inflation, keep the hallucinated-profanity guard.
2. **Native video per shot:** keep the shot cuts and send each shot as one multi-frame request (4–16 downscaled
   frames). Merge per-shot tags and descriptions, then build the summary as text in stages (shots → scenes →
   video). Never send a whole long video in one call. Long continuous shots need ~30 s windows. Cache per-shot
   results so a retry doesn't redo shots. Measure against the current path on description quality, speed and
   false positives.
3. **Multi-image reasoning:** one description from several frames of a shot.
4. **OCR / on-screen text** (titles, credits, lower-thirds, signage) as searchable metadata.

Noted, not planned: bounding boxes; scene attributes (shot type, camera motion, lighting); technical-quality
flags (black frames, focus, exposure), which need every frame analysed and so a fast deterministic pass rather
than the VLM; slate OCR → structured metadata; duplicate-shot grouping. Speech needs a separate ASR model.
