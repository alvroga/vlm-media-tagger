# Brazilian Portuguese OPUS-MT: findings and native-review brief

**Date:** 2026-07-30
**Status:** Portuguese now ships in `src/translation_models/manifest.json` (tc-big + base hybrid, the
same design as Spanish and Chinese). This brief records the engineering findings from before it shipped and
the questions only a native Brazilian Portuguese reader can answer; that review is still open.

## What differed from Spanish and Chinese

Portuguese was the best-behaved of the three, and it needed one extra step: a **base-model bake-off**.
tc-big converts directly from Hugging Face, but there's no dedicated bilingual `opus-mt-en-pt` there for the
base (safety-net) role. The first candidate, `opus-mt-en-ROMANCE` (one transformer for ~40 Romance
varieties), was compared against a Portuguese-dedicated model that exists only as a raw Tatoeba-Challenge
release (`opus-2021-02-18`, plain transformer). **The dedicated model won**: it fixed a live bug (the
`saco`/"bag" note below) and matched or beat `en-ROMANCE` on every real case tested. It is the base that
ships.

Across the 396-tag corpus (re-run with the shipped base), the hybrid picked **tc-big 323 times (82%)**, fell
back to **base 4 times (1%)**, **stayed English 39 times (10%)** (untranslatable short words on both layers,
or both rejected), and **30 times (8%)** were brand `*logo` tags that skip translation by design. So, as with
Chinese, **the quality bar is almost entirely tc-big's word choice**; the base is rarely load-bearing, and no
base-side `<unk>` gaps turned up in probing.

**Both layers need a target-selector prefix token.** tc-big is the Tatoeba-Challenge `eng-por` release
(targets `pob`/`por`: Brazilian vs European), and the shipped base comes from the same `eng-por` family, so
both use `>>pob<<`. **Confirmed empirically: `>>pob<<` gives consistently Brazilian vocabulary on both
layers** (ônibus/celular/trem/garota, not autocarro/telemóvel/comboio/rapariga). While `en-ROMANCE` was
still a candidate, **`>>pt_PT<<` on it was actively broken**, mixing in Spanish- and Italian-looking
fragments (`"L'autobus arrivò tarda"`, `"La girl está tocando nel cortillo"`). Same shape as Chinese's bare
`>>cmn<<` bug.

## The corpus

The same 396 real tags and 6 real English descriptions the Spanish and Chinese reviews used: a movie poster,
an animated-film poster, a screenshot, market signage, a product shot and a nature photo, chosen because
quote-, brand- and spec-heavy text is where translation problems hide. The tags and descriptions come from
golden-set runs, some of them private entries, so the raw corpus isn't committed.

**Also checked against a real footage library:** a 153-file metadata export from a real video library,
translated end to end, read-only. Result: clean, with zero `⁇`/replacement-character hits across all 153
files, and none of Chinese's subtitle-code or boilerplate-hallucination defects. One real finding: `burro`
(the correct word for "donkey", also Brazilian slang for "stupid") is on the vendored profanity list, like
`saco`/"bag" below. This time it discarded an **entire description**: tc-big and base both produced the
correct Portuguese sentence, the guardrail rejected both, and the cascade fell back to English. That is a
pattern, not a one-off: Portuguese has many ordinary animal and object words with a vulgar secondary sense
(`saco`, `burro`, and already-listed `vaca`/cow, `corno`/horn), which real footage will hit far more often
than genuinely hallucinated profanity.

## Questions for a native reviewer

1. **Do the tag translations read as correct, natural Brazilian Portuguese?** Flag anything wrong, awkward,
   or European rather than Brazilian.
2. **Is any translation wrong in a way that would look bad in a published tag?** See the known-bad tail.
3. **Should the profanity guardrail get a per-language carve-out** for ordinary words the vendored list
   over-flags (`burro`)? That's a policy call, not something to patch silently. (`saco` was resolved as a
   side effect: the shipped base translates "bag" → `bolsa`.)
4. Do any of the 39 stayed-English tags look like they should have translated?

## Known-bad tail (re-verified against the shipped base)

The same isolated-word ceiling Spanish (`cello` → `violín`) and Chinese (`coelacanth` → `长颈鹿`) hit: a
short tag, out of context, sometimes gets a confidently wrong answer instead of a near miss.

| English tag | tc-big (ships) | base (fallback) | Note |
|---|---|---|---|
| `chrysler building` | `construção de crisler` (misspelled, generic "construction") | `Edifício Chrysler` (correct) | base keeps the proper noun, tc-big mangles it |
| `bus icon` | `ícone de barramento` (a *data* bus) | `ícone de ônibus` (correct) | tc-big picked the wrong word sense |
| `coupons sign` | `Cupons assinam` ("sign" as a verb) | `Os cupons assinam` (same error) | both layers wrong the same way |
| `circle` | `círculo` (correct) | `Círculo.` (correct) | `en-ROMANCE`'s missing-accent `circulo` is gone with the new base |
| `bag` | *(stays English)* | `Bolsa.` → `bolsa` | **fixed by the base swap**: with `en-ROMANCE` both layers produced `saco`, the guardrail rejected it and the tag stayed English |

None of these are in the class of Chinese's two worst findings (a fabricated business-registration number, a
real product name swapped in for "mug handle"). Portuguese's tail reads as ordinary NMT word-sense errors,
not the model latching onto unrelated real-world text. It deserves another look on a larger real corpus: the
396-tag corpus missed both of Chinese's worst defects entirely.

## Fixed during this investigation (generic, not gated on the review)

- **Trailing sentence punctuation in tags.** Both models sometimes treat a bare noun as a one-word sentence
  and append a period (`airplane` → `avião.`, 22/396 tags). It was 0/396 on Spanish, so it was genuinely new
  here. Fixed generically in `opus_translate.translate_tags`: strip one trailing `.`/`!`/`?`, since a tag is
  never a sentence in any language. Verified as a no-op on the Spanish corpus.
- **End-of-sentence token needed on both layers** (as with Chinese; the opposite of Spanish). Without the
  manual `</s>` append, both layers show runaway repetition (`"pôr do sol sobre o oceano oceano sobre o
  oceano"`, `"pele de pele de pele."`). Wired through each component's `meta.json`, no code change.
- **tc-big character-coverage gap, same as Spanish:** accented capitals (`Ú` in `PÚBLICA` → `P⁇BLICA`) and
  symbols (`€ ± © ® × ÷ •`) come out as `<unk>`. The base handles all of them, so the hybrid falls back
  correctly.
