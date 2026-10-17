# Naming policy validation: brands and locations

**Date:** 2026-07-12
**Studies:** LogoDet-3K (brands, 470 images) + Places365 (locations, 300 images)
**Model:** openbmb/minicpm-v4.5
**Sources:** `tests/golden/results/logodet-2026-07-11-openbmb_minicpm-v4.5.json`,
`tests/golden/results/landmark-2026-07-12-openbmb_minicpm-v4.5.json` (full
per-image audit under `"audit"` in each file)

## Headline

Both studies test the same failure mode: does the confidence-gated naming
policy (adopted 2026-07-11: name a brand or place only when genuinely
certain) ever invent a specific, famous name for something genuinely obscure
or unidentifiable — the "sees a triangle, says a famous car brand" risk. An automated trap-word script flagged 61 cases across 770
images combined. Manually reviewing every flagged image against its actual
source file told a different story:

| Study | Sample | Raw script flag rate | Confirmed after manual audit |
|---|---|---|---|
| Brands (LogoDet-3K) | 470 obscure logos | 8.7% (39/450 new logos) | **0%** |
| Locations (Places365) | 300 landmark photos | 12.3% (37 matches / 22 images flagged; script mislabels this — accurately, 7.3% of images had ≥1 flag) | **0.3%** (1/300) |

The gap between raw and audited numbers isn't the model being lucky — it's
what a crude trap-word list can and can't tell apart, and it repeats in the
same shape across two very different domains (commercial logos vs.
architecture). That repetition is the more durable finding; see
[Why the raw numbers overcounted](#why-the-raw-numbers-overcounted).

## Brands: LogoDet-3K, 470 obscure logos, zero confirmed hallucinations

450 newly sampled obscure logos plus 18 from an earlier hand-picked round,
each run through the real stills pipeline (two-pass detection +
one description call) and checked against a 50-name "famous brand" trap
list. The raw script flagged 39 cases; none held up under review.

What the flags actually were:

- **Correct-but-unanticipated recognition** — the trap list didn't
  anticipate the model would correctly recognize Converse, Louis Vuitton,
  BP, or an iPod inside "obscure" sample images.
- **Genuinely multi-logo images** — a real PGA Tour pin for the "Funai
  Classic at Walt Disney World Resort" correctly surfaced "Disney" (it's
  printed on the pin); a meme graphic with Costa, Starbucks, and Heineken
  logos was read correctly, in position, for all three — "Costa Coffee…
  Starbucks Coffee logo displaying its mermaid symbol… reads HEINEKEN BEER
  PREMIUM QUALITY."
- **Validation-script false positives** — short generic trap words
  ("new", "apple") substring-matching unrelated text ("apple cider" on a
  cider bottle, "New!" marketing copy).

## Locations: Places365, 300 landmark photos, one confirmed miss

300 contemporary photographs across 16 outdoor architecture/landmark scene
categories (bridges, palaces, skyscrapers, stadiums, temples, towers, etc.),
sampled from Places365's crowdsourced validation set — which carries no
"hide the famous ones" curation. Same pipeline path, checked against a
city/named-landmark trap list.

Because the sample draws from real, uncurated web photography, most flags
turned out to be the model correctly naming a real landmark that genuinely
was in frame, several with detail specific enough to rule out coincidence:

- Named "Bank of China Tower" specifically inside a Hong Kong skyline
- Read "TORONTO" literally spelled out in stadium seat coloring
- Named "Hudson River"/"Manhattan" correctly on an aerial NYC shot
- **Recognized Burj Khalifa mid-construction, from silhouette alone** — no
  finished spire, no skyline context, just the distinctive tiered
  Y-shaped setback pattern under scaffolding
- **Hedged appropriately, and was right anyway** — one description named
  the Woolworth Building but added "though not definitively identified
  here" rather than stating it flatly. Visually confirmed correct.

**The one confirmed miss:** "Angkor Wat" named on an image that's actually
the Royal Palace in Phnom Penh — same country, same distinctive gold Khmer
roofline, wrong specific building. A narrow miss, not an unrelated guess.
Out of 300 images, this is the entire confirmed-miss column.

> The one real mistake in 300 images picked the wrong building in the right
> country, in the same architectural family. The failure mode we were
> testing for — a confident, unrelated guess — didn't show up once.

## Open question at the time: naming a real person

One flagged image (the Taj Mahal, correctly identified) also had a
description that confidently named the person standing in frame as a
well-known actor. It was plausibly correct, but nothing here can verify that
at scale.

This isn't a defect in either trap list — it's a gap in scope. The
confidence-gated naming policy governs places and brands. It never covered
naming a specific real person in description text, a different risk
category: an accuracy mistake there is a misattribution, not a missed tag.

**Resolved later:** descriptions never name a real person from their
appearance, only from printed text (a caption, a credit). The prompt clause
and its measurements (face-ID 4/19 on the 4.5 prompts → 0/19) are in
[the 4.7 evaluation](minicpm-v4.7-evaluation.md#fictional-characters-vs-real-people-2026-10-16).

## Why the raw numbers overcounted

Both studies used a simple trap-word list — flag any tag or description
mentioning a famous name that shouldn't be there — and both overcounted for
the same two reasons:

1. **Correct-but-unanticipated recognition.** An "obscure" sample still
   contains real, nameable things. The trap list reflects what we assumed
   the model would get wrong going in, not what's actually in the sample.
2. **Substring collisions on generic words.** A bare tag like `tower` or
   `new` word-matches inside a longer trap phrase like `eiffel tower` or
   `new balance` without the model ever having claimed the specific
   landmark or brand. Both scoring scripts had this bug (found and fixed
   2026-07-12, `logodet_tier.py` and `landmark_tier.py`); it's the majority
   contributor to the raw 12.3% on the locations side.

**Practical rule this sets:** a violation count from either bulk tier is a
worklist for manual review, never a number to report on its own. Every
number in this report is post-audit.

## References

- `tests/golden/logodet_tier.py`, `tests/golden/landmark_tier.py` — scoring scripts
- `tests/golden/logodet/`, `tests/golden/landmark/` — frozen samples
- `tests/golden/archival/` — the archival-photo sample that came out of the first landmark attempt
