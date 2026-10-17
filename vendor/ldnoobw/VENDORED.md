# Vendored LDNOOBW word lists — provenance

Source: `LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words`
(https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words),
commit `5faf2ba42d7b`, downloaded 2026-07-30. Originally Shutterstock's own
autocomplete/recommendation filter list, open-sourced.

License: CC-BY-4.0 (confirmed via the GitHub repo's own license API field).
Attribution: LDNOOBW / Shutterstock, per the license terms.

Files here are the unmodified per-language plain-text lists (one word/
phrase per line), pulled directly from the repo's `en`, `zh`, `pt`, `es`
files — the languages this project currently uses or has planned (English
is the source-language reference for `profanity.py`'s hallucination
check; Spanish/Chinese/Portuguese are the translation targets), all from
commit `5faf2ba42d7b`. A target language needs its list here: without one,
`contains_profanity` fails open (returns False) and the always-on
hallucination check is silently a no-op for that language. Pull additional
per-language files from the same repo (75 languages available) if/when a
new target language needs one — no other project code changes needed,
`profanity.py` discovers whatever's on disk here.

**Adding words to an existing language** (as was done for Chinese's
`混蛋`, below), two ways:
1. Edit `vendor/ldnoobw/<lang>.txt` directly (one term per line).
2. Add to `profanity.py`'s `_EXTRA_TERMS` dict — better for a small,
   reviewed supplement to the upstream list.

WHY THIS EXISTS: found 2026-07-30 building the Chinese Opus-MT hybrid —
a translation-layer hallucination produced `你这个混蛋` ("you bastard")
from the English input "bucket", something no <unk>/corruption-shape
check could catch since it's clean, well-formed (if wrong) text. See
docs/findings.md and `profanity.py`'s own docstring for
the detection design (source-vs-output diff, not a blanket ban — a
tag/description that's ACCURATELY reporting profanity actually visible in
the source media is a separate, user-configurable concern, not a
hallucination).

CAVEAT (list's own README): "what goes in these lists is subjective... in
the end we just have to make our best guess." Not exhaustive — e.g. this
project's own found case, standalone `混蛋`, is NOT in the vendored `zh.txt`
(only the compound `王八蛋`/`他妈的王八蛋` are) — supplemented in
`profanity.py`'s `_EXTRA_TERMS` with cases found empirically by this
project, the same "grow the list from real evidence" pattern already used
for `tag_filter.PERSON_NOUNS`.
