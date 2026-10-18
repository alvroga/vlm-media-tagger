"""
Profanity detection, for two DIFFERENT callers with two different rules:

1. opus_translate.py — ALWAYS ON, no setting. Catches a TRANSLATION
   HALLUCINATION (opus_translate introducing profanity that wasn't in the
   English source). Found 2026-07-30: a Chinese translation of "bucket"
   produced "你这个混蛋" ("you bastard") — clean, well-formed text, so
   neither <unk> detection nor the subtitle-code corruption check caught
   it. Always wrong, regardless of (2) below — a hallucination isn't real
   content. See docs/findings.md (Translation).
2. tagger.py — gated behind config.FILTER_PROFANITY (--filter-profanity,
   off by default). Profanity genuinely present in the source media (a
   sign, dialogue, a shirt) is real content, not a defect — the
   default keeps it ("more relevant tags > fewer"), but some uses
   want it filtered. `filter_tags`/`redact` implement the
   ACTUAL filtering for whichever caller wants it; `contains_profanity`
   alone is what (1) uses to detect the hallucination case.

Word lists: vendor/ldnoobw/{lang}.txt (CC-BY-4.0, see VENDORED.md) plus a
small supplement of terms found empirically that the upstream list misses.
"""
import re

import config

# Vendored LDNOOBW lists: checkout = vendor/ldnoobw/, built app =
# Resources/app/ldnoobw/ (briefcase lands vendor dirs flat under app/). A
# missing dir just means _terms() returns empty sets (fails open, see
# contains_profanity).
_VENDOR_DIR = next((d for d in (config.PROJECT_ROOT / "vendor" / "ldnoobw",
                                config.PROJECT_ROOT / "app" / "ldnoobw") if d.is_dir()),
                   config.PROJECT_ROOT / "vendor" / "ldnoobw")

# Found empirically 2026-07-30 (opus-mt hallucination: "bucket" -> 你这个混蛋)
# — not present in the upstream zh.txt (only compound 王八蛋 forms are). Same
# "grow the list from real evidence" pattern as tag_filter.PERSON_NOUNS.
_EXTRA_TERMS = {
    "zh": ["混蛋"],
}

# Languages with no whitespace word boundaries (CJK) need substring
# matching; space-delimited languages use word-level matching to avoid
# short-profane-term-inside-an-innocent-word false positives (the
# "Scunthorpe problem").
_NO_WORD_BOUNDARY_LANGS = {"zh", "ja", "ko"}

_cache: dict = {}


def _terms(lang: str) -> set:
    if lang not in _cache:
        path = _VENDOR_DIR / f"{lang}.txt"
        words = set()
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    w = line.strip().lower()
                    if w:
                        words.add(w)
            except OSError:
                pass
        words.update(w.lower() for w in _EXTRA_TERMS.get(lang, []))
        _cache[lang] = words
    return _cache[lang]


def contains_profanity(text: str, lang: str) -> bool:
    """True if `text` contains a vendored profane term for `lang`. No word
    list installed for `lang` -> always False (fails open, same posture as
    the rest of this project's optional per-language assets)."""
    if not text:
        return False
    terms = _terms(lang)
    if not terms:
        return False
    lowered = text.lower()
    if lang in _NO_WORD_BOUNDARY_LANGS:
        return any(term in lowered for term in terms)
    words = re.findall(r"[\w'-]+", lowered)
    wordset = set(words)
    return any(t in wordset or (" " in t and t in lowered) for t in terms)


def filter_tags(tags: list, lang: str) -> list:
    """Drop any tag containing a profane term as a whole word (or substring
    for CJK) — same "one bad word poisons the whole tag" rule as
    tag_filter.clean_tag's person-noun guard. Only called when
    config.FILTER_PROFANITY is on (see tagger.py); contains_profanity's
    hallucination-rejection use in opus_translate.py is unrelated and
    unconditional."""
    return [t for t in tags if not contains_profanity(t, lang)]


def redact(text: str, lang: str) -> str:
    """Replace each profane term found in `text` with asterisks of the same
    length, preserving surrounding sentence structure (a description is
    free-flowing prose — dropping the whole sentence over one word would
    lose real content the filter isn't meant to touch). Longest terms
    first so a multi-word match isn't pre-empted by a shorter one inside
    it. Only called when config.FILTER_PROFANITY is on."""
    if not text:
        return text
    terms = sorted(_terms(lang), key=len, reverse=True)
    out = text
    for term in terms:
        if lang in _NO_WORD_BOUNDARY_LANGS:
            pattern = re.escape(term)
        else:
            pattern = r"\b" + re.escape(term) + r"\b"
        out = re.sub(pattern, lambda m: "*" * len(m.group(0)), out, flags=re.IGNORECASE)
    return out
