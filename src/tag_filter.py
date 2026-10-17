"""
Light cleanup of VLM object nouns before they become output tags.

- Pure color/junk tags are dropped ("black", "white", "text", "white text").
- Color prefixes are stripped and the result deduped ("red shirt" → "shirt",
  "black and red scarf" → "scarf").
- Any tag containing a person noun as a word is dropped whole ("a man",
  "person's arm", "hand" left over after color-stripping). The prompt asks
  the model not to mention people, and _call_vlm drops exact-match person
  nouns, but multi-word phrasings slipped through both (caught by the
  golden-set harness 2026-07-10 on real footage) — this is the last line
  of defense before a tag reaches the output.
- Everything else passes through untouched.
"""

# People/body parts are NEVER tagged (project hard rule). Canonical set —
# detect_objects_vlm imports this for its own exact-match early drop.
PERSON_NOUNS = {
    "man", "woman", "boy", "girl", "person", "people", "crowd",
    "individual", "passenger", "pedestrian", "customer", "staff",
    "worker", "employee", "host", "guest", "child", "men", "women",
    "guy", "lady", "male", "female", "human",
    "hand", "hands", "face", "faces", "arm", "arms", "body", "bodies",
    "leg", "legs", "finger", "fingers", "head", "hair", "eye", "eyes",
}

_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "black",
    "white", "grey", "gray", "brown", "beige", "tan", "gold", "golden",
    "silver", "maroon", "navy", "teal", "turquoise", "cream", "violet",
    "dark", "light", "colorful", "colored",
}

_JUNK = {
    "text", "word", "words", "letter", "letters", "symbol", "symbols",
    "number", "numbers", "question mark", "exclamation mark", "line",
    "lines", "shape", "shapes", "background", "foreground", "pattern",
    "color", "colors", "object", "objects", "item", "items", "thing",
    "things", "stuff", "scene", "image", "photo", "picture frame edge",
    "this", "that", "it", "summer", "winter", "spring", "autumn", "fall",
    "decorative items", "decorative item", "original film", "title", "logo text",
    # meta-echoes of the stills prompt's category phrases (the model
    # sometimes writes the category instead of the actual visible words)
    "brand name", "brand", "company name", "product name", "title text",
    "readable text", "visible text",
}

_CONNECTORS = {"and", "with", "-", "&"}

# Language/script names — legitimate as part of a real transcription
# ("chaoyang road", "missha"), but "<language> text/characters/signs/
# banners" is the same lazy-tag pattern as "text overlay"/"hd watermark":
# a generic placeholder that carries zero search value on its own,
# emitted when a dense or unfamiliar-script scene got a weaker read than
# a clean one (found 2026-07-11 building the multilingual golden
# entries: Beijing's two well-separated signs got every street name
# transcribed correctly, but Seoul's dense overlapping signage sometimes
# produced bare "korean text banners"/"korean text signs" tags instead).
_LANGUAGES = {
    "chinese", "japanese", "korean", "arabic", "russian", "cyrillic",
    "foreign", "asian",
}


def clean_tag(tag: str) -> str | None:
    """Return the cleaned tag, or None if it should be dropped."""
    tag = tag.lower().strip()
    if not tag or tag in _JUNK:
        return None
    # Overlay/watermark debris — exhaustive prompts (2026-07-10) surface
    # broadcast/stock-footage furniture as tags: "text overlay",
    # "hd watermark", "text 1974", "flight information text". Generic text
    # mentions are junk; actual readable brand/title names survive because
    # they arrive as the words themselves ("fedex", "aperol spritz"), not
    # as "<something> text".
    if "watermark" in tag or "overlay" in tag:
        return None
    if tag.startswith("text ") or tag.endswith(" text"):
        return None
    words = tag.split()
    # "<language> text/characters/signs/banners/script" — a lazy
    # placeholder, not a real transcription; "japanese flag"/"korean bbq"
    # must survive, so this only fires when EVERY remaining word is
    # itself a generic text-ish noun, not just any noun.
    _TEXTY = {"text", "characters", "character", "signs", "sign",
              "banners", "banner", "script", "writing", "letters", "letter",
              "words", "word"}
    if words and words[0] in _LANGUAGES and words[1:] and \
            all(w in _TEXTY for w in words[1:]):
        return None
    # person guard: check each word with possessives normalized ("person's"
    # → "person"); one person word poisons the whole tag ("man aiming
    # rifle" is a person description, not an object)
    if any(w.rstrip(".,!?").replace("’", "'").removesuffix("'s")
           in PERSON_NOUNS for w in words):
        return None
    kept = [w for w in words if w not in _COLORS and w not in _CONNECTORS]
    if not kept:
        return None  # pure color ("black", "red and white")
    cleaned = " ".join(kept)
    if cleaned in _JUNK:
        return None  # "white text" → "text"
    return cleaned


def clean_tags(tags: list[str]) -> list[str]:
    """Clean a tag list: drop junk, strip colors, dedup preserving order."""
    seen = set()
    out = []
    for t in tags:
        c = clean_tag(t)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out
