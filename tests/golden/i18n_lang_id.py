"""Language-identification helpers for the multilingual-output tier
(i18n_tier.py). Pure text processing, no VLM/network calls — this is the
piece that turns an eyeballed "9 of 42 tags stayed English" into a repeatable
percentage.

Two different techniques on purpose, because the two target languages need
different treatment:
  - Chinese: CJK Unicode-range counting is exact and reliable even on a
    single short word — no ambiguity, no library needed.
  - Spanish: same Latin alphabet as English, so script-range checks can't
    tell them apart, and `langdetect` — the obvious off-the-shelf choice —
    turned out USELESS at tag length: caught empirically (2026-07-12)
    misclassifying "torre eiffel" as Norwegian, "árbol" as Hungarian,
    "cielo azul claro" as Romanian, all with high confidence. It's a
    statistical n-gram classifier tuned for paragraphs, not 1-3 word
    phrases, and short text simply doesn't carry enough signal for it.
    Verified separately that it DOES work correctly on full sentences
    (descriptions/summaries) — so `is_spanish` routes by length: real
    langdetect for sentence-length text, a small curated word-list +
    Spanish-diacritic heuristic for tag-length text, returning None
    (not a guess) when neither signal is conclusive. None results are a
    manual-review worklist, same discipline as the logodet/landmark
    tiers' flagged violations — never silently counted as pass or fail.
"""
import re

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0  # deterministic — langdetect is otherwise randomized
    _HAVE_LANGDETECT = True
except ImportError:
    _HAVE_LANGDETECT = False

# Word count at/above which langdetect is trusted directly. Below this,
# short-text heuristics take over (see is_spanish). Not tuned rigorously —
# a reasonable-looking cutoff; revisit if manual audits show it's wrong.
_SENTENCE_LEVEL_WORDS = 5

# Spanish-specific characters — any occurrence is a strong, unambiguous
# positive signal (English never uses these).
_SPANISH_MARKERS_RE = re.compile(r"[áéíóúñü¿¡]", re.IGNORECASE)

# Deliberately narrow: common English words likely to appear in VLM tag
# output (visual/object vocabulary) plus common function words — NOT a
# general dictionary. A short Spanish/English-ambiguous word not in this
# list (e.g. a real word this list doesn't know) correctly falls through
# to "unknown" rather than being force-classified — expand this list from
# manual-audit findings, the same way tag_filter's junk-word list grew.
_COMMON_ENGLISH_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "with", "near",
    "far", "no", "not", "yes", "text", "sign", "signage", "background",
    "watermark", "car", "cars", "house", "houses", "tree", "trees", "sky",
    "building", "buildings", "road", "street", "water", "window",
    "windows", "door", "doors", "fence", "light", "lights", "wall",
    "walls", "grass", "cloud", "clouds", "mountain", "mountains", "sun",
    "dog", "cat", "bridge", "tower", "boat", "plane", "airplane", "train",
    "bike", "bicycle", "chair", "table", "plant", "plants", "flower",
    "flowers", "rock", "rocks", "sand", "wave", "waves", "hill", "field",
    "park", "tire", "tires", "wheel", "wheels", "mirror", "license",
    "plate", "headlight", "headlights", "jacket", "shirt", "dress", "coat",
    "hat", "watch", "bag", "curtain", "screen", "clock", "roof",
    "driveway", "bush", "bushes", "white", "black", "red", "blue", "green",
    "yellow", "gray", "grey", "brown", "small", "large", "old", "new",
    "open", "closed", "night", "day", "sunlight", "shadow", "shadows",
}

# Mirror list for Spanish — common object/visual vocabulary WITHOUT
# diacritics (words that DO have one, e.g. "árbol"/"matrícula", are
# already caught by _SPANISH_MARKERS_RE and don't need to be here). Added
# 2026-07-12 after the diacritic-only check left common accent-free
# Spanish words like "cielo"/"coche"/"puerta" stuck in the unknown bucket
# — same "expand from manual-audit findings" discipline as the English
# list above, not an attempt at a complete dictionary.
_COMMON_SPANISH_WORDS = {
    "el", "la", "los", "las", "un", "una", "de", "en", "con", "sin", "y",
    "o", "no", "si", "cielo", "azul", "claro", "puerta", "blanca",
    "blanco", "coche", "torre", "casa", "agua", "roca", "rocas", "calle",
    "carretera", "edificio", "edificios", "ventana", "ventanas", "valla",
    "cerca", "luz", "luces", "sol", "nube", "nubes", "montana", "montanas",
    "perro", "gato", "puente", "barco", "avion", "tren", "bicicleta",
    "silla", "mesa", "planta", "plantas", "flor", "flores", "arena", "ola",
    "olas", "colina", "campo", "parque", "neumatico", "rueda", "espejo",
    "chaqueta", "camisa", "vestido", "abrigo", "sombrero", "reloj",
    "bolso", "cortina", "pantalla", "techo", "entrada", "arbusto",
    "arbustos", "negro", "rojo", "verde", "amarillo", "gris", "marron",
    "pequeno", "grande", "viejo", "nuevo", "abierto", "cerrado", "noche",
    "dia", "sombra", "manzana", "logo",
}

_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*(am|pm)?$", re.IGNORECASE)
_DATE_ABBREV_RE = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}$", re.IGNORECASE)


def is_verbatim_exempt(text: str) -> bool:
    """True for content that should never be scored for language
    consistency at all — URLs, clock times, short date abbreviations —
    these have no "translation," correctly staying as-is is the right
    answer, not a language-compliance failure. Automatic and general
    (safe, low false-positive risk). Entry-specific proper nouns (brand
    names, bylines, movie titles) can't be caught by a general pattern —
    those need a curated allowlist instead, see
    i18n_tier.TRANSLATED_EXPECTATIONS's "expected_untranslated". Found
    2026-07-12 auditing the first real i18n_tier run: raw "English stray"
    counts on poster-bbb and a news-site screenshot were dominated by exactly this
    kind of content (a URL, bylines, timestamps) — the same "raw
    automated numbers overcount" lesson as the logodet/landmark tiers,
    recurring in a new context."""
    t = text.strip()
    return bool(_URL_RE.search(t) or _TIME_RE.match(t) or _DATE_ABBREV_RE.match(t))


_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x3000, 0x303F),    # CJK punctuation
    (0xFF00, 0xFFEF),    # fullwidth forms (｜、etc.)
)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_fraction(text: str) -> float:
    """Fraction of CJK characters among all letter-like (CJK or Latin)
    characters — 0.0 if there's no letter content at all (empty/numeric)."""
    cjk = sum(1 for ch in text if _is_cjk_char(ch))
    latin = len(_LATIN_LETTER_RE.findall(text))
    total = cjk + latin
    return cjk / total if total else 0.0


def is_chinese(text: str, threshold: float = 0.5) -> bool:
    """True if CJK characters dominate the string. Exact even for a single
    Chinese word — Unicode ranges don't need statistical language ID."""
    return cjk_fraction(text) >= threshold


def has_english_leak(text: str, threshold: float = 0.3) -> bool:
    """True if a meaningful fraction of a Chinese-target string is plain
    Latin letters (not just a units/measurement stray) — the signature of
    the "responded in mixed English and Chinese" failure mode."""
    return 0 < cjk_fraction(text) < (1 - threshold) or (
        cjk_fraction(text) == 0.0 and bool(_LATIN_LETTER_RE.search(text))
    )


def is_spanish(text: str) -> bool | None:
    """True = classified Spanish, False = classified English, None =
    genuinely indeterminate (do NOT force a guess — feed to manual
    review). Sentence-length text (>= _SENTENCE_LEVEL_WORDS) uses real
    langdetect, verified reliable there. Shorter (tag-length) text uses a
    Spanish-diacritic check first (unambiguous when it fires), then falls
    back to the curated common-English-word list (only classifies False
    when EVERY word is a known-common English word — a word this list
    doesn't recognize correctly falls through to None instead of a wrong
    guess)."""
    text = text.strip()
    if not text:
        return None

    words = text.split()
    if len(words) >= _SENTENCE_LEVEL_WORDS:
        if not _HAVE_LANGDETECT:
            return None
        try:
            return detect(text) == "es"
        except Exception:
            return None

    if _SPANISH_MARKERS_RE.search(text):
        return True

    normalized = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    normalized = [w for w in normalized if w]
    if not normalized:
        return None
    if all(w in _COMMON_SPANISH_WORDS for w in normalized):
        return True
    if all(w in _COMMON_ENGLISH_WORDS for w in normalized):
        return False

    return None


## ---------------------------------------------------------------------
## Extended 2026-07-12 for a screen of 10 more languages
## (Hindi, Arabic, French, Portuguese, Italian, Russian, Japanese,
## German, Korean, Indonesian — a spread of scripts and language families,
## the world's most-used languages minus near-duplicates). Two genuinely
## different confidence tiers, not
## glossed over:
##
##   DISTINCT-SCRIPT languages (hi/ar/ru/ko, plus zh already above):
##   Unicode-range detection is EXACT, same reliability as the existing
##   Chinese check — no ambiguity, no word lists needed. Japanese is the
##   one exception WITHIN this tier: it shares the CJK ideograph block
##   with Chinese (kanji), so kanji-only text is genuinely ambiguous
##   between the two — hiragana/katakana presence is the one confident,
##   Japan-specific signal; kanji alone falls to unknown rather than a
##   forced guess, same "don't guess" discipline as everywhere else here.
##
##   LATIN-SCRIPT languages (fr/pt/it/de/id): same problem class as
##   Spanish (langdetect useless at tag length, needs diacritics + a
##   curated word list) — but these word lists are FIRST-PASS, built
##   from basic vocabulary knowledge in one sitting, not grown from real
##   audit findings across multiple failure-analysis rounds the way
##   Spanish's was. Treat results for fr/pt/it/de/id as more preliminary
##   than es/zh/hi/ar/ru/ko's — smaller lists mean more items correctly
##   fall to "unknown" (not wrong, just less coverage) rather than being
##   confidently classified. Indonesian has no reliable diacritics at
##   all, so it leans entirely on the word list — expect the most
##   "unknown" results of the five.
## ---------------------------------------------------------------------

_SCRIPT_RANGES = {
    "zh": _CJK_RANGES,
    "hi": ((0x0900, 0x097F),),                          # Devanagari
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),  # Arabic
    "ru": ((0x0400, 0x04FF),),                           # Cyrillic
    "ko": ((0xAC00, 0xD7A3), (0x1100, 0x11FF)),          # Hangul
}
_JA_KANA_RANGES = ((0x3040, 0x309F), (0x30A0, 0x30FF))   # Hiragana, Katakana


def _script_fraction(text: str, ranges: tuple) -> float:
    script_chars = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in ranges))
    latin = len(_LATIN_LETTER_RE.findall(text))
    total = script_chars + latin
    return script_chars / total if total else 0.0


def is_script_language(text: str, lang_code: str, threshold: float = 0.5) -> bool:
    """Exact Unicode-range check for any distinct-script language in
    _SCRIPT_RANGES — same technique as is_chinese, generalized."""
    return _script_fraction(text, _SCRIPT_RANGES[lang_code]) >= threshold


def is_japanese(text: str) -> bool | None:
    """True only on hiragana/katakana presence — the one unambiguous
    Japan-specific signal. Kanji-only text is genuinely ambiguous with
    Chinese (shared Unicode block) and falls to None rather than a
    guess in either direction."""
    text = text.strip()
    if not text:
        return None
    kana = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _JA_KANA_RANGES))
    if kana > 0:
        return True
    if _script_fraction(text, _SCRIPT_RANGES["zh"]) > 0:
        return None  # kanji present, no kana — can't distinguish from Chinese
    return False if _LATIN_LETTER_RE.search(text) else None


# Diacritic sets — any occurrence is a strong, unambiguous positive
# signal, same role as _SPANISH_MARKERS_RE. Indonesian has none reliable
# enough to include (rests entirely on the word list below).
_LATIN_DIACRITICS = {
    "fr": re.compile(r"[àâäéèêëïîôùûüÿçœæ]", re.IGNORECASE),
    "pt": re.compile(r"[ãõáàâéêíóôúç]", re.IGNORECASE),
    "it": re.compile(r"[àèéìòóù]", re.IGNORECASE),
    "de": re.compile(r"[äöüß]", re.IGNORECASE),
}

# First-pass common-word lists (see module-level note above on maturity)
# — same semantic categories as _COMMON_ENGLISH_WORDS/_COMMON_SPANISH_WORDS
# (basic visual/object vocabulary + function words), one list per
# language, words without diacritics only (diacritic forms are already
# caught by _LATIN_DIACRITICS above and don't need to be duplicated here).
_COMMON_WORDS = {
    "fr": {
        "le", "la", "les", "un", "une", "de", "et", "ou", "non", "oui",
        "voiture", "tour", "maison", "eau", "roche", "roches", "rue",
        "route", "arbre", "arbres", "pont", "mur", "herbe", "nuage",
        "nuages", "montagne", "montagnes", "soleil", "chien", "chat",
        "bateau", "avion", "train", "velo", "bicyclette", "chaise",
        "table", "plante", "plantes", "sable", "vague", "vagues",
        "colline", "champ", "parc", "pneu", "pneus", "roue", "roues",
        "veste", "chemise", "robe", "manteau", "chapeau", "montre",
        "sac", "rideau", "ecran", "horloge", "toit", "allee", "buisson",
        "noir", "noire", "rouge", "vert", "verte", "jaune", "gris",
        "grise", "marron", "petit", "petite", "grand", "grande", "vieux",
        "vieille", "nouveau", "nouvelle", "ouvert", "ouverte", "ferme",
        "fermee", "nuit", "jour", "ombre",
    },
    "pt": {
        "o", "a", "os", "as", "um", "uma", "de", "e", "ou", "nao", "sim",
        "carro", "torre", "casa", "agua", "rocha", "rochas", "rua",
        "estrada", "arvore", "arvores", "ponte", "muro", "grama",
        "nuvem", "nuvens", "montanha", "montanhas", "sol", "cachorro",
        "gato", "barco", "aviao", "trem", "bicicleta", "cadeira", "mesa",
        "planta", "plantas", "areia", "onda", "ondas", "colina", "campo",
        "parque", "pneu", "pneus", "roda", "rodas", "espelho", "jaqueta",
        "camisa", "vestido", "casaco", "chapeu", "relogio", "bolsa",
        "cortina", "tela", "telhado", "entrada", "arbusto", "preto",
        "preta", "vermelho", "verde", "amarelo", "cinza", "pequeno",
        "pequena", "grande", "velho", "velha", "novo", "nova", "aberto",
        "aberta", "fechado", "fechada", "noite", "dia", "sombra",
    },
    "it": {
        "il", "la", "i", "le", "un", "una", "di", "e", "o", "no", "si",
        "macchina", "torre", "casa", "acqua", "roccia", "rocce",
        "strada", "via", "albero", "alberi", "ponte", "muro", "erba",
        "nuvola", "nuvole", "montagna", "montagne", "sole", "cane",
        "gatto", "barca", "aereo", "treno", "bicicletta", "sedia",
        "tavolo", "pianta", "piante", "sabbia", "onda", "onde",
        "collina", "campo", "parco", "pneumatico", "pneumatici", "ruota",
        "ruote", "specchio", "giacca", "camicia", "vestito", "cappotto",
        "cappello", "orologio", "borsa", "tenda", "schermo", "tetto",
        "ingresso", "cespuglio", "nero", "nera", "rosso", "rossa",
        "verde", "giallo", "grigio", "grigia", "marrone", "piccolo",
        "piccola", "grande", "vecchio", "vecchia", "nuovo", "nuova",
        "aperto", "aperta", "chiuso", "chiusa", "notte", "giorno",
        "ombra",
    },
    "de": {
        "der", "die", "das", "ein", "eine", "und", "oder", "nein", "ja",
        "auto", "turm", "haus", "wasser", "fels", "felsen", "strasse",
        "baum", "baume", "brucke", "mauer", "wand", "gras", "wolke",
        "wolken", "berg", "berge", "sonne", "hund", "katze", "boot",
        "flugzeug", "zug", "fahrrad", "stuhl", "tisch", "pflanze",
        "pflanzen", "blume", "blumen", "sand", "welle", "wellen",
        "hugel", "feld", "park", "reifen", "rad", "rader", "spiegel",
        "jacke", "hemd", "kleid", "mantel", "hut", "uhr", "tasche",
        "vorhang", "bildschirm", "dach", "einfahrt", "busch", "schwarz",
        "rot", "grun", "gelb", "grau", "klein", "gross", "alt", "neu",
        "offen", "geschlossen", "nacht", "tag", "schatten",
    },
    "id": {
        "yang", "dan", "atau", "tidak", "ya", "langit", "biru", "cerah",
        "pintu", "putih", "mobil", "menara", "rumah", "air", "batu",
        "jalan", "jendela", "pohon", "jembatan", "tembok", "rumput",
        "awan", "gunung", "matahari", "anjing", "kucing", "perahu",
        "pesawat", "kereta", "sepeda", "kursi", "meja", "tanaman",
        "bunga", "pasir", "ombak", "bukit", "ladang", "taman", "ban",
        "roda", "cermin", "jaket", "kemeja", "gaun", "mantel", "topi",
        "jam", "tas", "tirai", "layar", "atap", "semak", "hitam",
        "merah", "hijau", "kuning", "abu", "kecil", "besar", "tua",
        "baru", "buka", "tutup", "malam", "siang", "bayangan",
    },
}


def is_latin_language(text: str, lang_code: str) -> bool | None:
    """Generalized version of is_spanish for fr/pt/de/id — same routing
    (langdetect at sentence length, diacritic+wordlist at tag length),
    same "return None rather than guess" discipline."""
    text = text.strip()
    if not text:
        return None

    words = text.split()
    if len(words) >= _SENTENCE_LEVEL_WORDS:
        if not _HAVE_LANGDETECT:
            return None
        try:
            return detect(text) == lang_code
        except Exception:
            return None

    diacritics = _LATIN_DIACRITICS.get(lang_code)
    if diacritics and diacritics.search(text):
        return True

    normalized = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    normalized = [w for w in normalized if w]
    if not normalized:
        return None
    if all(w in _COMMON_WORDS[lang_code] for w in normalized):
        return True
    if all(w in _COMMON_ENGLISH_WORDS for w in normalized):
        return False
    return None


_SUPPORTED_LANGS = {"es", "zh", "hi", "ar", "ru", "ko", "ja", "fr", "pt", "it", "de", "id"}


def detect_in_language(text: str, lang_code: str) -> bool | None:
    """Single dispatcher for any of the 12 supported language codes —
    routes to the right detection strategy (exact Unicode range,
    Japanese's kana-special-case, or Latin diacritic+wordlist)."""
    if lang_code == "es":
        return is_spanish(text)
    if lang_code == "zh":
        return is_chinese(text)
    if lang_code == "ja":
        return is_japanese(text)
    if lang_code in _SCRIPT_RANGES:
        return is_script_language(text, lang_code)
    if lang_code in _COMMON_WORDS:
        return is_latin_language(text, lang_code)
    raise ValueError(f"unsupported lang_code: {lang_code!r}")


def language_consistency_rate(
    items: list[str], lang_code: str, exempt: set[str] | frozenset[str] = frozenset()
) -> tuple[float | None, list[str], list[str]]:
    """Fraction of `items` (tags or sentences) CONFIDENTLY classified as
    the target language, out of items with a confident classification
    either way (unknowns are excluded from the rate, not counted against
    it). Returns (rate, confident_misses, unknowns) — both lists are a
    manual-review worklist, same discipline as the logodet/landmark
    tiers' flagged violations: never trust the rate alone. lang_code:
    any of _SUPPORTED_LANGS.

    `exempt` (case-insensitive) removes items BEFORE classification
    entirely — general verbatim content (is_verbatim_exempt: URLs,
    times, dates) plus any caller-supplied entry-specific proper nouns
    (brand names, bylines, titles) that are correct to leave untranslated.

    Returns rate=None (NOT 0.0) when there is nothing left to confidently
    classify either way — 0.0 would silently claim "confidently 0%
    correct," which is a different, wrong thing to say. Found 2026-07-12:
    the first real run reported "0%" on a single-tag Spanish result that
    was actually just fully ambiguous (no confident classification at
    all), indistinguishable at a glance from a genuine failure."""
    if lang_code not in _SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang_code: {lang_code!r}")
    exempt_lower = {e.lower() for e in exempt}
    items = [t for t in items if not is_verbatim_exempt(t) and t.lower() not in exempt_lower]

    if lang_code in _SCRIPT_RANGES:  # exact, no unknown bucket (incl. "zh")
        classified = [(t, detect_in_language(t, lang_code)) for t in items]
        misses = [t for t, ok in classified if ok is False]
        unknowns: list[str] = []
    else:  # "es", "ja", "fr", "pt", "de", "id" — can genuinely be ambiguous
        classified = [(t, detect_in_language(t, lang_code)) for t in items]
        misses = [t for t, ok in classified if ok is False]
        unknowns = [t for t, ok in classified if ok is None]
    confident_total = len(items) - len(unknowns)
    consistent = confident_total - len(misses)
    rate = (consistent / confident_total) if confident_total else None
    return rate, misses, unknowns


if __name__ == "__main__":
    # Quick self-check against real VLM output captured in an early
    # spot-check — no VLM call, just verifying the detector behaves sanely on known examples before
    # trusting it in the tier script.
    spanish_tags = [
        "torre eiffel", "cielo azul claro", "árbol", "césped verde",
        "puerta blanca", "car", "house", "coche", "matrícula",
    ]
    chinese_tags = ["埃菲尔铁塔", "蓝天", "car", "buildings", "云朵", "汽车"]
    chinese_sentence = "这段视频展示了不同场景中的人们与汽车互动的画面。"
    spanish_sentence = "El video muestra diferentes escenas que involucran coches y personas."
    english_sentence = "This is a photograph of the Eiffel Tower against a clear sky."

    rate, misses, unknowns = language_consistency_rate(spanish_tags, "es")
    print(f"Spanish tags: {rate:.0%} of confidently-classified consistent")
    print(f"  English strays: {misses}")
    print(f"  unknown/ambiguous (needs manual review, not scored either way): {unknowns}")

    rate, misses, unknowns = language_consistency_rate(chinese_tags, "zh")
    print(f"\nChinese tags: {rate:.0%} consistent, English strays: {misses}, unknowns: {unknowns}")

    print(f"\nChinese sentence is_chinese: {is_chinese(chinese_sentence)} (expect True)")
    print(f"Spanish sentence is_spanish: {is_spanish(spanish_sentence)} (expect True)")
    print(f"English sentence is_spanish: {is_spanish(english_sentence)} (expect False)")
    print(f"'coche' is_spanish: {is_spanish('coche')} (expect True — Spanish word, "
          f"no diacritic, but NOT in the English word list either -> falls to None "
          f"if this prints None, confirming the 'don't guess' behavior is real)")
