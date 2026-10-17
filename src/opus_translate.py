"""
Multi-language English->X translation via the OPUS-MT NMT engine (CTranslate2 +
SentencePiece) — CPU-only, in-process, deterministic (beam search, no
temperature), no Ollama / no Metal contention with the vision model.

WHY THIS IS A TWO-MODEL HYBRID (2026-07-23)
Each language uses TWO Opus-MT models, tried in order per segment:

  1. tc-big  (Helsinki-NLP opus-mt-TC-BIG-en-XX) — PREFERRED. Better word
     choice on short/isolated phrases (the tag failure tail that MiniCPM and
     the base model both get wrong: fork->tenedor not "hork", nestle logo
     named not "anidale").
  2. base    (Helsinki-NLP opus-mt-en-XX)        — SAFETY NET. Its
     SentencePiece was trained with character_coverage=1.0, so it can NEVER
     emit <unk>; whatever tc-big can't represent, base can.

The tc-big models were trained with character_coverage=0.9995 and NO
byte_fallback, so they physically cannot encode (source) or generate (target)
the rarest ~0.05% of characters — proven at the token level: their 32k target
vocab contains ZERO pieces for `ü`, ZERO for accented UPPERCASE (Í Ñ Ç …), and
lacks ~100 symbols (© € • ± × # @ …). So `pingüino` -> `ping<unk>ino`,
`PELÍCULA` -> `PEL<unk>CULA`, `€` dropped/misread. This is a KNOWN, family-wide
tc-big defect (independently reported on the HF forum for en-fr `Ç/È`), NOT our
CTranslate2 conversion — the reference `transformers` model does the same thing,
it just hides the <unk> as a silent misspelling ("pingino"). The base models
(coverage 1.0) handle every one of these correctly.

So per segment: try tc-big; if it hits <unk> on EITHER the source encoding
(char it can't read) OR the generated output (char it can't write), fall back
to base for that segment. On real content ~93% of sentences stay on tc-big and
base surgically catches the rest — the two models' weaknesses are disjoint
(tc-big = characters, base = a word-choice tail), so the hybrid beats either
alone. See docs/findings.md (Translation).

ADDING A LANGUAGE
Install its models via translation_models.py (downloads into
config.translation_model_dir(lang)/{base,tcbig}); no code change here. base is
mandatory (the safety net); tcbig is optional — a language with only base
installed simply runs base-only. The engine DISCOVERS what's installed.

A MULTI-TARGET GROUP LANGUAGE (e.g. Chinese)
Some Opus-MT pairs (en-zh) are trained on a GROUP of related varieties (Mandarin
Simplified/Traditional, Cantonese, Wu, ...) and require a sentence-initial
`>>id<<` selector token or the output script/variety is undefined — confirmed
empirically for `cmn` (Mandarin, unspecified script): it silently leaked raw
ASS/SSA subtitle-formatting codes into output on some inputs, almost certainly
subtitle-corpus contamination in that generic group's training data — always
use the specific `cmn_Hans`/`cmn_Hant` form, never the bare group code. A
component with this requirement ships an optional `meta.json` next to its
model files (`{"prefix_token": ">>cmn_Hans<<", "needs_eos": true}`) — the
prefix is a LITERAL vocab entry, not plain text (SentencePiece splits the
typed string `">>cmn_Hans<<"` into subword garbage), so it's prepended as its
own token in `_Layer.encode`, never run through `sp.encode`. `needs_eos` in
the same file can also override the tcbig/base default learned from Spanish —
Chinese's `base` layer needed the `</s>`-append fix too (Spanish's base
didn't), so don't assume the split holds for a new language without
empirically testing it the way Chinese was.

ALSO found via Chinese: `base`'s "shared vocabulary" is not always ONE
jointly-trained SentencePiece model the way Spanish's apparently is — for
en-zh, `source.spm` doesn't even recognize Chinese pieces (`piece_to_id`
returns `<unk>` for them) despite decode "succeeding" by falling back to raw
string concatenation, which leaves a stray un-stripped leading "▁" on any
output piece SentencePiece's decode doesn't otherwise recognize. `base` now
prefers separate `source.spm`/`target.spm` files (same convention as tcbig)
when both are present, falling back to the single `sentencepiece.model` file
for Spanish's already-shipped format. Always decode target-language output
through a tokenizer that was actually trained on that language.

FAILURE MODE
Never raises. A missing/corrupt model dir, or missing ctranslate2/sentencepiece,
degrades to "text/tags stay in English" (safe: an English tag is still
searchable and never wrong), not a crash.
"""
import json
import re

import config
import profanity

# lang -> list[_Layer] (preferred first, base last), or [] if nothing usable.
# `None` marks a language whose load has been attempted and yielded nothing.
_ENGINES: dict = {}


class _Layer:
    """One loaded Opus-MT model + its tokenizers."""

    def __init__(self, name, translator, sp_src, sp_tgt, needs_eos, prefix_token=None):
        self.name = name
        self.translator = translator
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.needs_eos = needs_eos
        self.prefix_token = prefix_token
        self.unk_id = sp_tgt.unk_id()
        self.unk_piece = sp_tgt.id_to_piece(self.unk_id)

    def source_has_unk(self, text: str) -> bool:
        return self.sp_src.unk_id() in self.sp_src.encode(text, out_type=int)

    def encode(self, text: str):
        toks = self.sp_src.encode(text, out_type=str)
        # A multi-target GROUP model (e.g. English->Chinese-family) needs a
        # sentence-initial `>>id<<` selector token or the output script/variety
        # is undefined. It's a LITERAL vocab
        # entry, not ordinary text: running it through SentencePiece splits it
        # into subword garbage, so it's prepended as its own token instead.
        if self.prefix_token:
            toks = [self.prefix_token] + toks
        return toks + ["</s>"] if self.needs_eos else toks

    def output_has_unk(self, hyp_tokens) -> bool:
        return self.unk_piece in hyp_tokens

    def decode(self, hyp_tokens) -> str:
        # sp.decode() only reliably converts SentencePiece's "▁" space-marker
        # to a real space for pieces the tokenizer recognizes in its OWN vocab
        # (found twice now, in two unrelated model files — Helsinki's base
        # decoded through the wrong-language spm, and separately Argos's
        # "shared" en-zh spm not recognizing some of its own generated
        # pieces): when the model's first generated token is an unrecognized
        # fused "▁word" piece, the raw "▁" survives instead of becoming a
        # space. Python's .strip() doesn't catch it (U+2581 isn't Unicode
        # whitespace). Fix at the source rather than per-model: strip a
        # leading "▁" unconditionally, same as the official Marian
        # postprocess.sh convention (`sed 's/ //g;s/▁/ /g'`).
        return self.sp_tgt.decode(hyp_tokens).lstrip("▁").lstrip()


def _layer_meta(comp_dir, default_needs_eos: bool) -> tuple:
    """Optional `meta.json` alongside a component's model files, overriding the
    (tcbig=needs_eos True, base=needs_eos False) default learned from Spanish
    and declaring a required group-selector prefix token (e.g. Chinese's
    `>>cmn_Hans<<`; Chinese's base needed this override too, unlike Spanish). Spanish's on-disk models predate this file
    and have none — the old hardcoded default applies unchanged for them."""
    meta_path = comp_dir / "meta.json"
    if not meta_path.exists():
        return default_needs_eos, None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"  WARNING: bad meta.json in {comp_dir} ({e})", flush=True)
        return default_needs_eos, None
    return meta.get("needs_eos", default_needs_eos), meta.get("prefix_token")


def _load_layers(lang: str) -> list:
    """Discover + load the installed models for `lang`, preferred first.
    tcbig/ (separate source/target spm) then base/ (single shared spm).
    Returns [] if neither is usable."""
    try:
        import ctranslate2
        import sentencepiece as spm
    except Exception as e:  # pragma: no cover - import guard
        print(f"  WARNING: translation engine unavailable ({e}) — output stays "
              "in English", flush=True)
        return []

    root = config.translation_model_dir(lang)
    layers = []

    tc = root / "tcbig"
    if (tc / "model.bin").exists():
        needs_eos, prefix_token = _layer_meta(tc, default_needs_eos=True)
        try:
            layers.append(_Layer(
                "tcbig",
                ctranslate2.Translator(str(tc), device="cpu"),
                spm.SentencePieceProcessor(model_file=str(tc / "source.spm")),
                spm.SentencePieceProcessor(model_file=str(tc / "target.spm")),
                needs_eos=needs_eos, prefix_token=prefix_token,
            ))
        except Exception as e:
            print(f"  WARNING: tc-big {lang} model failed to load ({e})", flush=True)

    base = root / "base"
    base_ct2 = base / "ctranslate2"
    if (base_ct2 / "model.bin").exists():
        needs_eos, prefix_token = _layer_meta(base, default_needs_eos=False)
        try:
            src_spm, tgt_spm = base / "source.spm", base / "target.spm"
            if src_spm.exists() and tgt_spm.exists():
                # Two GENUINELY separate SentencePiece models sharing one
                # numeric embedding ID space, NOT one jointly-trained shared
                # vocab (Spanish's `sentencepiece.model` case below) — found
                # via Chinese: `source.spm` doesn't even recognize Chinese
                # pieces (piece_to_id -> unk), and decoding target output
                # through it leaves a stray un-stripped leading "▁" on any
                # fused first-piece SentencePiece doesn't recognize as its
                # own. Always decode through target.spm when it exists.
                sp_src = spm.SentencePieceProcessor(model_file=str(src_spm))
                sp_tgt = spm.SentencePieceProcessor(model_file=str(tgt_spm))
            else:
                shared = spm.SentencePieceProcessor(
                    model_file=str(base / "sentencepiece.model"))
                sp_src = sp_tgt = shared
            layers.append(_Layer(
                "base",
                ctranslate2.Translator(str(base_ct2), device="cpu"),
                sp_src, sp_tgt, needs_eos=needs_eos, prefix_token=prefix_token,
            ))
        except Exception as e:
            print(f"  WARNING: base {lang} model failed to load ({e})", flush=True)

    if not layers:
        print(f"  WARNING: no OPUS-MT model installed for '{lang}' — output "
              "stays in English (run the language download)", flush=True)
    return layers


def _engine(lang: str) -> list:
    """Lazy, cached per-language load."""
    if lang not in _ENGINES:
        _ENGINES[lang] = _load_layers(lang)
    return _ENGINES[lang]


def _split_sentences(text: str) -> list:
    """Split on sentence-final .!? + whitespace, KEEPING any immediately
    following close-quote with that sentence (a plain lookbehind split eats
    the closing " at a `." ` boundary). Per-sentence translation is MANDATORY
    for NMT: a whole paragraph in one call truncates once it's long enough
    (both models)."""
    marked = re.sub(r'([.!?]["”]?)\s+', lambda m: m.group(1) + "\x00", text.strip())
    return [p for p in marked.split("\x00") if p]


_BEAM = 4

# Found 2026-07-30 testing a Chinese candidate layer against real
# tags (bucket, curtain, fur — ordinary, common words, not edge cases): two
# corruption shapes that DON'T set the SentencePiece <unk> token, so
# output_has_unk alone misses them entirely. (1) A literal ASS/SSA subtitle
# styling code leaking into the translation verbatim — e.g. "bucket" produced
# `{\fn黑体\fs20\shad2...}你这个混蛋` (that tail is a Chinese profanity, not
# a mistranslation of "bucket" at all) — almost certainly subtitle-corpus
# contamination in training data, same root cause as the `>>cmn<<` bare
# group-token bug found earlier. (2) Runaway word repetition post-decode
# ("fur" -> "皮毛 皮毛 皮毛 皮毛 皮毛 皮毛") that the </s>-append fix does not
# universally prevent. Both are worse than <unk> — confidently wrong/offensive
# output, not a visibly missing character — so they get the same cascade
# treatment: reject and fall through to the next (safer) layer.
_SUBTITLE_CODE_RE = re.compile(r"\{\\[a-zA-Z]")


def _looks_corrupted(text: str) -> bool:
    if _SUBTITLE_CODE_RE.search(text):
        return True
    words = text.split()
    return any(words[i] == words[i + 1] == words[i + 2]
               for i in range(len(words) - 2))


def _translate_segment(layers: list, text: str, max_len: int, lang: str) -> str:
    """Run one segment through the layer cascade: return the first layer's
    output that is <unk>-free and not otherwise obviously corrupted (see
    _looks_corrupted). Every layer is validated uniformly, INCLUDING the
    last one (base) — found 2026-07-30 against real footage tags that
    base's assumed "character_coverage=1.0, can never emit <unk>" property,
    true for Spanish, does NOT hold for Chinese (e.g. "basil"/"skewers"/
    "quiche" produced a real, visible <unk> glyph through base too). An
    unconditionally-accepted last layer would ship that glyph straight into
    a written tag/description. If EVERY layer fails validation (including a
    single-layer or empty list), the ORIGINAL English text is returned
    instead — "untranslated" is searchable and safe, a visible <unk> or
    other corruption is not, matching this project's established Spanish
    precedent (opus-mt's worst case was always "leave it untranslated,"
    never "confidently wrong"). Also rejects HALLUCINATED profanity — output
    containing a profane term (profanity.contains_profanity) that wasn't
    already present in the English source — same treatment as corruption,
    not a content-policy decision (see profanity.py's docstring: whether to
    keep profanity that's genuinely IN the source is a separate, unrelated
    question this function has no opinion on)."""
    if not text:
        return text
    source_has_profanity = profanity.contains_profanity(text, "en")
    for layer in layers:
        # Source-side hole: this model can't even read the input faithfully
        # (e.g. tc-big on "€50" drops the € and hallucinates a currency) —
        # a translation attempt on unfaithfully-read input isn't trustworthy
        # even as a last resort, so this defers/skips uniformly too.
        if layer.source_has_unk(text):
            continue
        try:
            hyp = layer.translator.translate_batch(
                [layer.encode(text)], beam_size=_BEAM,
                max_decoding_length=max_len)[0].hypotheses[0]
        except Exception as e:
            print(f"  WARNING: {layer.name} translate failed ({e})", flush=True)
            continue
        if layer.output_has_unk(hyp):
            continue
        decoded = layer.decode(hyp)
        if _looks_corrupted(decoded):
            continue
        if not source_has_profanity and profanity.contains_profanity(decoded, lang):
            continue
        return decoded
    return text


# --- brand pass-through -------------------------------------------------------
# Tags ending in "logo" are brand names that are ALSO ordinary English words
# (apple logo, blender institute); translating them by literal meaning is wrong
# ("logo de manzana"). Keep the English tag verbatim instead — brands that
# aren't English homographs (nike, toyota) already copy through untranslated.
def _is_brand_passthrough(tag: str) -> bool:
    return tag.strip().lower().endswith("logo")


def translate_tags(tags: list, lang: str) -> list:
    """EN -> `lang` for an already-clean tag list. Per tag: brand pass-through,
    else the tc-big->base cascade. Never drops, reorders, or raises — an
    untranslatable tag stays English."""
    if not tags:
        return tags
    layers = _engine(lang)
    if not layers:
        return tags
    out = []
    for tag in tags:
        if _is_brand_passthrough(tag):
            out.append(tag)
            continue
        es = _translate_segment(layers, tag, max_len=64, lang=lang).strip().lower()
        # Found 2026-07-30 testing Portuguese (0/396 on Spanish's own corpus,
        # so this isn't a pre-existing regression there): both pt models
        # sometimes treat a bare noun/short phrase as an implicit one-word
        # SENTENCE and append terminal punctuation ("airplane" -> "avião."),
        # which would otherwise ship straight into a written tag. A tag is
        # never a sentence, so a single trailing '.'/'!'/'?' is always noise,
        # never real content, in any language.
        es = es.rstrip(".!?") or es
        out.append(es or tag)
    return out


def translate_text(text: str, lang: str) -> str:
    """EN -> `lang` for a description/summary. Sentence-split (mandatory for
    NMT), each sentence through the cascade, rejoined."""
    if not text:
        return text
    layers = _engine(lang)
    if not layers:
        return text
    return " ".join(_translate_segment(layers, s, max_len=512, lang=lang)
                    for s in _split_sentences(text))

