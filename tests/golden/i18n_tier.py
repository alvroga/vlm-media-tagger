"""Multilingual-output tier: can tags/descriptions/summaries be produced in
another language instead of English? Two candidate approaches:

  native    — the VLM generates directly in the target language (the
              pipeline's own prompts, with a "respond in {language}"
              instruction appended).
  translate — the VLM generates the normal ENGLISH output first (the
              real pipeline path, untouched), then a SEPARATE text-only
              call (no image) translates that already-correct English
              text into the target language.

Reuses EXISTING golden-set content (a sample of `entries/*-still` items +
public `video-full` entries) and EXISTING golden-set ground truth
(hand-translated into `TRANSLATED_EXPECTATIONS` below, from the English
`expected.json` files — never machine-translated, to avoid validating a
translator's ground truth with an untested translator). Scoring:

  language consistency — via i18n_lang_id.language_consistency_rate:
      what fraction of tags/sentences are actually in the target
      language, vs. English leaking through.
  content fidelity — via harness._satisfied/_mentions against
      TRANSLATED_EXPECTATIONS: did it still get the CONTENT right, a
      genuinely separate question from language compliance.

Both are rates across N runs, never pass/fail — this is a comparison
instrument (native vs. translate, Spanish vs. Chinese), not a certification
gate. Manually audit a sample of the saved output before trusting any
number here, same discipline as logodet_tier.py/landmark_tier.py.

The shipped answer is the OPUS-MT step (src/opus_translate.py); this tier
is how the VLM-only alternatives were measured against it. Don't run it
alongside another VLM-heavy job: they contend for the GPU and both slow down.

    python tests/golden/i18n_tier.py --dry-run          # sanity-check sampling/prompts, no VLM calls
    python tests/golden/i18n_tier.py --langs es --approaches native --runs 1 --limit 3
    python tests/golden/i18n_tier.py --save
"""
import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

GOLDEN = Path(__file__).parent
sys.path.insert(0, str(GOLDEN.parent.parent / "src"))
sys.path.insert(0, str(GOLDEN))

from i18n_lang_id import detect_in_language, language_consistency_rate  # noqa: E402

import config  # noqa: E402
import vlm  # noqa: E402
from detect_objects_vlm import (  # noqa: E402
    DETECT_PASSES,
    OBJECT_PROMPT,
    STILL_OBJECT_PROMPT,
    _call_vlm,
    detect_objects_vlm,
)
from harness import _mentions, _satisfied  # noqa: E402
from summarize import (  # noqa: E402
    CAPTION_PROMPT,
    IMAGE_CAPTION_PROMPT,
    _caption_frame,
    clean_description,
    summarize_video,
)
from tag_filter import clean_tags  # noqa: E402

LANGS = {
    "es": "Spanish (español)",
    "zh": "Simplified Chinese (简体中文)",
    # A wider 10-language screen. Only the "translate" approach is worth
    # running for these: "native" lost on es/zh for structural reasons
    # (the English-only tag filter can't see its output), not
    # language-specific ones, so re-testing it per language doubles cost
    # for a question already answered.
    "hi": "Hindi (हिन्दी)",
    "ar": "Arabic (العربية)",
    "fr": "French (français)",
    "pt": "Portuguese (português)",
    "it": "Italian (italiano)",
    "ru": "Russian (русский)",
    "ja": "Japanese (日本語)",
    "de": "German (Deutsch)",
    "ko": "Korean (한국어)",
    "id": "Indonesian (Bahasa Indonesia)",
}
APPROACHES = ["native", "translate"]

# A domain-diverse sample of existing "still" golden entries — landmarks
# (localization test), brands/logos (proper-noun-should-NOT-translate
# test), text-heavy/design content (OCR+translation interaction), and a
# source that's ITSELF about
# reading Chinese text (multilingual-chinese — a deliberate confound
# check: does asking for Chinese OUTPUT interact oddly with a source
# image that's ABOUT transcribing Chinese signage?). Not exhaustive —
# a starter set sized for a real but bounded first deep run.
SAMPLE_STILL_ENTRIES = [
    "landmark-eiffel", "landmark-golden-gate", "landmark-toledo",
    "logo-real-apple", "logo-real-obscure-caramac",
    "night-street-lit", "poster-bbb", "multilingual-chinese",
]
SAMPLE_VIDEO_FULL_ENTRIES = [
    "video-long-pan-skyline", "video-nature-water-rocks",
]

# Hand-translated from each entry's own English expected.json (nice_to_have
# / description_must_mention) — NOT machine-translated, to avoid using an
# untested translator to build this tier's own ground truth. Deliberately
# a SUBSET of the sample above: proper nouns (Big Buck Bunny, brand names)
# are left UNTRANSLATED on purpose, since they should stay as-is in real
# output too — landmark-toledo (a false-guess trap, not a positive-content
# check) and the proper-noun-heavy entries are scored on
# language-consistency only, no translated nice_to_have.
TRANSLATED_EXPECTATIONS = {
    "landmark-eiffel": {
        "es": {"nice_to_have": ["torre eiffel|torre"],
               "description_must_mention": ["torre eiffel", "parís|paris"]},
        "zh": {"nice_to_have": ["埃菲尔铁塔|铁塔"],
               "description_must_mention": ["埃菲尔铁塔", "巴黎"]},
    },
    "landmark-golden-gate": {
        "es": {"nice_to_have": ["puente golden gate|puente"],
               "description_must_mention": ["puente golden gate", "san francisco"]},
        "zh": {"nice_to_have": ["金门大桥|大桥"],
               "description_must_mention": ["金门大桥", "旧金山"]},
    },
    # Brand-name localization probe: "Apple" should stay "Apple" in
    # Spanish (translating to "manzana" — the fruit — would be a real
    # over-translation bug), but "苹果" IS the normal, correct
    # Chinese name for the company (unlike Spanish) — asymmetric on
    # purpose, this is the interesting case, not a copy-paste.
    "logo-real-apple": {
        "es": {"nice_to_have": ["apple"]},
        "zh": {"nice_to_have": ["apple|苹果"]},
    },
    "poster-bbb": {
        "es": {"nice_to_have": ["cartel|póster", "ardilla|roedor",
                                 "mariposa", "animación|dibujos animados"]},
        "zh": {"nice_to_have": ["海报", "松鼠|龟齿动物",
                                 "蝴蝶", "动画|卡通"]},
        # studio credit names + the film's own URL: verbatim printed text,
        # correct to leave untranslated (the first real run flagged these
        # as false "English leakage").
        # "big buck bunny" (the title) also exempted: real-world movie
        # titles conventionally keep their original release name rather
        # than translating, so leaving it as-is is the CORRECT behavior,
        # not a failure to translate.
        "expected_untranslated": ["wavemage", "maqina", "filmmore",
                                   "blender institute", "big buck bunny"],
    },
    "night-street-lit": {
        "es": {"nice_to_have": ["calle|carretera", "farola|poste de luz",
                                 "semáforo", "árbol", "noche"]},
        "zh": {"nice_to_have": ["街道|道路", "路灯",
                                 "红绿灯|交通灯", "树", "夜晚"]},
    },
}


def tag_prompt(base: str, lang_label: str) -> str:
    return (
        base
        + f"\n\nRespond in {lang_label}: keep the JSON key \"objects\" in "
        f"English, but write every object-name VALUE in {lang_label}. Do "
        "not translate the example placeholder words into your answer — "
        "they must never appear regardless of language."
    )


def prose_prompt(base: str, lang_label: str) -> str:
    return base + f"\n\nWrite your entire response in {lang_label}."


def _safe_caption(path: str, model: str, prompt: str) -> str:
    """Wraps summarize._caption_frame the way its own callers do
    (try/except, "" on failure): _caption_frame has no internal
    protection by design, the caller owns it."""
    try:
        return _caption_frame(path, model, prompt=prompt)
    except Exception as e:
        print(f"  WARNING: caption call failed ({e})", flush=True)
        return ""


def _translate_text_once(text: str, lang_label: str, model: str) -> str:
    """Returns "" on any server error instead of raising, like
    detect_objects_vlm._call_vlm. Without this, a single server 500 (a
    grammar-constrained JSON failure) crashed a whole 10-language run on
    its first combo instead of failing just that one call."""

    prompt = (
        f"Translate the following text into {lang_label}. Output ONLY the "
        f"translation, nothing else — no notes, no quotation marks, no "
        f"restating the original:\n\n{text}"
    )
    try:
        text = vlm.chat([{"role": "user", "content": prompt}], model=model,
            temperature=0.2, num_predict=512)
        return clean_description(config.strip_thinking(text))
    except Exception as e:
        print(f"  WARNING: translate_text call failed ({e})", flush=True)
        return ""


def translate_text(text: str, lang_label: str, model: str, lang_code: str) -> str:
    """Text-only translation call (no image) — the 'translate' approach's
    core primitive. Isolates translation quality from image-understanding
    entirely; also the vehicle for testing the "MiniCPM-V's own text
    backbone as a translator, no new model download" hypothesis.

    Retries once if the output is empty (the call failed — see
    _translate_text_once) OR didn't actually land in the target language
    (checked directly via detect_in_language, not just hoped) —
    symmetric with translate_tags's retry-on-echo fix. Uses the generic
    detect_in_language dispatcher, never a hard-coded per-language check
    (which would silently mis-score every other language)."""
    def _in_target(t: str) -> bool:
        return bool(detect_in_language(t, lang_code))

    result = _translate_text_once(text, lang_label, model)
    if not result or not _in_target(result):
        retried = _translate_text_once(text, lang_label, model)
        if retried and _in_target(retried):
            result = retried
    return result


_TRANSLATE_CHUNK_SIZE = 20  # see translate_tags — bounds each call's output size


def _looks_untranslated(original: list[str], translated: list[str]) -> bool:
    """True if a majority of `translated` is identical (case-insensitive)
    to the corresponding `original` English item — the signature of a
    chunk the model silently echoed back instead of translating: one whole
    20-item chunk came back English, valid JSON, right count, so the
    length/emptiness checks alone don't catch it."""
    if len(original) != len(translated) or not original:
        return False
    same = sum(1 for o, t in zip(original, translated) if o.strip().lower() == t.strip().lower())
    return same / len(original) > 0.5


def _translate_tags_once(tags: list[str], lang_label: str, model: str) -> list[str]:
    """Returns [] on any server error instead of raising — same
    fix and same reason as _translate_text_once above."""
    from detect_objects_vlm import _parse_vlm_objects

    prompt = (
        f"Translate each of these English words/phrases into {lang_label}, "
        "one for one, preserving the order and count exactly. Keep real "
        "brand/product/proper names, URLs, and verbatim printed text "
        "as-is unless there is a standard, widely-used "
        f"{lang_label} name for that specific brand. Respond "
        'with ONLY this JSON format: {"objects": ["translation one", '
        f'"translation two", ...]}}\n\nWords: {json.dumps(tags)}'
    )
    # generous per-item budget (~40 tokens) — chunking already bounds the
    # worst case, this is defense in depth, not the primary fix
    num_predict = min(2048, max(512, len(tags) * 40))
    try:
        text = vlm.chat([{"role": "user", "content": prompt}], model=model,
            temperature=0.2, num_predict=num_predict)
        return _parse_vlm_objects(text)
    except Exception as e:
        print(f"  WARNING: translate_tags call failed ({e})", flush=True)
        return []


def translate_tags(tags: list[str], lang_label: str, model: str) -> list[str]:
    """Text-only translation of an already-clean English tag list, JSON
    in/out — same structural contract as detection so results are
    directly comparable to the native approach's tag output.

    Chunks into batches of _TRANSLATE_CHUNK_SIZE. A multi-shot video
    (50+ unique tags in one call) came back almost entirely untranslated
    at the old fixed
    num_predict=512 — very likely truncation on a batch that size, not a
    genuine model limitation (a single still's handful of tags translated
    fine). Chunking bounds each call's expected output regardless of how
    large the source tag list is, which is a more robust fix than trying
    to guess one num_predict value that covers every possible list size.

    Retries once on a failed/echoed-back chunk (empty parse, OR a majority
    of items coming back identical to the English input — see
    _looks_untranslated) — the same "retry once on failure" pattern the
    stills path uses, since chunking alone didn't fully close the gap: the
    first chunked run still had one whole chunk echo back English on the
    Chinese run (83% vs Spanish's clean 100%)."""
    if not tags:
        return tags
    if len(tags) > _TRANSLATE_CHUNK_SIZE:
        out: list[str] = []
        for i in range(0, len(tags), _TRANSLATE_CHUNK_SIZE):
            out += translate_tags(tags[i:i + _TRANSLATE_CHUNK_SIZE], lang_label, model)
        return out

    translated = _translate_tags_once(tags, lang_label, model)
    if not translated or _looks_untranslated(tags, translated):
        retried = _translate_tags_once(tags, lang_label, model)
        if retried and not _looks_untranslated(tags, retried):
            translated = retried

    if not translated:
        return tags  # fall back rather than silently dropping
    if len(translated) != len(tags):
        print(f"  (warning: translate_tags length mismatch — {len(tags)} in, "
              f"{len(translated)} out; keeping the partial translation, not "
              "discarding it)", flush=True)
    elif _looks_untranslated(tags, translated):
        print("  (warning: translate_tags chunk still looks untranslated "
              "after retry — keeping it anyway rather than silently "
              "discarding real output)", flush=True)
    return translated


def still_baseline(img: Path, model: str | None) -> tuple[list[str], str]:
    """Real pipeline stills path in ENGLISH — same shape as
    logodet_tier.py/landmark_tier.py's process(). Used directly by the
    'translate' approach, and as the reference point for judging the
    'native' approach's content fidelity."""
    union = []
    for _ in range(DETECT_PASSES):
        union += _call_vlm({"path": str(img)}, model=model, prompt=STILL_OBJECT_PROMPT)
    tags = clean_tags(union)
    m = model or config.VISION_MODEL
    desc = clean_description(_safe_caption(str(img), m, prompt=IMAGE_CAPTION_PROMPT))
    return tags, desc


def still_native(img: Path, lang_label: str, model: str | None) -> tuple[list[str], str]:
    """'native' approach for a still: ask the VLM to respond directly in
    the target language, tags and description both."""
    prompt = tag_prompt(STILL_OBJECT_PROMPT, lang_label)
    union = []
    for _ in range(DETECT_PASSES):
        union += _call_vlm({"path": str(img)}, model=model, prompt=prompt)
    m = model or config.VISION_MODEL
    desc_prompt = prose_prompt(IMAGE_CAPTION_PROMPT, lang_label)
    desc = clean_description(_safe_caption(str(img), m, prompt=desc_prompt))
    return union, desc  # raw union, NOT clean_tags — that filter is English-only


def still_translate(img: Path, lang_label: str, model: str | None, lang_code: str) -> tuple[list[str], str]:
    """'translate' approach for a still: real English baseline, then a
    separate text-only translation pass over it."""
    tags, desc = still_baseline(img, model)
    m = model or config.VISION_MODEL
    return translate_tags(tags, lang_label, m), translate_text(desc, lang_label, m, lang_code)


def _combine_captions(captions: list[str], lang_label: str, model: str) -> str:
    """Returns "" on any server error instead of raising — same
    fix and same reason as _translate_text_once/_translate_tags_once."""

    joined = "\n".join(f"- {c}" for c in captions)
    prompt = (
        "Here are captions from frames sampled across one video, in order:\n"
        f"{joined}\n\nWrite a concise 2-3 sentence overall summary of what "
        f"this video is about. Write your entire response in {lang_label}."
    )
    try:
        text = vlm.chat([{"role": "user", "content": prompt}], model=model,
            temperature=0.2, num_predict=512)
        return clean_description(config.strip_thinking(text))
    except Exception as e:
        print(f"  WARNING: combine-captions call failed ({e})", flush=True)
        return ""


@contextmanager
def _scratch_video_dir(entry_dir: Path):
    """Stages a video-full entry's frozen cuts.json+frames into a scratch
    config.DATA_DIR so the REAL detect_objects_vlm()/summarize_video()
    (and their internal frame-selection helpers — shot-slicing OR the
    has_cuts=False interval fallback, whichever the entry's cuts.json
    actually calls for) run completely unmodified — same technique
    harness.py's run_video_full_entry uses. Frames are symlinked, never
    copied or written to. Yields video_id to pass into those functions."""
    import shutil
    import tempfile

    video_id = entry_dir.name
    raw_index = json.loads((entry_dir / "frames" / "index.json").read_text())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        work_dir = tmp_path / video_id
        (work_dir / "frames").mkdir(parents=True)
        for e in raw_index:
            (work_dir / "frames" / e["filename"]).symlink_to(
                entry_dir / "frames" / e["filename"])
        (work_dir / "frames" / "index.json").write_text(json.dumps(
            [{**e, "path": str(work_dir / "frames" / e["filename"])} for e in raw_index]))
        shutil.copy(entry_dir / "cuts.json", work_dir / "cuts.json")

        orig_data_dir = config.DATA_DIR
        config.DATA_DIR = tmp_path
        try:
            yield video_id
        finally:
            config.DATA_DIR = orig_data_dir


def video_full_baseline(entry_dir: Path, model: str | None) -> tuple[list[str], str]:
    """Real pipeline video-full path in ENGLISH. Used by the 'translate'
    approach, and as the reference point for the 'native' approach's
    content-fidelity scoring."""
    with _scratch_video_dir(entry_dir) as video_id:
        segments = detect_objects_vlm(video_id, model=model)
        tags = clean_tags(sorted(set(s["class"] for s in segments)))
        summary = summarize_video(video_id, model=model) or ""
    return tags, summary


def video_full_native(entry_dir: Path, lang_label: str, model: str | None) -> tuple[list[str], str]:
    """'native' approach for a video-full entry: the REAL
    detect_objects_vlm() with a language-appended object_prompt — this
    gets correct shot-selection/interval-fallback, DETECT_PASSES, and
    threading for free by delegating to the real function rather than
    reimplementing them (an earlier version of this function globbed ALL
    raw frames directly, which was wrong on three counts: too many frames,
    single-pass instead of DETECT_PASSES, and it would have silently
    skipped video-nature-water-rocks's whole reason for existing — the
    has_cuts=False interval fallback — since it never calls
    detect_objects_vlm() to trigger that branch at all). For the summary,
    summarize_video() combines captions with an English prompt of its own,
    so the caption+combine loop is reimplemented here, but frame SELECTION still goes through the
    real summarize._select_summary_frames() for the same fidelity reason."""
    from summarize import _select_summary_frames

    tag_p = tag_prompt(OBJECT_PROMPT, lang_label)
    with _scratch_video_dir(entry_dir) as video_id:
        segments = detect_objects_vlm(video_id, model=model, object_prompt=tag_p)
        tags = sorted(set(s["class"] for s in segments))  # raw, NOT clean_tags (English-only filter)

        m = model or config.VISION_MODEL
        frames = _select_summary_frames(video_id)
        cap_prompt = prose_prompt(CAPTION_PROMPT, lang_label)
        captions = [c for f in frames
                    if (c := _safe_caption(f["path"], m, prompt=cap_prompt))]
        summary = _combine_captions(captions, lang_label, m)
    return tags, summary


def video_full_translate(entry_dir: Path, lang_label: str, model: str | None, lang_code: str) -> tuple[list[str], str]:
    tags, summary = video_full_baseline(entry_dir, model)
    m = model or config.VISION_MODEL
    return translate_tags(tags, lang_label, m), translate_text(summary, lang_label, m, lang_code)


def score_item(tags: list[str], text: str, lang_code: str,
               expectations: dict | None, expected_untranslated: list[str] = ()) -> dict:
    exempt = frozenset(expected_untranslated)
    tag_rate, tag_misses, tag_unknowns = language_consistency_rate(tags, lang_code, exempt)
    text_rate, text_misses, _ = language_consistency_rate([text], lang_code, exempt)

    fidelity = {}
    if expectations:
        for key in ("nice_to_have", "description_must_mention", "summary_must_mention"):
            for exp in expectations.get(key, []):
                if key == "nice_to_have":
                    fidelity[exp] = _satisfied(exp, tags)
                else:
                    fidelity[exp] = _mentions(exp, text)

    return {
        # None (not 0.0) means nothing was confidently classifiable either
        # way — see language_consistency_rate's docstring for why that
        # distinction matters.
        "tag_language_rate": None if tag_rate is None else round(tag_rate, 3),
        "tag_english_strays": tag_misses,
        "tag_unknowns": tag_unknowns,
        "text_in_target_language": None if text_rate is None else bool(text_rate),
        "fidelity": fidelity,
    }


def _iter_sample(entries_filter: str | None):
    for name in SAMPLE_STILL_ENTRIES:
        if entries_filter and entries_filter not in name:
            continue
        yield "still", name, GOLDEN / "entries" / name
    for name in SAMPLE_VIDEO_FULL_ENTRIES:
        if entries_filter and entries_filter not in name:
            continue
        yield "video-full", name, GOLDEN / "entries" / name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None)
    ap.add_argument("--langs", default="es,zh", help="comma-separated: es,zh")
    ap.add_argument("--approaches", default="native,translate",
                     help="comma-separated: native,translate")
    ap.add_argument("--entries", default=None, help="substring filter on sample entry names")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="cap total (entry,lang,approach) combos")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the sampled entries/prompts/combo count, make NO VLM calls")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    model_name = args.model or config.VISION_MODEL
    langs = [c.strip() for c in args.langs.split(",") if c.strip()]
    approaches = [c.strip() for c in args.approaches.split(",") if c.strip()]
    for c in langs:
        assert c in LANGS, f"unknown lang code {c!r}, expected one of {list(LANGS)}"
    for a in approaches:
        assert a in APPROACHES, f"unknown approach {a!r}, expected one of {APPROACHES}"

    sample = list(_iter_sample(args.entries))
    combos = [
        (kind, name, path, lang, approach)
        for kind, name, path in sample
        for lang in langs
        for approach in approaches
    ]
    if args.limit:
        combos = combos[: args.limit]

    print(f"i18n tier: {len(sample)} entries x {len(langs)} lang(s) x "
          f"{len(approaches)} approach(es) = {len(combos)} combos, "
          f"{args.runs} run(s) each, model {model_name}")
    if args.dry_run:
        for kind, name, path, lang, approach in combos:
            has_expect = name in TRANSLATED_EXPECTATIONS and lang in TRANSLATED_EXPECTATIONS[name]
            print(f"  [{kind}] {name} / {LANGS[lang]} / {approach}"
                  f"{'  (has translated expectations)' if has_expect else ''}")
        print("\n--dry-run: no VLM calls made.")
        return

    t0 = time.time()
    results: dict = {}
    for n, (kind, name, path, lang, approach) in enumerate(combos, 1):
        lang_label = LANGS[lang]
        expectations = TRANSLATED_EXPECTATIONS.get(name, {}).get(lang)
        expected_untranslated = TRANSLATED_EXPECTATIONS.get(name, {}).get("expected_untranslated", [])
        run_scores = []
        for _ in range(args.runs):
            if kind == "still":
                img = next((path / "frames").glob("*.jpg"))
                tags, text = (still_native(img, lang_label, args.model) if approach == "native"
                              else still_translate(img, lang_label, args.model, lang))
            else:
                tags, text = (video_full_native(path, lang_label, args.model) if approach == "native"
                              else video_full_translate(path, lang_label, args.model, lang))
            run_scores.append({"tags": tags, "text": text,
                                **score_item(tags, text, lang, expectations, expected_untranslated)})

        results[f"{name}/{lang}/{approach}"] = {
            "kind": kind, "entry": name, "lang": lang, "approach": approach,
            "runs": run_scores,
        }
        known_rates = [r["tag_language_rate"] for r in run_scores if r["tag_language_rate"] is not None]
        rate_str = f"{sum(known_rates) / len(known_rates):.0%}" if known_rates else "N/A (all ambiguous)"
        elapsed = time.time() - t0
        eta = time.strftime("%H:%M", time.localtime(
            time.time() + (elapsed / n) * (len(combos) - n)))
        print(f"  [{n}/{len(combos)}] {name}/{lang}/{approach}: "
              f"tag-language {rate_str} | {elapsed / 60:.1f} min elapsed, "
              f"ETA {eta}", flush=True)

    print(f"\n{'=' * 60}\ndone, {(time.time() - t0) / 60:.1f} min")

    if args.save:
        safe_model = re.sub(r"[^\w.-]", "_", model_name)
        # A subset run (--entries/--langs/--approaches narrower than the
        # full sample) must never collide with the full-set filename, or a
        # quick re-test silently overwrites a full run's results.
        is_full = (not args.entries and set(langs) == set(LANGS)
                   and set(approaches) == set(APPROACHES) and not args.limit)
        suffix = ""
        if not is_full:
            scope = "-".join(filter(None, [
                re.sub(r"[^\w.-]", "_", args.entries or "")[:30],
                "-".join(langs) if set(langs) != set(LANGS) else "",
                "-".join(approaches) if set(approaches) != set(APPROACHES) else "",
            ]))
            suffix = f"-{scope}" if scope else "-subset"
        out = GOLDEN / "results" / f"i18n-{date.today()}-{safe_model}{suffix}.json"
        if out.exists() and is_full:
            print(f"\n(overwriting existing full-set baseline {out} — "
                  "make sure this is intentional)")
        out.write_text(json.dumps({
            "date": str(date.today()), "model": model_name,
            "langs": langs, "approaches": approaches, "runs": args.runs,
            "results": results,
        }, indent=1, default=list, ensure_ascii=False) + "\n")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
