"""The tuned VLM prompts, plus the reasoning behind each tuning step.

Every prompt here was changed one clause at a time and measured against the
golden harness (tests/golden/) before being kept; the comments record what
each clause fixed, what it was measured against, and the alternatives that
were tested and rejected. Read them before changing a prompt — several
"obvious" rewrites were already tried and lost.

There are two sets: the constants below are the MiniCPM-V 4.5 prompts, and
PROMPT_SETS["minicpm-v4.7"] (bottom of the file) derives the 4.7 set from
them by named clause swaps. config.PROMPT_SET picks one (it follows the
model by default).

Code reads prompts through get(name), never a bare constant, so an
experiment can swap one in without editing this file:

    import prompts
    with prompts.override(IMAGE_CAPTION_PROMPT="..."):
        ...
"""
import contextlib

PROMPT_NAMES = (
    "SYSTEM_PROMPT",
    "OBJECT_PROMPT",
    "STILL_OBJECT_PROMPT",
    "CAPTION_PROMPT",
    "IMAGE_CAPTION_PROMPT",
    "VIDEO_SUMMARY_PROMPT",
)

_overrides: dict[str, str] = {}


def get(name: str) -> str:
    if name not in PROMPT_NAMES:
        raise KeyError(f"Unknown prompt: {name}")
    if name in _overrides:
        return _overrides[name]
    import config
    return PROMPT_SETS[config.PROMPT_SET][name]


@contextlib.contextmanager
def override(**replacements: str):
    """Temporarily replace prompts by name (prompt experiments, harness variants)."""
    unknown = set(replacements) - set(PROMPT_NAMES)
    if unknown:
        raise KeyError(f"Unknown prompt(s): {', '.join(sorted(unknown))}")
    saved = dict(_overrides)
    _overrides.update(replacements)
    try:
        yield
    finally:
        _overrides.clear()
        _overrides.update(saved)


# --------------------------------------------------------------------------
# From detect_objects_vlm.py
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a visual content analyst identifying physical objects and notable "
    "visual elements in images. Only report what you can actually observe in THIS "
    "image — never default to example values or guess."
)

# Revised 2026-07-08 after finding minicpm-v anchors on literal example values in
# the JSON-format instruction when uncertain (confirmed by swapping the example
# words and watching the hallucination follow) — the old example
# ["car", "bow tie", "neon sign", "fence"] was leaking into unrelated frames
# (e.g. a sci-fi robot hologram scene). Also added a natural-elements category
# (was missing entirely — caused empty results on nature/landscape shots) and
# explicit blur/bokeh handling (was missing — caused empty results on
# out-of-focus close-ups even when the subject was clearly identifiable).
#
# Retuned 2026-07-10 for MiniCPM-V 4.5 (adopted that day), each change
# harness-verified in isolation:
# - "Max 12 objects" → "up to 25, more complete is better": v4.5 OBEYS caps
#   (2.6 ignored them) and was silently rationing away small/distant items
#   (license plate 67→100%).
# - readable-brand clause: v4.5 reads text superbly but doesn't volunteer it
#   as tags unasked (airline liveries 0%→67-100% with the clause). Same
#   never-guess discipline as everything else.
# - defining-condition clause: static noise 0→100%, underwater murk/seaweed
#   recovered. KNOWN LIMIT: v4.5 still won't tag 'night' on lit night
#   scenes, and an explicit day/night instruction was tested and REJECTED —
#   it false-tagged golden-hour footage as night (day-guard 3/3). Scene
#   attributes like day/night belong to a deterministic post-step,
#   not the VLM (detect_objects_vlm.apply_deterministic_tags).
OBJECT_PROMPT = (
    "List every physical object or notable visual element in this image — foreground AND background. "
    "Be comprehensive. Include: "
    "vehicles (car, truck, motorcycle, bus, van, bicycle) even when they fill the frame; "
    "vehicle parts in close-up (wheel, tire, headlight, grille, dashboard, steering wheel); "
    "clothing and accessories (hat, jacket, bow tie, sunglasses, watch, gloves, uniform, bag, backpack); "
    "buildings, structures, bridges, street lamps, fences; "
    "props, equipment, tools, instruments, furniture; "
    "signage, posters, neon signs, menu boards, banners, logos, screens; "
    "food, drink, and containers (cup, mug, glass, bottle, plate, bowl) even in extreme close-up; "
    "transit infrastructure (escalator, stairs, handrail, turnstile, platform, barrier); "
    "natural elements (water, rock, tree, plant, flower, sand, sky, cloud, animal). "
    "If the image is out-of-focus, blurred, or a stylized visual effect (bokeh, lens flare, "
    "abstract lighting), still name whatever is identifiable by shape or color, AND include the "
    'effect itself as an entry (e.g. "bokeh effect", "blurred lights", "lens flare"). '
    "Exclude ONLY: people and body parts (arm, hand, face, leg). "
    "If a brand, company, airline, or product name is clearly readable "
    "anywhere in the image (on a vehicle, aircraft, sign, building, or "
    "product), include the name itself, exactly as written, as its own "
    "entry — only when the letters are genuinely readable in THIS image; "
    "never guess or invent a name. "
    "If a landmark, building, monument, or company/product logo is clearly "
    "and confidently identifiable from its distinctive visual appearance — "
    "even with no readable text at all — include its specific name as its "
    "own entry, but only when you are genuinely certain; if you are not "
    "certain, do not guess. "
    "When one visual condition defines the entire image — such as a night "
    "scene, heavy blur or bokeh, underwater murkiness, fog, or visual "
    "static, noise, or film grain — include that condition itself as an "
    "entry. "
    "List up to 25 entries — more complete is better; include "
    "small, distant, background, and partially visible items. "
    "Use spaces not underscores. Each entry must be 1–3 words, the most specific noun. "
    "Never return an empty list unless the image is exclusively people against a plain/solid background "
    "with nothing else visible — landscapes, nature, close-ups of a single object, and abstract or "
    "blurred scenes must still yield at least one entry.\n\n"
    'Respond with ONLY this JSON format, double-quoted:\n'
    '{"objects": ["example item one", "example item two", "example item three"]}\n'
    "The words above are FORMAT placeholders only — never copy them into your answer unless they "
    "genuinely and specifically appear in THIS image."
)

# Stills-only variant (2026-07-09; rebalanced 2026-07-11 — design-kind lead
# clause + poster-is-always-a-design + title-priority, each harness-verified
# in isolation, golden 73→76% with COCO recall held at 78%),
# used by the stills path in tagger.py — NOT for
# video frames (OBJECT_PROMPT above is hand-tuned for those; keep them
# separate so stills tuning can't regress video detection). Still-image
# libraries hold logos, graphic designs, packaging, screenshots — content with
# no "physical objects" at all, on which OBJECT_PROMPT returns nouns like
# "text"/"letters" that tag_filter rightly junks, leaving zero tags
# (observed on LogoDet-3K samples: a flat wordmark got no tags at all).
# This variant asks for the design kind, readable brand/title text
# (searchable tags — the high-value case for logo/design libraries), and
# objects depicted in artwork. Same format-placeholder discipline as
# OBJECT_PROMPT (see the anchoring finding in docs/findings.md).
STILL_OBJECT_PROMPT = (
    "List every notable thing in this image — it may be a photograph, a logo, "
    "a graphic design, packaging, a poster, an illustration, a screenshot, or "
    "a document. "
    "If the image is a designed graphic rather than an ordinary photograph, "
    "the kind of design itself (logo, wordmark, poster, banner, packaging, "
    "label, advertisement, illustration, screenshot, document) must appear "
    "among your first entries — a movie poster, one-sheet, album cover, or "
    "book cover is ALWAYS a designed graphic and must yield the entry "
    '"poster" or "cover", even when its artwork looks photographic or '
    "painted; any image dominated by title text, credits, or a cast list is "
    "a poster. Never use the word photograph itself as an entry. Include: "
    "the exact words of any clearly readable brand, company, product, or "
    "title text printed in the image — copy the actual visible words "
    "themselves as entries; never write a category description instead of "
    "the words, and never invent words that are not actually visible; keep "
    "each name as its OWN separate entry — never combine a brand, a product "
    "name, and an object into one long entry; on a poster or cover, the "
    "main TITLE is the most important text of all and must be its own "
    "entry, exactly as written — read it even when it is drawn in large "
    "stylized, decorative, or logo-style lettering rather than plain "
    "print, and include it BEFORE any credits, names, or small print; "
    "physical objects shown, including objects depicted inside artwork "
    "(vehicles, clothing, buildings, food, drink, containers, tools, "
    "furniture, instruments, animals) and natural elements (water, rock, "
    "tree, plant, flower, sand, sky, cloud); "
    "signage, posters, neon signs, banners, logos, screens appearing inside "
    "a photo. "
    "If the image is out-of-focus, blurred, or a stylized visual effect "
    "(bokeh, lens flare, abstract lighting), still name whatever is "
    'identifiable AND include the effect itself as an entry (e.g. "bokeh '
    'effect", "blurred lights", "lens flare"). '
    "Exclude ONLY: people and body parts (arm, hand, face, leg). "
    "When one visual condition defines the entire image — such as a night "
    "scene, heavy blur or bokeh, underwater murkiness, fog, or visual "
    "static, noise, or film grain — include that condition itself as an "
    "entry. "
    "List up to 25 entries — more complete is better; include "
    "small, distant, background, and partially visible items. "
    "Use spaces not underscores. Each entry must be 1–3 "
    "words. Never return an empty list — a logo or design must at minimum "
    "yield its kind and its readable text.\n\n"
    'Respond with ONLY this JSON format, double-quoted:\n'
    '{"objects": ["example item one", "example item two", "example item three"]}\n'
    "The words above are FORMAT placeholders only — never copy them into "
    "your answer unless they genuinely and specifically appear in THIS image."
)


# --------------------------------------------------------------------------
# From summarize.py
# --------------------------------------------------------------------------

# Objectivity guardrail added 2026-07-18 (hands-on review): the caption/combine
# pipeline had no equivalent of IMAGE_CAPTION_PROMPT's "never add anything
# not actually visible" clause, and it showed — a real video summary called
# two women being interviewed on a deck "two friends", inventing a
# relationship the frames never established. Same anchoring/hallucination
# shape as the brand/landmark-guessing gotchas elsewhere in this project,
# just never patched at this call site. Fixed at BOTH stages (this prompt
# and the combine prompt in summarize_video) since a speculative per-frame
# caption would otherwise poison the final summary anyway.
CAPTION_PROMPT = (
    "Describe only what is visibly happening in this frame, in one short, "
    "concise sentence. Stick to neutral, observable facts — people, "
    "objects, actions, and setting — and do not infer relationships (such "
    "as friends, family, or coworkers), emotions, identities, or "
    "intentions that are not clearly evident from the image itself."
)

# Adaptive by image kind (2026-07-09): still-image libraries hold
# a huge variety — photos, logos, graphic designs, screenshots — and pure
# photo-composition language ("camera angle") forces nonsense on a flat
# logo. The model states the kind first, then describes what's relevant:
# composition for photographs, purpose/text/layout/colors for designed
# images. Tuned empirically against minicpm-v on photo/logo/poster test
# images; hard-won phrasing constraints:
# - "Sentence 1:/Sentence 2:" scaffolding and a "for a … library"
#   audience phrase both got copied verbatim into outputs (same anchoring
#   quirk as the detection prompt, see docs/findings.md) — structure must be
#   phrased as prose, audience framing kept out.
# - Shot type + camera angle are MANDATORY for photos (optional → the model
#   skips composition entirely on some images).
# - "never add words that are not shown" is what stopped it fabricating
#   text onto posters (it invented "to midnight" on an event poster before
#   this clause); "never guess place names/brand names" stopped a bridge
#   photo being confidently placed in Boston.
IMAGE_CAPTION_PROMPT = (
    "Describe this image in 2-4 short sentences of plain prose. Start by "
    "stating what kind of image it is: a photograph, a logo, a graphic "
    "design such as a poster or flyer, an illustration, a screenshot, or a "
    "document. If it is a photograph, describe the main subject and its "
    "setting, and always name the shot type (wide, medium, or close-up) "
    "and the camera angle, plus lighting and dominant colors when clearly "
    "identifiable. If it is a logo, graphic design, illustration, or "
    "document, always state what it is for, its dominant colors, and its "
    "overall style, and quote its visible text exactly as written — never "
    "add words that are not shown. If a landmark, building, monument, "
    "city skyline, or company/product logo is clearly and confidently "
    "identifiable from its distinctive appearance, name it specifically — "
    "but only when you are genuinely certain; if you are not certain, "
    "describe what is visible without guessing a specific name, place, or "
    "brand. Never add anything not actually visible in the image. "
    "No lists, no labels, "
    "no line breaks."
)


# Video-summary combine prompt. Formatted with
# .format(joined=..., length_instruction=...) at the call site.
#
# "Synthesize... skip minor/redundant" (added 2026-07-18, same day as
# the length-scaling revision) fixes a regression the objectivity
# clause caused: banning invented context also suppressed normal
# thematic compression ("a vibrant city at night") since that's itself
# a mild form of adding interpretation beyond the literal captions —
# the model fell back to literally enumerating every caption detail
# instead, chained with commas/semicolons into dense run-on sentences.
# This keeps the ban narrow (specific unsupported facts: relationships,
# emotions, motivations) while explicitly permitting/encouraging the
# abstraction that makes a short summary actually read like a summary.
VIDEO_SUMMARY_PROMPT = (
    "Here are captions from frames sampled across one video, in order:\n"
    "{joined}\n\n"
    "Write {length_instruction} summarizing what this video is about. "
    "Synthesize the captions into a natural, flowing summary of the "
    "main subject, setting, and action — you do not need to mention "
    "every object or repeated detail; skip minor or redundant ones "
    "rather than listing them all. Use neutral, factual language, and "
    "do not invent relationships (such as friends, family, or "
    "coworkers), emotions, or motivations that the captions don't "
    "support."
)


# ==========================================================================
# MiniCPM-V 4.7 35B-A3B prompt set (2026-10-15/16 study,
# docs/research/minicpm-v4.7-evaluation.md)
# ==========================================================================
# Built FROM the 4.5 prompts above by exact clause replacement (_sub asserts
# each clause occurs once), so every difference is named and measured here.
# Same rules as above: no concrete example names, structure as prose, no
# audience framing.
#
# Why 4.7 needs its own set: it reads "only when you are genuinely certain"
# far more strictly than 4.5 — city/vehicle naming collapsed (positives
# 6/21) even though it knows the names when asked directly. Name-first,
# recognition-gated wording (D4) restored them (17/21) with the curated
# false-guess traps still at 0; at scale ~0.7% of hard architecture images
# got a wrong city. 4.7 also copies whole headlines into tag entries,
# overflowing the token budget (T1 + the parser's truncation salvage).

def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"clause not found exactly once: {old[:60]!r}"
    return text.replace(old, new)


# Stills description, design clause (D1): describe what is pictured FIRST —
# 4.7 otherwise spent the whole description on a poster's typography.
_DESIGN_45 = ("If it is a logo, graphic design, illustration, or "
              "document, always state what it is for, its dominant colors, and its "
              "overall style, and quote its visible text exactly as written — never "
              "add words that are not shown. ")
_DESIGN_47 = ("If it is a logo, graphic design, illustration, or document, first "
              "describe what is actually pictured — the people, objects, vehicles, "
              "scenery and action — then its dominant colors and overall style, and "
              "finally quote its title, its tagline, and the names in its credits "
              "exactly as written — never add words that are not shown. ")

# Stills description, naming clause (D4 + D10's real-person guard). D10 was
# the only wording with ZERO real people named from their faces (0/19 vs
# 3-5/19 for the others, 4/19 for production 4.5) while still naming
# drawn/printed fictional characters (6/16). Names PRINTED in the image (a
# caption, a logo) still come through — that is reading, not recognition.
_NAMING_45 = ("If a landmark, building, monument, "
              "city skyline, or company/product logo is clearly and confidently "
              "identifiable from its distinctive appearance, name it specifically — "
              "but only when you are genuinely certain; if you are not certain, "
              "describe what is visible without guessing a specific name, place, or "
              "brand. ")
_NAMING_47 = ("When you recognize the specific place, city, landmark, vehicle make "
              "or model, fictional character, or brand from its distinctive "
              "appearance, use its proper name in the first sentence instead of a "
              "generic description; if you do not recognize it, describe it "
              "generically without guessing. Never identify a real person in a "
              "photograph from their face or appearance — do not name actors, "
              "celebrities, or public figures — but a fictional character shown in a "
              "poster, illustration, artwork, or animation may be named by the "
              "character's name. Even when you recognize a real person, describe them "
              "only by what is visible, such as clothing, expression, and action, and "
              "never by name unless their name is printed in the image. ")

# Long-form stills descriptions (D7, 2026-10-16): 4-8 sentences instead of 2,
# quoting title/tagline/credits. On film posters it covered every visible
# element without inventing plot facts that aren't on the poster. Needs summarize's larger still
# num_predict (text-dense posters came close to 512 tokens).
_LENGTH_45 = "Describe this image in 2-4 short sentences of plain prose."
_LENGTH_47 = "Describe this image in 4-8 sentences of plain prose."

# Tags (T1): hard brevity — 4.7 copied whole headlines on screenshots,
# overflowed the budget, and the truncated JSON parsed to nothing.
_STILL_LEN_45 = ("List up to 25 entries — more complete is better; include small, "
                 "distant, background, and partially visible items. Use spaces not "
                 "underscores. Each entry must be 1–3 words. ")
_STILL_LEN_47 = ("List up to 25 entries — more complete is better; include small, "
                 "distant, background, and partially visible items, and stop after 25. "
                 "Use spaces not underscores. Each entry must be 1–3 words: never copy "
                 "a whole sentence, headline, or paragraph — for long text keep only "
                 "the key name or title. ")
_VIDEO_LEN_45 = ("List up to 25 entries — more complete is better; include small, "
                 "distant, background, and partially visible items. Use spaces not "
                 "underscores. Each entry must be 1–3 words, the most specific noun. ")
_VIDEO_LEN_47 = ("List up to 25 entries — more complete is better; include small, "
                 "distant, background, and partially visible items, and stop after 25. "
                 "Use spaces not underscores. Each entry must be 1–3 words, the most "
                 "specific noun: never copy a whole sentence or headline — for long "
                 "text keep only the key name or title. ")

# Video captions (C4): the D4 naming wording for per-frame captions, minus
# fictional characters — video frames are mostly real people.
_CAPTION_TAIL_47 = (" When you recognize the specific place, city, landmark, vehicle "
                    "make or model, or brand from its distinctive appearance, use its "
                    "proper name instead of a generic description; if you do not "
                    "recognize it, describe it generically without guessing. Never "
                    "identify a real person from their face or appearance.")

# Video summary (S1): the combine step otherwise dropped the names the
# captions found (on-screen names, places). Video certification:
# all 21 golden video entries pass; no real-person naming on a 7-clip panel.
_SUMMARY_45 = "Use neutral, factual language, and do not invent relationships"
_SUMMARY_47 = ("Keep the specific names the captions give — places, landmarks, brands, "
               "vehicle models, and on-screen titles — since those are the most useful "
               "details to keep. Use neutral, factual language, and do not invent "
               "relationships")

PROMPT_SETS = {
    "minicpm-v4.5": {n: globals()[n] for n in PROMPT_NAMES},
}
PROMPT_SETS["minicpm-v4.7"] = {
    "SYSTEM_PROMPT": SYSTEM_PROMPT,
    "OBJECT_PROMPT": _sub(OBJECT_PROMPT, _VIDEO_LEN_45, _VIDEO_LEN_47),
    "STILL_OBJECT_PROMPT": _sub(STILL_OBJECT_PROMPT, _STILL_LEN_45, _STILL_LEN_47),
    "CAPTION_PROMPT": CAPTION_PROMPT + _CAPTION_TAIL_47,
    "IMAGE_CAPTION_PROMPT": _sub(_sub(_sub(IMAGE_CAPTION_PROMPT,
                                           _DESIGN_45, _DESIGN_47),
                                      _NAMING_45, _NAMING_47),
                                 _LENGTH_45, _LENGTH_47),
    "VIDEO_SUMMARY_PROMPT": _sub(VIDEO_SUMMARY_PROMPT, _SUMMARY_45, _SUMMARY_47),
}
