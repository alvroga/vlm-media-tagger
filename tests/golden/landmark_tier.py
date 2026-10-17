"""Places365 contemporary-photo landmark/city false-guess-rate tier (bulk
statistical validation, not hand-curated golden entries — see
landmark/freeze_sample.py's docstring for why this is a separate tier from
tests/golden/entries/landmark-*).

Scales up the landmark/city-naming side of the 2026-07-11 confidence-gated
naming policy with much stronger statistical weight than the 10 curated
landmark-* entries: does the policy ever hallucinate a specific FAMOUS city
or landmark name on a generic, non-identifying architectural photo (a
stadium, a bridge, a palace, a downtown skyline with no distinguishing
signage) — the same "sees a shape, names something famous" risk
logodet_tier.py checks for brands. Runs the full production stills path
(tags via 2-pass detection, matching DETECT_PASSES, plus one description
call) over 300 Places365 images across 16 outdoor architecture/landmark
scene categories, one run each — breadth over depth.

    python tests/golden/landmark_tier.py                # full frozen sample
    python tests/golden/landmark_tier.py --limit 20     # smoke run
    python tests/golden/landmark_tier.py --save
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

GOLDEN = Path(__file__).parent
sys.path.insert(0, str(GOLDEN.parent.parent / "src"))

from detect_objects_vlm import DETECT_PASSES, STILL_OBJECT_PROMPT, _call_vlm  # noqa: E402
from summarize import IMAGE_CAPTION_PROMPT, _caption_frame, clean_description  # noqa: E402
from tag_filter import clean_tags  # noqa: E402

# Globally iconic city and landmark names — none of these are the actual
# identity of any image in this sample (Places365 gives scene CATEGORY
# labels like "skyscraper"/"palace"/"bridge", never a specific place), so
# any confident mention is a false guess. Cities picked to avoid common-word
# collisions (learned from logodet_tier.py's FAMOUS_TRAP audit: generic
# words like "new"/"apple" produced false hits) — no dictionary-word city
# names (e.g. no "Nice", "Reading", "Mobile").
FAMOUS_TRAP = [
    # cities
    "paris", "london", "tokyo", "new york", "rome", "venice", "dubai",
    "singapore", "hong kong", "shanghai", "beijing", "moscow", "berlin",
    "barcelona", "sydney", "cairo", "istanbul", "bangkok", "amsterdam",
    "prague", "vienna", "athens", "rio de janeiro", "los angeles",
    "san francisco", "chicago", "las vegas", "miami", "toronto", "mumbai",
    "seoul", "shenzhen", "kuala lumpur",
    # named landmarks
    "eiffel tower", "big ben", "colosseum", "golden gate bridge",
    "empire state building", "statue of liberty", "times square",
    "brooklyn bridge", "burj khalifa", "sydney opera house", "taj mahal",
    "great wall", "buckingham palace", "notre dame", "sagrada familia",
    "leaning tower of pisa", "christ the redeemer", "machu picchu",
    "angkor wat", "tower bridge", "london eye", "space needle",
    "willis tower", "cn tower", "petra", "kremlin", "louvre",
]


def process(img: Path, model: str | None) -> tuple[list[str], str]:
    """Real production stills path for one file: 2-pass detection
    (matching DETECT_PASSES), one retry on empty tags, one description
    call — same shape as tagger.process_file's still path."""
    union = []
    for _ in range(DETECT_PASSES):
        union += _call_vlm({"path": str(img)}, model=model, prompt=STILL_OBJECT_PROMPT)
    tags = clean_tags(union)
    if not tags:
        union = []
        for _ in range(DETECT_PASSES):
            union += _call_vlm({"path": str(img)}, model=model, prompt=STILL_OBJECT_PROMPT)
        tags = clean_tags(union)

    import config as _config
    m = model or _config.OLLAMA_VISION_MODEL
    desc = clean_description(_caption_frame(str(img), m, prompt=IMAGE_CAPTION_PROMPT))
    return tags, desc


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """One-directional whole-phrase containment: does `phrase` appear in
    `haystack`, word-boundaried? Deliberately NOT harness._matches's
    bidirectional check (also true when `haystack` is a substring of
    `phrase`) — that's correct for golden-entry satisfaction scoring but
    wrong here: with multi-word trap entries ("eiffel tower", "empire
    state building"), it would flag any generic single-word tag the model
    correctly produces on architecture photos ("tower", "building",
    "bridge", "palace") as a false-guess violation, since that word is a
    whole-word substring of the trap PHRASE. Found 2026-07-12 smoke-testing
    this tier — one bare "towers" tag flagged 5 different landmark names.
    logodet_tier.py's FAMOUS_TRAP mostly avoided this by being single-word
    brand names; this tier's landmark names are mostly multi-word, so the
    bug would dominate the results here without this fix."""
    return re.search(rf"\b{re.escape(phrase)}\b", haystack) is not None


def find_violations(tags: list[str], desc: str) -> list[str]:
    hits = []
    for t in tags:
        t_norm = re.sub(r"[^\w\s'-]", " ", t.lower())
        for place in FAMOUS_TRAP:
            if _contains_phrase(t_norm, place):
                hits.append(f"tag:{t!r} matches {place!r}")
    desc_norm = re.sub(r"[^\w\s'-]", " ", desc.lower())
    for place in FAMOUS_TRAP:
        if _contains_phrase(desc_norm, place):
            hits.append(f"description matches {place!r}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import config as _config
    model_name = args.model or _config.OLLAMA_VISION_MODEL

    frozen = json.loads((GOLDEN / "landmark" / "expected.json").read_text())
    items = list(frozen["images"].items())
    if args.limit:
        items = items[: args.limit]

    print(f"Landmark contemporary-tier: {len(items)} images, model {model_name}", flush=True)
    t0 = time.time()
    violations_total = 0
    per_image = {}
    for n, (fname, info) in enumerate(items, 1):
        tags, desc = process(GOLDEN / "landmark" / "images" / fname, args.model)
        viol = find_violations(tags, desc)
        violations_total += len(viol)
        per_image[fname] = {"category": info["category"], "tags": tags,
                            "description": desc, "violations": viol}
        if viol:
            print(f"  ⚠ [{n}/{len(items)}] {fname} (category: {info['category']!r}): "
                  f"{viol}", flush=True)
        if n % 10 == 0 or n == len(items):
            elapsed_sec = time.time() - t0
            elapsed = elapsed_sec / 60
            rate = violations_total / n
            sec_per_image = elapsed_sec / n
            remaining = (len(items) - n) * sec_per_image / 60
            eta = time.strftime("%H:%M", time.localtime(time.time() + remaining * 60))
            print(f"  [{n}/{len(items)}] {violations_total} violations so far "
                  f"({rate:.1%} of images) | {elapsed:.1f} min elapsed, "
                  f"~{remaining:.0f} min remaining (ETA {eta}), "
                  f"{sec_per_image:.1f}s/image", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 60}")
    print(f"{len(items)} images, {violations_total} false-place-guess "
          f"violations ({violations_total / len(items):.1%} of images), "
          f"{elapsed:.1f} min")
    if violations_total == 0:
        print("ZERO false guesses — no hallucinated famous city/landmark "
              "names on any generic scene in this sample.")

    if args.save:
        safe_model = re.sub(r"[^\w.-]", "_", model_name)
        out = GOLDEN / "results" / f"landmark-{date.today()}-{safe_model}.json"
        out.write_text(json.dumps({
            "date": str(date.today()), "model": model_name,
            "images": len(items), "violations": violations_total,
            "violation_rate": round(violations_total / len(items), 4),
            "per_image": per_image,
        }, indent=1, default=list) + "\n")
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
