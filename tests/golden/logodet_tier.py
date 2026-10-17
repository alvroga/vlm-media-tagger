"""LogoDet-3K obscure-brand false-guess-rate tier (bulk statistical
validation, not hand-curated golden entries — see logodet/freeze_sample.py's
docstring for why this is a separate tier from tests/golden/entries/
logo-real-obscure-*).

Scales up ONE specific finding from the 2026-07-11 landmark/brand naming
investigation with much stronger statistical weight: does the confidence-
gated naming policy (shipped that day, in both OBJECT_PROMPT/
STILL_OBJECT_PROMPT tags and IMAGE_CAPTION_PROMPT descriptions) ever
hallucinate a specific FAMOUS brand name on a genuinely obscure/unfamiliar
logo — the "sees a triangle, says Volkswagen" risk. The original evidence
was 18 obscure logos x 3 runs (54 generations); this tier runs the FULL
stills path (tags via 2-pass detection, matching DETECT_PASSES, PLUS one
description call, as tagger.py does for a still) over 450 NEW obscure logos, one run each — breadth over depth,
since the question is "does this ever happen across a wide sample," not
"what's the per-image variance."

    python tests/golden/logodet_tier.py                # full frozen sample
    python tests/golden/logodet_tier.py --limit 20     # smoke run
    python tests/golden/logodet_tier.py --save
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

# Same broad "commonly over-pattern-matched" brand list as the curated
# logo-real-obscure-* entries (2026-07-11), extended slightly — none of
# these brands are actually present in any image in this sample.
FAMOUS_TRAP = [
    "nike", "google", "samsung", "amazon", "microsoft", "volkswagen",
    "starbucks", "pepsi", "gucci", "chanel", "louis vuitton", "mercedes",
    "audi", "ford", "honda", "sony", "ikea", "walmart", "disney", "netflix",
    "apple", "coca-cola", "coca cola", "mcdonald", "bmw", "toyota",
    "adidas", "asus", "adobe",
    # extended for this larger run
    "reebok", "puma", "levis", "levi's", "calvin klein", "north face",
    "under armour", "new balance", "converse", "prada", "versace",
    "rolex", "hermes", "burberry", "dior", "armani", "fedex", "ups", "dhl",
    "shell", "bp", "exxon", "nestle", "kelloggs", "heinz",
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
    wrong here: with a multi-word trap entry ("new balance"), it flags any
    generic tag that's a substring word of it ("new") as a violation. This
    is why "new"→"new balance" false-flagged hot-pockets.jpg/cheesybite.jpg/
    burts-bees.jpg in the 2026-07-11/12 run (caught by manual
    audit, not by the script) — same latent bug found and fixed properly in
    landmark_tier.py 2026-07-12, applied here too for future re-runs."""
    return re.search(rf"\b{re.escape(phrase)}\b", haystack) is not None


def find_violations(tags: list[str], desc: str) -> list[str]:
    hits = []
    for t in tags:
        t_norm = re.sub(r"[^\w\s'-]", " ", t.lower())
        for brand in FAMOUS_TRAP:
            if _contains_phrase(t_norm, brand):
                hits.append(f"tag:{t!r} matches {brand!r}")
    desc_norm = re.sub(r"[^\w\s'-]", " ", desc.lower())
    for brand in FAMOUS_TRAP:
        if _contains_phrase(desc_norm, brand):
            hits.append(f"description matches {brand!r}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import config as _config
    model_name = args.model or _config.OLLAMA_VISION_MODEL

    frozen = json.loads((GOLDEN / "logodet" / "expected.json").read_text())
    items = list(frozen["images"].items())
    if args.limit:
        items = items[: args.limit]

    print(f"LogoDet obscure-tier: {len(items)} images, model {model_name}", flush=True)
    t0 = time.time()
    violations_total = 0
    per_image = {}
    for n, (fname, info) in enumerate(items, 1):
        tags, desc = process(GOLDEN / "logodet" / "images" / fname, args.model)
        viol = find_violations(tags, desc)
        violations_total += len(viol)
        per_image[fname] = {"brand": info["brand"], "tags": tags,
                            "description": desc, "violations": viol}
        if viol:
            print(f"  ⚠ [{n}/{len(items)}] {fname} (actual: {info['brand']!r}): "
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
    print(f"{len(items)} images, {violations_total} false-brand-guess "
          f"violations ({violations_total / len(items):.1%} of images), "
          f"{elapsed:.1f} min")
    if violations_total == 0:
        print("ZERO false guesses — no hallucinated famous-brand names on "
              "any obscure logo in this sample.")

    if args.save:
        safe_model = re.sub(r"[^\w.-]", "_", model_name)
        out = GOLDEN / "results" / f"logodet-{date.today()}-{safe_model}.json"
        out.write_text(json.dumps({
            "date": str(date.today()), "model": model_name,
            "images": len(items), "violations": violations_total,
            "violation_rate": round(violations_total / len(items), 4),
            "per_image": per_image,
        }, indent=1, default=list) + "\n")
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
