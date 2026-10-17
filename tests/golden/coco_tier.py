"""COCO val2017 anti-overfit breadth tier (breadth check beyond the curated golden set).

Runs the production stills detection path (STILL_OBJECT_PROMPT, one call,
one retry on an empty result, clean_tags) over the frozen 200-image sample
in coco/images/ and scores recall of the human-labeled COCO categories.

    python tests/golden/coco_tier.py                # full frozen sample (~25 min)
    python tests/golden/coco_tier.py --limit 20     # smoke run
    python tests/golden/coco_tier.py --save         # results/coco-<date>-<model>.json
    python tests/golden/coco_tier.py --baseline results/coco-2026-07-11-….json

THE POINT: nothing here is ever tuned against. The golden entries steer
prompt work; this tier only answers "did we teach to the test?" at
certification time. If golden scores rise while COCO recall falls, a
prompt change memorized the golden set instead of getting better.
Consequences: the sample is frozen (seeded, committed — see
coco/freeze_sample.py), COCO_SYNONYMS below is frozen (widen it ONLY for
a genuine vocabulary mismatch, never to lift a score), and 'person' is
excluded by design — this pipeline deliberately never tags people.

Scoring: a category counts as expected when its largest instance is at
least MIN_AREA px² (32² — COCO's own small/medium boundary; a VLM tagger
isn't judged on thumb-sized objects), and additionally as "prominent"
at PROMINENT_AREA px² (64²). Recall is micro-averaged over labels.
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

from detect_objects_vlm import STILL_OBJECT_PROMPT, _call_vlm  # noqa: E402
from harness import _matches  # noqa: E402  (same word-boundary matching as tags)
from tag_filter import clean_tags  # noqa: E402

MIN_AREA = 32 * 32
PROMINENT_AREA = 64 * 64

# FROZEN vocabulary bridge, COCO category -> our tag vocabulary.
# Widen only for a real synonym the VLM legitimately uses ("couch"/"sofa"),
# NEVER to make a number look better. 'person' is deliberately absent.
COCO_SYNONYMS = {
    "airplane": "airplane|plane|aircraft|jet|airliner",
    "apple": "apple",
    "backpack": "backpack|rucksack|knapsack",
    "banana": "banana",
    "baseball bat": "baseball bat|bat",
    "baseball glove": "baseball glove|glove|mitt",
    "bear": "bear",
    "bed": "bed|mattress",
    "bench": "bench",
    "bicycle": "bicycle|bike",
    "bird": "bird|seagull|pigeon|duck|goose|parrot|crow|sparrow",
    "boat": "boat|ship|sailboat|yacht|canoe|kayak|ferry",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl",
    "broccoli": "broccoli",
    "bus": "bus",
    "cake": "cake",
    "car": "car|automobile|taxi|sedan|suv",
    "carrot": "carrot",
    "cat": "cat|kitten",
    "cell phone": "cell phone|phone|smartphone|mobile phone",
    "chair": "chair|stool",
    "clock": "clock",
    "couch": "couch|sofa",
    "cow": "cow|cattle|bull|calf|ox",
    "cup": "cup|mug",
    "dining table": "dining table|table|desk",
    "dog": "dog|puppy",
    "donut": "donut|doughnut",
    "elephant": "elephant",
    "fire hydrant": "fire hydrant|hydrant",
    "fork": "fork",
    "frisbee": "frisbee|flying disc",
    "giraffe": "giraffe",
    "hair drier": "hair drier|hair dryer|hairdryer",
    "handbag": "handbag|purse|bag",
    "horse": "horse|pony",
    "hot dog": "hot dog|hotdog|sausage",
    "keyboard": "keyboard",
    "kite": "kite",
    "knife": "knife",
    "laptop": "laptop|notebook computer|computer",
    "microwave": "microwave",
    "motorcycle": "motorcycle|motorbike|moped|scooter",
    "mouse": "mouse",
    "orange": "orange",
    "oven": "oven|stove",
    "parking meter": "parking meter",
    "pizza": "pizza",
    "potted plant": "potted plant|plant|houseplant|flower pot|planter",
    "refrigerator": "refrigerator|fridge",
    "remote": "remote|remote control",
    "sandwich": "sandwich|burger|hamburger",
    "scissors": "scissors",
    "sheep": "sheep|lamb",
    "sink": "sink|washbasin|basin",
    "skateboard": "skateboard",
    "skis": "skis|ski",
    "snowboard": "snowboard",
    "spoon": "spoon",
    "sports ball": "sports ball|ball|soccer ball|basketball|baseball|"
                   "tennis ball|football|volleyball|rugby ball",
    "stop sign": "stop sign",
    "suitcase": "suitcase|luggage|briefcase",
    "surfboard": "surfboard",
    "teddy bear": "teddy bear|teddy|stuffed animal|stuffed toy|plush",
    "tennis racket": "tennis racket|racket|racquet",
    "tie": "tie|necktie|bow tie",
    "toaster": "toaster",
    "toilet": "toilet",
    "toothbrush": "toothbrush",
    "traffic light": "traffic light|traffic signal|stoplight",
    "train": "train|locomotive|tram",
    "truck": "truck|lorry|pickup|van",
    "tv": "tv|television|monitor|screen",
    "umbrella": "umbrella|parasol",
    "vase": "vase",
    "wine glass": "wine glass|glass|goblet",
    "zebra": "zebra",
}


def detect(img: Path, model: str | None) -> list[str]:
    """Production stills path: one call, one retry on empty, clean_tags."""
    tags = clean_tags(_call_vlm({"path": str(img)}, model=model,
                                prompt=STILL_OBJECT_PROMPT))
    if not tags:
        tags = clean_tags(_call_vlm({"path": str(img)}, model=model,
                                    prompt=STILL_OBJECT_PROMPT))
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override OLLAMA_VISION_MODEL")
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--baseline", default=None,
                    help="previous coco-*.json to diff recall against")
    args = ap.parse_args()

    import config as _config
    model_name = args.model or _config.OLLAMA_VISION_MODEL

    frozen = json.loads((GOLDEN / "coco" / "expected.json").read_text())
    items = list(frozen["images"].items())
    if args.limit:
        items = items[: args.limit]

    print(f"COCO tier: {len(items)} images, model {model_name}", flush=True)
    per_image, t0 = {}, time.time()
    stats = {"expected": 0, "hit": 0, "prom_expected": 0, "prom_hit": 0}
    per_cat: dict[str, list[int]] = {}  # name -> [hit, expected]
    for n, (iid, info) in enumerate(items, 1):
        tags = detect(GOLDEN / "coco" / "images" / info["file"], args.model)
        hits, misses = [], []
        for cat, area in info["categories"].items():
            if cat not in COCO_SYNONYMS or area < MIN_AREA:
                continue  # person (by design) or below visibility floor
            hit = any(_matches(COCO_SYNONYMS[cat], t) for t in tags)
            stats["expected"] += 1
            stats["hit"] += hit
            if area >= PROMINENT_AREA:
                stats["prom_expected"] += 1
                stats["prom_hit"] += hit
            c = per_cat.setdefault(cat, [0, 0])
            c[0] += hit
            c[1] += 1
            (hits if hit else misses).append(cat)
        per_image[iid] = {"file": info["file"], "tags": tags,
                          "hits": hits, "misses": misses}
        if n % 10 == 0 or n == len(items):
            print(f"  [{n}/{len(items)}] recall so far "
                  f"{stats['hit']}/{stats['expected']} "
                  f"({stats['hit'] / max(stats['expected'], 1):.0%}), "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    recall = stats["hit"] / max(stats["expected"], 1)
    prom = stats["prom_hit"] / max(stats["prom_expected"], 1)
    print(f"\n{'=' * 60}")
    print(f"category recall {stats['hit']}/{stats['expected']} ({recall:.0%})  |  "
          f"prominent (≥64²px) {stats['prom_hit']}/{stats['prom_expected']} "
          f"({prom:.0%})  |  {(time.time() - t0) / 60:.1f} min")
    print("\nworst categories (≥3 expected):")
    ranked = sorted(((h / e, h, e, c) for c, (h, e) in per_cat.items() if e >= 3))
    for rate, h, e, cat in ranked[:15]:
        print(f"  {cat:20s} {h}/{e} ({rate:.0%})")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        print(f"\nvs baseline {args.baseline} (model {base['model']}): "
              f"recall {base['recall']:.0%} → {recall:.0%}, "
              f"prominent {base['prominent_recall']:.0%} → {prom:.0%}")

    if args.save:
        safe_model = re.sub(r"[^\w.-]", "_", model_name)
        out = GOLDEN / "results" / f"coco-{date.today()}-{safe_model}.json"
        out.write_text(json.dumps({
            "date": str(date.today()), "model": model_name,
            "images": len(items), "min_area": MIN_AREA,
            "prominent_area": PROMINENT_AREA,
            "recall": round(recall, 4), "prominent_recall": round(prom, 4),
            "per_category": {c: {"hit": h, "expected": e}
                             for c, (h, e) in sorted(per_cat.items())},
            "per_image": per_image,
        }, indent=1) + "\n")
        print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
