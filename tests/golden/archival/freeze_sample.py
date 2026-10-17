"""One-shot: freeze a large, reproducible random sample of ARCHIVAL/
HISTORICAL landmark and building photos — a distinct golden tier for how
the VLM handles old photography (period B&W, tinted postcards,
documentation photos, engravings), separate from the contemporary-photo
landmark-naming tier in tests/golden/landmark/ (Places365).

Source: Google Landmarks Dataset v2, `index` split (all CC0 / public
domain per the GLDv2 README), downloaded once into datasets/ (git-ignored)
rather than scraped live. The category keyword lists below (GOOD/BAD)
keep architecture and landmarks and cut archaeological close-ups,
ethnographic engravings and the like. Even filtered, the split skews
archival/vintage in STYLE — useless as a contemporary landmark-naming set,
which is exactly what makes it a good test of "what do we get from a big
box of archive pictures". There is no scoring runner yet; the frozen
sample is the input for one.

    python tests/golden/archival/freeze_sample.py             # 300 images, fixed seed
    python tests/golden/archival/freeze_sample.py --size 300 --source datasets/gld-v2-index
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).parent
SEED = 20260712

GOOD = ("skyline", "cityscape", "downtown", "city", "town", "building", "tower",
        "bridge", "castle", "cathedral", "church", "palace", "square", "plaza",
        "harbor", "harbour", "port", "station", "stadium", "monument", "hall",
        "temple", "mosque", "opera", "skyscraper", "fortress", "abbey", "basilica")
BAD = ("cave", "grave", "cemetery", "sculpture", "statue", "mine", "tomb",
       "skeleton", "fossil", "interior", "painting", "engraving", "portrait",
       "nude", "museum", "art", "manuscript", "coin", "fresco", "mural",
       "archaeological", "excavation", "carving")

# Loose substring exclusion against the curated tests/golden/entries/landmark-*
# set (city/site names, not GLDv2 category names — matched case-insensitively
# as substrings) so this bulk tier is genuinely new coverage, not a re-sample
# of landmarks already individually tested.
ALREADY_CURATED = ("akron", "barcelona", "chicago", "denver", "eiffel",
                   "golden gate", "pittsburgh", "seattle", "sydney opera",
                   "toledo")


def load_category_map(meta_dir: Path) -> dict[str, str]:
    label_to_cat = {}
    with open(meta_dir / "index_label_to_category.csv") as f:
        for row in csv.DictReader(f):
            name = unquote(row["category"].rsplit("Category:", 1)[-1]).replace("_", " ")
            label_to_cat[row["landmark_id"]] = name

    img_to_cat = {}
    with open(meta_dir / "index_image_to_landmark.csv") as f:
        for row in csv.DictReader(f):
            cat = label_to_cat.get(row["landmark_id"])
            if cat:
                img_to_cat[row["id"]] = cat
    return img_to_cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/gld-v2-index")
    ap.add_argument("--meta", default="datasets/gld-v2-metadata")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_json = HERE / "expected.json"
    img_dir = HERE / "images"
    if out_json.exists() and not args.force:
        raise SystemExit("expected.json already exists — the sample is frozen. "
                         "Use --force only if you accept breaking comparability.")

    src = Path(args.source).expanduser()
    meta_dir = Path(args.meta).expanduser()
    img_to_cat = load_category_map(meta_dir)

    files = list(src.rglob("*.jpg"))
    pool = []
    for f in files:
        cat = img_to_cat.get(f.stem)
        if not cat:
            continue
        cl = cat.lower()
        if any(k in cl for k in BAD):
            continue
        if not any(k in cl for k in GOOD):
            continue
        if any(k in cl for k in ALREADY_CURATED):
            continue
        pool.append((f, cat))

    print(f"{len(files)} total images, {len(pool)} available after "
          f"category filtering + excluding already-curated landmarks")

    random.Random(SEED).shuffle(pool)
    sample = pool[: args.size]

    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for f, cat in sample:
        slug = cat.lower().replace(" ", "-").replace("'", "").replace("&", "and")
        slug = "".join(c for c in slug if c.isalnum() or c == "-").strip("-")[:50]
        out_name = f"{slug}.jpg"
        n = 2
        while (img_dir / out_name).exists():
            out_name = f"{slug}-{n}.jpg"
            n += 1
        shutil.copy2(f, img_dir / out_name)
        manifest[out_name] = {"category": cat, "source_image_id": f.stem}

    out_json.write_text(json.dumps({
        "meta": {
            "source": "Google Landmarks Dataset v2 (index split), category-"
                      "filtered, frozen 2026-07-12",
            "seed": SEED, "size": len(manifest),
            "note": "Bulk false-guess-rate validation tier (landmark/city "
                    "side) — category-filtered for architecture/landmark "
                    "relevance, scored in aggregate. NEVER individually "
                    "curated; see tests/golden/entries/landmark-* for the "
                    "small hand-reviewed set this scales up. Content skews "
                    "archival/vintage in style — see module docstring.",
        },
        "images": manifest,
    }, indent=1) + "\n")
    print(f"froze {len(manifest)} images -> {img_dir}")


if __name__ == "__main__":
    main()
