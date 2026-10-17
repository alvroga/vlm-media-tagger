"""One-shot: freeze a large, reproducible random sample of CONTEMPORARY
landmark/building/cityscape photos INTO the project, for the landmark/
city-naming side of the 2026-07-11 naming policy (false-guess-rate check,
mirrors tests/golden/logodet/freeze_sample.py's role for brands).

Source: Places365-Standard validation split (val_large — full resolution,
100 images per each of 365 scene categories), downloaded into
datasets/places365/ 2026-07-12 (project-local, gitignored — a single
~2.3GB sequential download plus a 67MB filelist archive, not scraped).
Chosen specifically for CONTEMPORARY photography: an earlier attempt using
Google Landmarks Dataset v2 turned out to skew archival/vintage in style
(old B&W documentation photos, tinted postcards) even after filtering for
topical relevance. Places365 is crowdsourced/web-scraped scene
photography, not archival, and comes with precise category ground truth
(no need to keyword-match noisy free-text category names). All Places365
images are usable under the terms of the Places2 dataset release (research/
non-commercial use — see http://places2.csail.mit.edu); this is internal
test-fixture use, not redistribution.

RELEVANT_CATEGORIES below is the outdoor architecture/landmark subset of
the 365 scene categories (bridges, towers, plazas, stadiums, places of
worship, downtown/skyscraper scenes, etc.) — indoor variants (e.g.
church/indoor) are deliberately excluded: we want images where a landmark/building is
actually the visible subject.

    python tests/golden/landmark/freeze_sample.py             # 300 images, fixed seed
    python tests/golden/landmark/freeze_sample.py --size 300 --source datasets/places365
"""
import argparse
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260712

RELEVANT_CATEGORIES = {
    "bridge", "castle", "church/outdoor", "downtown", "harbor",
    "mosque/outdoor", "palace", "plaza", "rope_bridge", "skyscraper",
    "stadium/baseball", "stadium/football", "stadium/soccer",
    "temple/asia", "tower", "water_tower",
}


def load_categories(meta_dir: Path) -> dict[int, str]:
    """{category_index: 'bridge' / 'stadium/baseball' / ...} (leading /<letter>/
    stripped from e.g. '/b/bridge')."""
    out = {}
    for line in (meta_dir / "categories_places365.txt").read_text().splitlines():
        path, idx = line.rsplit(" ", 1)
        name = path.split("/", 2)[-1]  # '/b/bridge' -> 'bridge'; '/c/church/outdoor' -> 'church/outdoor'
        out[int(idx)] = name
    return out


def load_val_labels(meta_dir: Path) -> dict[str, int]:
    out = {}
    for line in (meta_dir / "places365_val.txt").read_text().splitlines():
        fname, idx = line.split(" ")
        out[fname] = int(idx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/places365")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_json = HERE / "expected.json"
    img_dir = HERE / "images"
    if out_json.exists() and not args.force:
        raise SystemExit("expected.json already exists — the sample is frozen. "
                         "Use --force only if you accept breaking comparability.")

    src = Path(args.source).expanduser()
    meta_dir = src / "filelist"
    val_dir = src / "val_large" / "val_large"

    idx_to_name = load_categories(meta_dir)
    val_labels = load_val_labels(meta_dir)

    pool = []
    for fname, idx in val_labels.items():
        cat = idx_to_name.get(idx)
        if cat in RELEVANT_CATEGORIES:
            pool.append((fname, cat))

    print(f"{len(val_labels)} total validation images, {len(pool)} in "
          f"{len(RELEVANT_CATEGORIES)} relevant categories")

    random.Random(SEED).shuffle(pool)
    sample = pool[: args.size]

    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fname, cat in sample:
        slug = cat.replace("/", "-")
        out_name = f"{slug}-{fname}"
        shutil.copy2(val_dir / fname, img_dir / out_name)
        manifest[out_name] = {"category": cat, "source_file": fname}

    out_json.write_text(json.dumps({
        "meta": {
            "source": "Places365-Standard val_large, category-filtered, "
                      "frozen 2026-07-12",
            "seed": SEED, "size": len(manifest),
            "note": "Bulk false-guess-rate validation tier (landmark/city "
                    "side, CONTEMPORARY photography) — category-filtered "
                    "for outdoor architecture/landmark relevance, scored "
                    "in aggregate. NEVER individually curated; see "
                    "tests/golden/entries/landmark-* for the small "
                    "hand-reviewed set this scales up.",
        },
        "images": manifest,
    }, indent=1) + "\n")
    print(f"froze {len(manifest)} images -> {img_dir}")


if __name__ == "__main__":
    main()
