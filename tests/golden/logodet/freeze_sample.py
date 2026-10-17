"""One-shot: freeze a large, reproducible random sample of OBSCURE
LogoDet-3K brand logos INTO the project. Raw source lives in
datasets/LogoDet-3K (project-local, gitignored). The exact files of the
committed sample are listed in expected.json (`source_file`), so it can be
rebuilt from that list regardless of sampling changes.

This is a bulk STATISTICAL validation tier (large N, sampled, scored in
aggregate), NOT hand-curated golden entries like tests/golden/entries/
logo-real-* — those 26 (8 famous + 18 obscure) stay as small, permanent,
human-reviewed regression sentinels. This tier exists to put much
stronger statistical weight behind ONE specific finding from the
2026-07-11 landmark/brand naming investigation: zero false brand guesses
across 18 obscure logos x 3 runs. That's thin evidence for a policy we
shipped; this scales the obscure-tier sample size by ~25x.

    python tests/golden/logodet/freeze_sample.py             # 450 images, fixed seed
    python tests/golden/logodet/freeze_sample.py --size 450 --source datasets/LogoDet-3K

Deliberately excludes every brand already used in the curated
tests/golden/entries/logo-real-* set (both tiers) — this is NEW obscure
coverage, not a re-sample of brands already tested individually.
"""
import argparse
import json
import random
import zlib
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260712  # frozen — a new seed (not 20260711) so this sample doesn't
                  # just re-derive the same 18 obscure brands already curated


# Brands already used in tests/golden/entries/logo-real-* — excluded so
# this tier is genuinely NEW obscure coverage, not a re-sample.
ALREADY_CURATED = {
    "Electronic/Apple", "Food/coca cola", "Food/McDonald's", "Transportation/BMW",
    "Transportation/toyota", "Clothes/Adidas SB", "Electronic/ASUS",
    "Others/Adobe Systems",
    "Electronic/nixon watches-1", "Food/Caramac", "Clothes/pequignet",
    "Others/Mad River", "Food/Adrian's", "Clothes/rapha", "Sports/razor",
    "Food/wuyutai", "Necessities/La Aurora", "Food/Capri Sun",
    "Others/china railway group limited", "Clothes/seavees",
    "Necessities/brown jordan", "Others/stage stores inc", "Necessities/Lacto",
    "Food/Amora", "Necessities/toscanello", "Necessities/markor furnishings",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/LogoDet-3K")
    ap.add_argument("--size", type=int, default=450)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_json = HERE / "expected.json"
    img_dir = HERE / "images"
    if out_json.exists() and not args.force:
        raise SystemExit("expected.json already exists — the sample is frozen. "
                         "Use --force only if you accept breaking comparability "
                         "with any existing logodet-*.json result.")

    src = Path(args.source).expanduser()
    brand_dirs = sorted(d for d in src.glob("*/*") if d.is_dir())
    pool = [d for d in brand_dirs if str(d.relative_to(src)) not in ALREADY_CURATED]
    print(f"{len(brand_dirs)} total brand folders, {len(pool)} available "
          f"after excluding {len(ALREADY_CURATED)} already-curated")

    random.Random(SEED).shuffle(pool)
    sample = sorted(pool[: args.size], key=lambda d: str(d.relative_to(src)))

    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for d in sample:
        rel = str(d.relative_to(src))
        imgs = sorted(d.glob("*.jpg"))
        if not imgs:
            continue
        # zlib.crc32, not hash(): str hashes are salted per process, which made
        # this pick differ on every run.
        src_img = imgs[random.Random(SEED + zlib.crc32(rel.encode()) % 100000).randrange(len(imgs))]
        slug = d.name.lower().replace(" ", "-").replace("'", "").replace("&", "and")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")[:40]
        out_name = f"{slug}.jpg"
        # de-dup filenames (different categories can share a brand-name slug)
        n = 2
        while (img_dir / out_name).exists():
            out_name = f"{slug}-{n}.jpg"
            n += 1
        import shutil
        shutil.copy2(src_img, img_dir / out_name)
        manifest[out_name] = {"brand": d.name, "category": rel.split("/")[0],
                              "source_file": str(src_img.relative_to(src))}

    out_json.write_text(json.dumps({
        "meta": {
            "source": "LogoDet-3K (Guo et al.), frozen 2026-07-12",
            "seed": SEED, "size": len(manifest),
            "note": "Bulk false-guess-rate validation tier — obscure brands "
                    "only, scored in aggregate. NEVER individually curated; "
                    "see tests/golden/entries/logo-real-obscure-* for the "
                    "small hand-reviewed set this scales up.",
        },
        "images": manifest,
    }, indent=1) + "\n")
    print(f"froze {len(manifest)} images -> {img_dir}")


if __name__ == "__main__":
    main()
