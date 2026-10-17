"""One-shot: freeze a reproducible random sample of COCO val2017 INTO the
project. Images are COPIED here and the needed ground truth is extracted
into expected.json — the raw source (datasets/coco, project-local,
gitignored) may get deleted/replaced at any time and the tier must keep
working from the frozen sample alone (raw bulk datasets live in datasets/).

    python tests/golden/coco/freeze_sample.py            # 200 images, fixed seed
    python tests/golden/coco/freeze_sample.py --source datasets/coco --size 200

Refuses to overwrite an existing freeze (--force to override): results are
only comparable across time on the IDENTICAL subset. The seed is fixed so a
re-freeze from an intact COCO copy reproduces the same sample bit-for-bit.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260711  # frozen — never change (comparability of every future run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/coco",
                    help="COCO root holding annotations/ and val2017/")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing freeze (breaks comparability!)")
    args = ap.parse_args()

    out_json = HERE / "expected.json"
    img_dir = HERE / "images"
    if out_json.exists() and not args.force:
        raise SystemExit("expected.json already exists — the sample is frozen. "
                         "Use --force only if you accept breaking comparability "
                         "with every existing coco-*.json result.")

    src = Path(args.source).expanduser()
    inst = json.loads((src / "annotations" / "instances_val2017.json").read_text())

    cats = {c["id"]: c["name"] for c in inst["categories"]}
    images = {i["id"]: i for i in inst["images"]}

    # per image: category name -> largest instance area (px²) — scoring
    # applies its visibility thresholds later, the freeze stays raw
    gt: dict[int, dict[str, float]] = {}
    for ann in inst["annotations"]:
        if ann.get("iscrowd"):
            continue
        by_cat = gt.setdefault(ann["image_id"], {})
        name = cats[ann["category_id"]]
        by_cat[name] = max(by_cat.get(name, 0.0), ann["area"])

    # only images that actually have annotations, deterministic order + seed
    ids = sorted(gt.keys())
    sample = sorted(random.Random(SEED).sample(ids, args.size))

    img_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for iid in sample:
        info = images[iid]
        shutil.copy2(src / "val2017" / info["file_name"], img_dir / info["file_name"])
        entries[str(iid)] = {
            "file": info["file_name"],
            "width": info["width"],
            "height": info["height"],
            "categories": {k: round(v, 1) for k, v in sorted(gt[iid].items())},
        }

    out_json.write_text(json.dumps({
        "meta": {
            "source": "COCO val2017 (instances_val2017.json), frozen 2026-07-11",
            "seed": SEED,
            "size": args.size,
            "note": "Anti-overfit breadth tier — never tune any prompt against "
                    "these images or the synonym map in coco_tier.py.",
        },
        "images": entries,
    }, indent=1) + "\n")
    total = sum(len(e["categories"]) for e in entries.values())
    print(f"froze {len(entries)} images ({total} category labels) → {img_dir}")


if __name__ == "__main__":
    main()
