"""Prompt regression harness — runs the current prompts over the frozen
golden set N times and scores hit rates against each entry's expected.json.

    python tests/golden/harness.py                      # full set, 5 runs
    python tests/golden/harness.py --quick              # high-signal subset
    python tests/golden/harness.py --entries logo --runs 3
    python tests/golden/harness.py --model qwen3-vl:8b
    python tests/golden/harness.py --save               # results/<date>-<model>.json
    python tests/golden/harness.py --baseline results/2026-07-20-openbmb_minicpm-v4.5-p2.json
    python tests/golden/harness.py --canary             # 60s pre-measurement gate

Scoring (see README.md for the entry format):
  must_have    — must be satisfied in EVERY run (union of the entry's frame
                 tags per run). Any miss = hard FAIL.
  nice_to_have — fraction of runs satisfied; reported as %, judged by
                 movement against a --baseline, never pass/fail.
  must_not     — entry-specific: word-boundary match in any run's tags (or,
                 with --descriptions, in a still's description text) = FAIL.
  description_must_mention — with --descriptions, stills only: fraction of
                 runs whose description text mentions it; rate-scored like
                 nice_to_have (movement vs baseline, never pass/fail).
  globals      — person nouns / placeholder echoes / junk nouns must never
                 survive the pipeline's own defenses; exact-match on every
                 entry. A hit means a defense regressed, not the prompt.

  video-full entries (type: "video-full") freeze a WHOLE video's cuts.json
  + every extracted frame and run the REAL detect_objects_vlm()/
  summarize_video() unmodified — the only place shot-merging and video
  summaries get scored. must_have/nice_to_have/must_not/violations work as
  above; summary_must_mention mirrors description_must_mention (needs
  --descriptions); min_segment_counts is a hard per-run floor on how many
  DISCRETE segments a class must produce (catches _merge_segments
  collapsing a recurring object across shots into one giant segment).
  --passes is ignored for these (DETECT_PASSES is fixed inside
  detect_objects_vlm by design). See README.md.

--canary: one reliable entry (product-brand), one run, detection AND
description, diffed against the newest saved baseline for the model —
run it BEFORE any measurement session (and after ANY Ollama upgrade, see
docs/findings.md): catches contaminated environments (empty outputs, thinking-
mode flips, large slowdowns) in ~a minute. Exit code 0 = safe to measure.

Expectations describe POST-FILTER output (clean_tags), i.e. what would be
written to the output JSON. `a|b` in an expectation means "any of these".
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

GOLDEN = Path(__file__).parent
# Entries whose source media can't be redistributed live under private/ (git-ignored) with the same layout.
# They run alongside the public ones; their results are saved under
# private/results/ so a public baseline never carries their output.
PRIVATE = GOLDEN / "private"
sys.path.insert(0, str(GOLDEN.parent.parent / "src"))

from detect_objects_vlm import (  # noqa: E402
    _PERSON_NOUNS,
    OBJECT_PROMPT,
    STILL_OBJECT_PROMPT,
    _call_vlm,
    apply_deterministic_tags,
)
from tag_filter import clean_tags  # noqa: E402

# Defenses that must hold on every entry regardless of content. Exact-match
# (mirrors _call_vlm's own person-noun drop) so "hand truck" isn't flagged.
GLOBAL_MUST_NOT = set(_PERSON_NOUNS) | {
    "example item one", "example item two", "example item",
    "text", "background", "brand name",
}

# --quick: the highest-signal subset — one entry per major failure mode.
QUICK = [
    "night-cityscape-pan", "landmark-barcelona", "logo-real-obscure-amora",
    "bokeh-car", "nature-creek", "noise-frame", "portrait-still", "poster-bbb",
    "logo-flat", "multilingual-japanese",
]

# --canary: single fast control entry. product-brand: 1 frame, a hard
# must_have floor ('paperpro' — brand extraction, historically the first
# thing a strangled model loses), and it's a still, so one run covers the
# detection AND description code paths in two VLM calls.
CANARY = "product-brand"


def _norm(phrase: str) -> str:
    """Cheap word-wise singularization so 'skyscrapers' matches
    'skyscraper' (both sides get the same treatment, so imperfect stemming
    like 'glasses'→'glasse' still compares equal)."""
    words = []
    for w in phrase.lower().split():
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def _matches(expectation: str, tag: str) -> bool:
    """Word-boundary containment, either direction, any `|` alternative,
    plural-tolerant."""
    tag = _norm(tag.strip())
    for alt in expectation.split("|"):
        alt = _norm(alt.strip())
        if re.search(rf"\b{re.escape(alt)}\b", tag) or \
           re.search(rf"\b{re.escape(tag)}\b", alt):
            return True
    return False


def _satisfied(expectation: str, tags: list[str]) -> bool:
    return any(_matches(expectation, t) for t in tags)


def _mentions(expectation: str, text: str) -> bool:
    """Word-boundary search of any `|` alternative in description text,
    plural-tolerant (both sides get _norm's treatment). Punctuation is
    stripped from the text first so 'staplers,' still singularizes."""
    norm_text = _norm(re.sub(r"[^\w\s'-]", " ", text))
    return any(re.search(rf"\b{re.escape(_norm(alt.strip()))}\b", norm_text)
               for alt in expectation.split("|"))


def run_video_full_entry(entry_dir: Path, exp: dict, runs: int,
                          model: str | None, descriptions: bool) -> dict:
    """video-full entries freeze a WHOLE video's cuts.json + every extracted
    frame (not a hand-picked subset), so the REAL production functions —
    detect_objects_vlm.detect_objects_vlm() (real selection, multi-pass
    detection, _merge_segments) and summarize.summarize_video() — run
    UNMODIFIED against frozen data via a scratch config.DATA_DIR swap. This
    is the deliberate opposite tradeoff from video-frames entries (see
    README): those isolate the PROMPT from selection/merging; this isolates
    nothing — it's the only place shot-merging and video summaries get
    scored at all. DETECT_PASSES is NOT parameterizable here (it's a fixed
    module constant inside detect_objects_vlm, by design)."""
    import shutil
    import tempfile

    import config as _config
    from detect_objects_vlm import _select_shot_frames
    from detect_objects_vlm import detect_objects_vlm as _detect
    from summarize import MAX_SUMMARY_FRAMES
    from summarize import summarize_video as _summarize

    frames_src = entry_dir / "frames"
    cuts_path = entry_dir / "cuts.json"
    raw_index = json.loads((frames_src / "index.json").read_text())

    def _with_paths(base: Path) -> list[dict]:
        return [{**e, "path": str(base / e["filename"])} for e in raw_index]

    # frame count for display + summary-call budgeting only — the real
    # selection happens fresh inside _detect()/_summarize() per run below
    n_selected = len(_select_shot_frames(_with_paths(frames_src), cuts_path))
    n_summary_frames = min(n_selected, MAX_SUMMARY_FRAMES)

    video_id = entry_dir.name
    run_tags: list[list[str]] = []
    summary_texts: list[str] = []
    segment_counts_by_run: list[dict[str, int]] = []

    for _ in range(runs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            work_dir = tmp_path / video_id
            (work_dir / "frames").mkdir(parents=True)
            for e in raw_index:
                (work_dir / "frames" / e["filename"]).symlink_to(
                    frames_src / e["filename"])
            (work_dir / "frames" / "index.json").write_text(
                json.dumps(_with_paths(work_dir / "frames")))
            shutil.copy(cuts_path, work_dir / "cuts.json")

            orig_data_dir = _config.DATA_DIR
            _config.DATA_DIR = tmp_path
            try:
                segments = _detect(video_id, model=model)
                counts: dict[str, int] = {}
                for s in segments:
                    counts[s["class"]] = counts.get(s["class"], 0) + 1
                segment_counts_by_run.append(counts)
                run_tags.append(clean_tags(sorted(set(counts))))

                if descriptions:
                    summary_texts.append(_summarize(video_id, model=model) or "")
            finally:
                _config.DATA_DIR = orig_data_dir

    must_misses = []
    for i, tags in enumerate(run_tags):
        for m_ in exp["must_have"]:
            if not _satisfied(m_, tags):
                must_misses.append((m_, i))
    # shot-merging sentinel: a class collapsing toward too few segments in
    # ANY run is the one thing video-frames entries structurally cannot
    # test (they freeze a hand-picked subset, not real per-shot detections
    # + boundaries) — folded into must_misses so it fails the run the same
    # way and reuses the existing print path.
    for i, counts in enumerate(segment_counts_by_run):
        for cls, floor in exp.get("min_segment_counts", {}).items():
            got = counts.get(cls, 0)
            if got < floor:
                must_misses.append(
                    (f"{cls}: only {got} segment(s), need ≥{floor} "
                     "(shot-merging regression)", i))

    nice_rates = {n: sum(_satisfied(n, tags) for tags in run_tags) / runs
                  for n in exp["nice_to_have"]}

    desc_rates = {}
    if summary_texts:
        desc_rates = {m_: sum(_mentions(m_, t) for t in summary_texts) / len(summary_texts)
                      for m_ in exp.get("summary_must_mention", [])}

    violations = []
    for i, tags in enumerate(run_tags):
        for t in tags:
            if t in GLOBAL_MUST_NOT:
                violations.append((f"GLOBAL:{t}", i))
            for mn in exp["must_not"]:
                if _matches(mn, t):
                    violations.append((f"{mn} (tag: {t})", i))
    for text in summary_texts:
        for mn in exp["must_not"]:
            for alt in mn.lower().split("|"):
                if re.search(rf"\b{re.escape(alt)}\b", text.lower()):
                    violations.append((f"{mn} (summary)", -1))

    calls = runs * n_selected * 2  # DETECT_PASSES, fixed inside detect_objects_vlm
    if summary_texts:
        calls += runs * (n_summary_frames + 1)  # + one combine call per run

    return {
        "sentinel_for": exp["sentinel_for"],
        "frames": n_selected,
        "calls": calls,
        "run_tags": run_tags,
        "must_have": exp["must_have"],
        "must_misses": must_misses,
        "nice_rates": nice_rates,
        "desc_rates": desc_rates,
        "violations": violations,
        "descriptions": summary_texts,
        "passed": not must_misses and not violations,
        "kind": "video-full",
        "segment_counts": segment_counts_by_run,
    }


def run_entry(entry_dir: Path, runs: int, model: str | None,
              descriptions: bool, passes: int = 1) -> dict:
    exp = json.loads((entry_dir / "expected.json").read_text())
    if exp.get("type") == "video-full":
        if passes != 1:
            print(f"  (note: --passes ignored for {entry_dir.name} — "
                  "video-full entries always run detect_objects_vlm's own "
                  "fixed DETECT_PASSES)")
        return run_video_full_entry(entry_dir, exp, runs, model, descriptions)

    frames = sorted((entry_dir / "frames").glob("*.jpg"))
    prompt = STILL_OBJECT_PROMPT if exp.get("prompt") == "still" else OBJECT_PROMPT

    run_tags: list[list[str]] = []
    desc_texts: list[str] = []
    for _ in range(runs):
        union: list[str] = []
        for f in frames:
            # multi-pass detection experiment: K calls per frame, unioned —
            # temp-0.2 variance becomes recall for ~linear cost
            frame_nouns: list[str] = []
            for _p in range(passes):
                for n in _call_vlm({"path": str(f)}, model=model, prompt=prompt):
                    if n not in frame_nouns:
                        frame_nouns.append(n)
            # deterministic (non-VLM) tags, once per frame — mirrors
            # detect_objects_vlm._process_frame exactly, so scored rates
            # reflect real pipeline output
            union += apply_deterministic_tags(frame_nouns, str(f))
        run_tags.append(clean_tags(union))
    if descriptions and exp.get("prompt") == "still":
        # one description per run so description_must_mention gets real
        # hit-rates, through the same post-filter the pipeline writes through
        import config as _config
        from summarize import IMAGE_CAPTION_PROMPT, _caption_frame, clean_description
        m = model or _config.OLLAMA_VISION_MODEL
        desc_texts = [clean_description(
                          _caption_frame(str(frames[0]), m,
                                         prompt=IMAGE_CAPTION_PROMPT))
                      for _ in range(runs)]

    must_misses = []   # (expectation, run_index)
    for i, tags in enumerate(run_tags):
        for m_ in exp["must_have"]:
            if not _satisfied(m_, tags):
                must_misses.append((m_, i))

    nice_rates = {n: sum(_satisfied(n, tags) for tags in run_tags) / runs
                  for n in exp["nice_to_have"]}

    desc_rates = {}
    if desc_texts:
        desc_rates = {m_: sum(_mentions(m_, t) for t in desc_texts) / len(desc_texts)
                      for m_ in exp.get("description_must_mention", [])}

    violations = []
    for i, tags in enumerate(run_tags):
        for t in tags:
            if t in GLOBAL_MUST_NOT:
                violations.append((f"GLOBAL:{t}", i))
            for mn in exp["must_not"]:
                if _matches(mn, t):
                    violations.append((f"{mn} (tag: {t})", i))
    for text in desc_texts:
        for mn in exp["must_not"]:
            for alt in mn.lower().split("|"):
                if re.search(rf"\b{re.escape(alt)}\b", text.lower()):
                    violations.append((f"{mn} (description)", -1))

    # VLM calls made, for per-call timing comparisons (canary): detection
    # is runs × frames × passes, plus one description call per run if run.
    calls = runs * len(frames) * passes + (runs if desc_texts else 0)

    return {
        "sentinel_for": exp["sentinel_for"],
        "frames": len(frames),
        "calls": calls,
        "run_tags": run_tags,
        "must_have": exp["must_have"],
        "must_misses": must_misses,
        "nice_rates": nice_rates,
        "desc_rates": desc_rates,
        "violations": violations,
        "descriptions": desc_texts,
        "passed": not must_misses and not violations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default=None, help="override OLLAMA_VISION_MODEL")
    ap.add_argument("--entries", default=None,
                    help="substring filter on entry names; comma-separated "
                         "alternatives (e.g. 'poster,logo,aerial')")
    ap.add_argument("--quick", action="store_true",
                    help=f"only the high-signal subset: {', '.join(QUICK)}")
    ap.add_argument("--passes", type=int, default=1,
                    help="VLM calls per frame per run, tags unioned "
                         "(multi-pass detection experiment)")
    ap.add_argument("--descriptions", action="store_true",
                    help="also run still descriptions: score "
                         "description_must_mention rates, check must_not text")
    ap.add_argument("--save", action="store_true",
                    help="write results/<date>-<model>.json")
    ap.add_argument("--baseline", default=None,
                    help="previous results JSON to diff nice_to_have rates against")
    ap.add_argument("--canary", action="store_true",
                    help=f"pre-measurement gate: run '{CANARY}' once (detection "
                         "+ description) vs the newest saved baseline; exit 0 "
                         "= environment sane, safe to measure")
    args = ap.parse_args()

    import config as _config
    model_name = args.model or _config.OLLAMA_VISION_MODEL

    if args.canary:
        args.entries, args.quick, args.runs = CANARY, False, 1
        args.descriptions = True
        if args.save:
            print("(--save ignored in canary mode — a 1-entry file must "
                  "never become a baseline)")
            args.save = False

    dirs = sorted((d for root in (GOLDEN / "entries", PRIVATE / "entries")
                   if root.is_dir() for d in root.iterdir()
                   if (d / "expected.json").exists()), key=lambda d: d.name)
    if args.quick:
        dirs = [d for d in dirs if d.name in QUICK]
    if args.entries:
        subs = [s.strip() for s in args.entries.split(",") if s.strip()]
        dirs = [d for d in dirs if any(s in d.name for s in subs)]
    if not dirs:
        sys.exit("no entries matched")

    if args.canary:
        # Warm-up call so the timing comparison below measures a warm model
        # (baseline entries always ran warm mid-set); a cold first call
        # includes model load and would false-alarm the slowdown check.
        print("(canary: warming up model…)", flush=True)
        warm_frame = sorted((dirs[0] / "frames").glob("*.jpg"))[0]
        _call_vlm({"path": str(warm_frame)}, model=args.model,
                  prompt=STILL_OBJECT_PROMPT)

    print(f"{len(dirs)} entries × {args.runs} runs × {args.passes} pass(es), "
          f"model {model_name}\n", flush=True)
    results, t0 = {}, time.time()
    for d in dirs:
        t1 = time.time()
        r = run_entry(d, args.runs, args.model, args.descriptions, args.passes)
        r["seconds"] = round(time.time() - t1, 1)
        results[d.name] = r
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {d.name}  ({r['frames']} frames, "
              f"{r['seconds']:.0f}s)", flush=True)
        for m_, i in r["must_misses"]:
            print(f"       MISS must_have '{m_}' in run {i + 1}")
        for v, i in r["violations"]:
            where = "description" if i == -1 else f"run {i + 1}"
            print(f"       VIOLATION {v} ({where})")
        for n, rate in r["nice_rates"].items():
            print(f"       nice '{n}': {rate:.0%}")
        desc_label = "summary" if r.get("kind") == "video-full" else "desc"
        for n, rate in r["desc_rates"].items():
            print(f"       {desc_label} '{n}': {rate:.0%}")
        for text in r["descriptions"]:
            print(f"       » {text}")

    failed = [n for n, r in results.items() if not r["passed"]]
    all_nice = [v for r in results.values() for v in r["nice_rates"].values()]
    mean_nice = sum(all_nice) / len(all_nice) if all_nice else 0.0
    all_desc = [v for r in results.values() for v in r["desc_rates"].values()]
    print(f"\n{'=' * 60}")
    print(f"{len(results) - len(failed)}/{len(results)} entries passed, "
          f"mean nice_to_have hit-rate {mean_nice:.0%}, "
          + (f"mean description hit-rate {sum(all_desc) / len(all_desc):.0%}, "
             if all_desc else "")
          + f"{(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"FAILED: {', '.join(failed)}")

    baseline_path = args.baseline
    if args.canary and not baseline_path:
        # newest saved full-set results for this model (≥5 entries so a
        # stray subset run can never become the reference)
        cands = []
        for p in sorted((GOLDEN / "results").glob("*.json")):
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # results/ can also hold one-off STUDY output that isn't a
            # harness run (a bare JSON list, or a dict with an `entries`
            # LIST), which used to crash --canary after it had already
            # passed (found 2026-07-30). Hence the schema check below: a usable baseline is a real
            # harness run, i.e. `runs` plus an `entries` MAPPING of per-entry
            # results. `len()` alone can't tell a mapping from a list.
            if not isinstance(data, dict) or "runs" not in data:
                continue
            entries = data.get("entries")
            if not isinstance(entries, dict) or len(entries) < 5:
                continue
            if data.get("model") == model_name:
                cands.append((data.get("date", ""), p))
        baseline_path = str(max(cands)[1]) if cands else None
        if not baseline_path:
            print(f"\n(no saved baseline found for model {model_name} — "
                  "pass/violation checks only; run a full --save first)")

    base = None
    if baseline_path:
        base = json.loads(Path(baseline_path).read_text())
        twin = PRIVATE / "results" / Path(baseline_path).name
        if twin.exists():
            base["entries"].update(json.loads(twin.read_text())["entries"])
        print(f"\nvs baseline {baseline_path} "
              f"(model {base['model']}, {base['runs']} runs):")
        for name, r in results.items():
            b = base["entries"].get(name)
            if not b:
                continue
            if b["passed"] != r["passed"]:
                print(f"  {name}: {'PASS→FAIL  ⚠' if b['passed'] else 'FAIL→PASS'}")
            for key, label in (("nice_rates", ""), ("desc_rates", "desc ")):
                for n, rate in r[key].items():
                    brate = b.get(key, {}).get(n)
                    if brate is not None and abs(rate - brate) >= 0.005:
                        arrow = "↑" if rate > brate else "↓"
                        print(f"  {name} {label}'{n}': {brate:.0%} → {rate:.0%} {arrow}")

    if args.save:
        out_dir = GOLDEN / "results"
        out_dir.mkdir(exist_ok=True)
        safe_model = re.sub(r"[^\w.-]", "_", model_name)
        suffix = f"-p{args.passes}" if args.passes > 1 else ""
        # A subset run (--entries/--quick) must never collide with the
        # full-set filename — that silently overwrote the certified
        # baseline THREE times in one session (git-restored each time).
        # Full runs keep the plain name so existing --baseline references
        # in docs/scripts don't rot.
        if args.entries or args.quick:
            scope = re.sub(r"[^\w.-]", "_", args.entries or "quick")[:40]
            suffix += f"-{scope}"
        out = out_dir / f"{date.today()}-{safe_model}{suffix}.json"
        if out.exists() and not (args.entries or args.quick):
            print(f"\n(overwriting existing full-set baseline {out} — "
                  "make sure this is intentional)")
        private = {d.name for d in dirs if d.parent.parent == PRIVATE}
        for path, subset in (
                (out, {n: r for n, r in results.items() if n not in private}),
                (PRIVATE / "results" / out.name,
                 {n: r for n, r in results.items() if n in private})):
            if not subset:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "date": str(date.today()), "model": model_name,
                "runs": args.runs, "passes": args.passes,
                "entries": subset,
            }, indent=2, default=list))
            print(f"\nsaved → {path}")

    if args.canary:
        r = results[CANARY]
        problems = []
        if r["must_misses"] or r["violations"]:
            problems.append("must_have/violation FAIL (see above)")
        if not any(r["run_tags"]):
            problems.append("EMPTY tag output — the thinking-mode/"
                            "contamination signature")
        if args.descriptions and not any(r["descriptions"]):
            problems.append("EMPTY description output")
        b = base["entries"].get(CANARY) if base else None
        if b and b.get("seconds") and b.get("calls"):
            ratio = (r["seconds"] / r["calls"]) / (b["seconds"] / b["calls"])
            print(f"\ncanary timing: {r['seconds'] / r['calls']:.1f}s/call "
                  f"vs baseline {b['seconds'] / b['calls']:.1f}s/call "
                  f"({ratio:.1f}×)")
            if ratio > 2.5:
                problems.append(f"per-call time {ratio:.1f}× baseline — "
                                "environment changed (Ollama upgrade?)")
        elif base:
            print("\n(baseline has no per-entry timing — predates canary; "
                  "timing unchecked)")
        if problems:
            print("\nCANARY FAIL — do NOT trust measurements until resolved:")
            for p in problems:
                print(f"  ⚠ {p}")
            sys.exit(1)
        print("\nCANARY PASS — environment sane, safe to measure.")


if __name__ == "__main__":
    main()
