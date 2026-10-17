# Scene-level chronological summary: feasibility test and output example

**Date:** 2026-07-19
**Test video:** a broadcast TV travel segment shot on Tokyo streets (a
private golden fixture; it can't be redistributed), 340 s, **131 shots**:
deliberately the fastest-cut, hardest clip available, on the reasoning that
if scene grouping works here it works on almost anything.
**Models:** `openbmb/minicpm-v4.5` for captions and summaries,
`nomic-embed-text` (via Ollama) for segmentation. The study script is not
part of this repo.
**Quoted output:** verbatim, except the broadcaster's on-screen logo, which
is redacted as `[channel logo]`.

> **What this feature is:** replace the single flat paragraph
> `summarize_video` produces today with a short, ordered list of ONE
> sentence per **scene** (a run of consecutive cuts sharing the same
> place/people/action) — readable top-to-bottom to reconstruct the video
> chronologically without watching it. Acceptance bar: aggressive
> same-topic grouping; a few sentences on one topic is fine; **one
> sentence per cut is the thing to avoid**; over-merging is acceptable.
>
> **This is not built into the pipeline yet.** It's a proven-feasible
> backlog item; this file is the saved output example for future review.

---

## ✅ The winning approach: embedding-segmented scene summary

**Method:** caption every shot (Stage A, existing `_caption_frame`) →
embed each caption with `nomic-embed-text` → cut a new scene wherever
adjacent-caption cosine similarity dips below a threshold → fold 1-cut
blocks into their neighbor → one summarize call per resulting block. The
scene-boundary decision is deterministic (embeddings), NOT the VLM's job —
that's the whole trick (see "Why the first two approaches failed" below).

**Result — 130 captions → 9 scene lines, threshold T=0.70:**

| In–Out | Cuts | Scene |
|--------|-----:|-------|
| 00:00–00:17 | 9 | A man in a gray shirt interacts with various vending machines filled with drinks while accompanied by another person near outdoor street-side locations marked by signs including no-smoking notices and Japanese text throughout multiple angles of view. |
| 00:20–00:26 | 4 | People walk along a street lined with parked vehicles under clear daylight conditions near buildings surrounded by trees; some individuals move on sidewalks while others cross streets among pedestrians in various attire and carrying items close to their faces or hands as plants grow nearby amidst greenery further back toward structures marked [channel logo]. |
| 00:28–00:30 | 2 | A person with light-colored hair is seen outdoors near buildings and greenery; they drink water from an Aquafina bottle while standing next to large cylindrical objects wrapped in silver material before performing a push-up on their hands and knees amidst orange traffic cones nearby trees under the [channel logo]. |
| 00:32–01:31 | **30** | A man in a gray shirt interacts with various vending machines filled with drinks, snacks, model train cars, books, and other items while conversing with another person near an industrial setting marked by Japanese text on signs and walls; both individuals exchange objects throughout the scene as they examine different selections within multiple open-fronted appliances. |
| 01:33–01:36 | 3 | A man in a blue shirt is on public transportation with other passengers who are either sitting nearby using mobile devices or standing around him while advertisements featuring images of buildings and various stickers appear throughout the space; another person wearing a gray shirt stands near metal structures indoors as [channel logo] appears at the bottom left corner. |
| 01:40–01:47 | 5 | The scene captures a bustling urban environment with people moving through streets lined with shops displaying colorful merchandise under overcast skies near tall buildings; some individuals gather outdoors for photos in front of electronic displays and themed stores like Hello Kitty-themed ice cream parlors featuring various products advertised inside, while others navigate the area carrying bags or wearing casual attire. |
| 01:53–02:19 | 12 | A group of people in colorful costumes gather outdoors near a building entrance with graffiti-covered walls; some gesture while speaking to one another as bicycles are parked nearby on various sides and posters adorn the surrounding structures throughout multiple angles showing different individuals interacting or standing together amidst vibrant decorations, signs indicating opening hours (12:00 OPEN), oversized bear figures holding "6% DOKIDOKI" banners in Japanese script. |
| 02:20–02:25 | 5 | The scene features various colorful fashion accessories displayed in a store with [channel logo] branding visible throughout multiple angles of different products including shoes, earrings, clothing items like t-shirts and costumes featuring characters such as bananas or large-headed figures wearing aprons against vibrant backgrounds under bright lighting conditions where the [channel logo] consistently appears. |
| 02:26–05:29 | **60** | A group of people interact in a brightly colored store filled with various clothing items, toys, decorations, and accessories while examining garments on display shelves under multicolored walls made from interlocking blocks; some individuals wear vibrant costumes or rainbow-themed attire as they engage each other amidst the lively environment. |

**Why this passes the bar:** the two big set-pieces collapsed exactly as
wanted — the **30-cut vending-machine gag → one line**, the **60-cut /
~3-minute colorful-store scene → one line**. Zero one-per-cut anywhere.
Boundaries land on real topic changes.

**Known rough edges (tunable, not structural):** a few lines still run on
(00:20, 01:40, 01:53) — the same tendency the existing summary prompt
already fights; addressable via the summarize prompt + `num_predict`. The
in/out timecodes come from the first/last cut's timestamp in each block.

---

## Tuning dial

One threshold controls how aggressively cuts merge (this video):

| Threshold | Blocks (before 1-cut absorb) |
|-----------|------------------------------|
| 0.68 | 7 (most aggressive) |
| 0.70 | 12 → **9 after absorb** (shown above) |
| 0.72 | 18 |
| 0.76 | 32 |
| 0.80 | 51 (least aggressive) |

Lower = fewer, bigger blocks. **Calibration note:** an absolute-level /
running-centroid threshold does NOT work here — these captions are
globally high-similarity (adjacent cosine min 0.60, **median 0.83**, max
0.96, because they all share "person / channel logo / Japanese signage"
vocabulary), so everything merges into one block. Only the *relative dips*
in adjacent similarity discriminate. Open question before building it:
fixed threshold vs. normalize per-video off that video's own
adjacent-similarity distribution.

---

## Cost (on this 131-shot video)

| Stage | Time | Notes |
|-------|------|-------|
| Stage A — caption all 130 shots | **438.8s** | The real bottleneck. Unchanged by any Stage-B approach; same open question as the fixed-10-frame-cap backlog item. |
| Embedding (nomic-embed-text) | 2.9s | Essentially free. |
| Summarize 9 blocks | 20.3s | 9 text-only calls. |
| **Stage B total (embed + summarize)** | **~23s** | *Cheaper* than the two failed approaches (59s / 101s). |

---

## For comparison — today's shipped output (flat paragraph)

Same video, the current `summarize_video` (capped at 10 frames, one
paragraph). Useful but a wall of text with no chronology or navigability:

> The video features various scenes involving individuals interacting in
> different settings: an outdoor area with vending machines where a man
> hands something to a woman near industrial equipment; and indoor spaces
> including what appears to be storage areas marked by [channel logo]s, pipes,
> striped structures, open doorways. It also showcases colorful high-heeled
> shoes on display against abstract backgrounds featuring the same logo
> text. The video transitions into vibrant retail environments filled with
> clothing items like hats hanging above shelves set in multicolored block
> walls and displays of various merchandise such as masks or pillows; it
> includes scenes where people examine clothes, wear bright accessories
> including blue hair adorned with flowers. The settings are lively stores
> decorated to attract attention through colorful blocks arranged on
> ceilings. The video concludes by depicting a group scene within these
> vividly colored retail spaces featuring individuals in elaborate costumes
> and performance attire amidst toys decorations creating an energetic
> atmosphere; the final scenes show people interacting closely, some
> wearing distinctive yellow or multicolored outfits.

---

## Why the first two approaches failed (kept on record so they aren't re-tried)

Both let the VLM decide scene boundaries AND summarize in one step — the
boundary decision is the unreliable part.

1. **Single monolithic Stage-B call** over all 130 captions: under 3:1
   compression (46 lines), truncated before the video ended, narrated
   shot-to-shot continuity, emitted duplicate lines. Ollama's runtime
   context also defaults to 4096 tokens — 130 captions alone is 7777, a
   hard error until `num_ctx` is raised.
2. **Map-reduce batching** (9 batches of 15, group each, then merge):
   fixed the truncation but not the compression — per-batch results were
   wildly inconsistent (one batch 15→1, three others ~1:1), and the merge
   step spliced unrelated clauses into bloated run-ons. Slower too (101s).

The fix was to stop asking the VLM to find boundaries at all. The batched
test had already proven the VLM summarizes a *genuinely coherent* block
into one clean sentence perfectly — so hand it only coherent blocks, and
let embeddings draw the lines.
