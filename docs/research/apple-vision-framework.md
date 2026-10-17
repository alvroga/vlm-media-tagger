# Apple Intelligence and the Vision framework: replacement or supplement for the VLM?

**Date:** 2026-09-11
**Question:** some macOS apps index "red car" and do face recognition with Apple's on-device frameworks. Could
those replace MiniCPM-V for tagging, or run alongside it?
**Machine:** Mac Studio, M1 Ultra, 128 GB RAM, macOS 26.4, Xcode SDKs up to 26.5, no beta SDK. The Swift probe
scripts were one-offs and are not part of this repo.

## TL;DR

- **Foundation Models (the Apple Intelligence LLM) can't take images on this OS.** Image input
  (`Attachment(imageURL:)`) was announced for the next OS generation, not the shipping 26.4/26.5 SDK.
  Confirmed by inspecting the installed framework interface, not just by reading about it. Text-only
  Foundation Models had already been ruled out for translation (2026-07-19).
- **Vision's classifier can't recognise brands or logos at all:** 0/3 on three brand test images. It is a
  fixed taxonomy of a few thousand generic nouns and attributes, not an open-vocabulary namer. Its OCR is
  excellent when a brand is printed text and useless when it is only iconography.
- **On generic COCO objects, Vision's raw recall matches the VLM's, but only at ~31 unfiltered, hierarchical
  labels per image**, many of them useless as tags (`conveyance`, `portal`, `material`, `land`). At a
  similarly sized tag list its recall drops well below the VLM's. It is ~375× faster and runs on the Neural
  Engine, so it doesn't contend with the VLM for the GPU.
- **Face detection works; face *naming* is not a public API.** Bounding boxes are free; "who is this" means
  building an embedding, enrollment and matching layer. That also cuts directly against this project's rule
  of never identifying real people.

## 1. Foundation Models image input: not available here

Verified live rather than assumed:

```swift
import FoundationModels
let model = SystemLanguageModel.default
// => availability: available   (Apple Intelligence is on and downloaded)
```

A plain text prompt round-trips fine (~11 s including model load). But the installed
`FoundationModels.swiftmodule` interface (both the 26.4 command-line tools SDK and the 26.5 Xcode SDK) has only
a text case:

```
public enum Segment : Swift.Sendable, Swift.Identifiable, Swift.Equatable {
    case text(FoundationModels.Transcript.TextSegment)
```

No image or attachment case exists, and a compile using `Attachment` fails (`cannot find 'Attachment' in
scope`). The API described in September 2026 write-ups belongs to an OS release that isn't installed.

**Implication:** nothing to build yet. Re-run this check once that OS and its SDK ship.

## 2. Brand and logo recognition: a hard no

`VNClassifyImageRequest` + `VNRecognizeTextRequest` + `VNDetectFaceRectanglesRequest` on three brand images:

| Image | Vision top classify labels | OCR | MiniCPM-V 4.5 |
|---|---|---|---|
| Toyota logo timeline | `document` 0.06, `printed_page` 0.05 | `TOYOTA` @ 1.00 | "toyota" ✓ |
| Apple silhouette (no text) | `document` 0.83, `screenshot` 0.04 | (nothing) | "apple" ✓ |
| KFC delivery bikes on a street | `machine` 0.75, `vehicle` 0.75, `bicycle` 0.75, `people` 0.51 | garbled fragments | "kfc" ✓ |

The classifier has no concept of "toyota", "apple" or "kfc": it isn't prompt-driven and the taxonomy is
fixed. OCR reads brand text when it is literally rendered, and is blank or garbled when the brand is
iconography or partly obscured. OCR never invents text, but it misses obscured, curved and angled labels; the
VLM reads those, and sometimes misreads them.

## 3. Generic object detection: a fair benchmark

`tests/golden/coco_tier.py` is the right tool: a frozen 200-image COCO val2017 sample with pixel-area-weighted
ground truth. The same 200 images went through `VNClassifyImageRequest`, scored with the tier's own
`COCO_SYNONYMS` map and `harness._matches` matcher (the identical scoring code). The whole run took **~4 s**,
≈375× faster than the VLM (~25 min).

| Confidence floor | Recall | Prominent recall | Avg labels/image |
|---|---|---|---|
| ≥0.00 (accept everything) | 80.2% | 83.8% | 31.3 |
| ≥0.05 | 74.9% | 79.1% | 21.7 |
| ≥0.10 | 65.2% | 70.0% | 15.9 |
| ≥0.15 | 57.7% | 64.3% | — |
| ≥0.20 | 53.8% | 61.7% | 11.8 |
| **MiniCPM-V 4.5 (re-run with rebalanced sampling)** | **78.0%** | **82.0%** | ~10–25 clean tags |

The committed 2026-07-11 result (`tests/golden/results/coco-2026-07-11-openbmb_minicpm-v4.5.json`) reads
76.9% / 80.5%. 4.7 later measured 81.9% / 84.8% ([evaluation](minicpm-v4.7-evaluation.md)).

**The headline number misleads on its own.** Vision only edges out the VLM by emitting ~31 labels per image,
including hierarchical synonym spam (one bus fires `machine` + `vehicle` + `automobile` + `bus`, correlated,
not independent detections) and abstract scene words nobody wants as a tag. Making that a usable tag list
would need a curation layer at least as big as `tag_filter.py`. At a confidence floor that yields a
similarly sized list (~12–16 labels, ≈0.10–0.15), **recall drops to 58–65%, well below the VLM's 78%.**

Per category (threshold 0.05, categories with ≥3 ground-truth instances):

- **Strong (100%)** on rigid, distinctive objects: car, truck, motorcycle, laptop, zebra, wine glass, tennis
  racket, fork, sheep, teddy bear.
- **Weak** where context or fine distinctions matter: remote (0%), cell phone (11%), bench (25%), kite (33%),
  umbrella (43%), sandwich (44%), cup (50%), chair (50%).

**The real advantage is cost.** Vision runs on the Neural Engine with near-zero RAM and GPU impact, doesn't
queue behind the VLM, and could run alongside the VLM call rather than replace it.

## 4. Faces: naming is a build project, not an API call

`VNDetectFaceRectanglesRequest` returns bounding boxes for free (and correctly reported 0 faces on the
non-people images). No public Apple API returns a *name* for a face outside the Photos app's own People
album. An app would have to detect a face, compute an embedding and match it against enrollment photos the
user labels.

It would also be a new axis for this project. The pipeline deliberately never identifies real people: person
nouns are dropped from tags (`tag_filter.py`), and descriptions never name someone from their appearance
(see the 4.7 evaluation). Named faces would need their own explicit opt-in and consent design, and their own
output field, not a reuse of the tag pipeline.

## Recommendation

- **Don't chase Foundation Models image input** until it ships on a real OS.
- **Vision object classification is not a VLM replacement.** Comparable raw recall needs noisy output that
  costs as much curation as it saves; at a usable confidence level it underperforms. Its niche (perfect hits
  on common rigid objects, near-zero cost, runs in parallel) makes it a plausible *supplementary* signal,
  like OCR.
- **Named-face tagging** is an interesting candidate, but only after an explicit privacy decision on how it
  coexists with the never-identify-people rule.
