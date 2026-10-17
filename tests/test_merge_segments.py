"""_merge_segments: per-frame noun detections -> merged segments.

The core behavior under test (pipeline improvement, 2026-07-10):
two detections of the same noun only merge into one segment if they fall
within the SAME shot — this is what stops a 30-second ad with a car in 9
different shots from collapsing into "one car segment" (car present the
whole time). Falls back to a time-gap heuristic when no shot boundaries
are available (interval-sampled videos with no detected cuts).
"""
from detect_objects_vlm import _merge_segments


def _frame(ts, nouns):
    return {"timestamp_sec": ts, "nouns": nouns}


def test_same_shot_hits_merge_into_one_segment():
    frames = [_frame(1.0, ["car"]), _frame(3.0, ["car"])]
    boundaries = [0.0, 10.0]
    segments = _merge_segments(frames, boundaries=boundaries)
    assert len(segments) == 1
    assert segments[0]["class"] == "car"
    assert segments[0]["frame_count"] == 2
    assert segments[0]["start_time"] == 1.0
    assert segments[0]["end_time"] == 3.0


def test_cross_shot_hits_stay_separate_segments():
    # the exact regression this feature exists for: same noun, different
    # shots, must NOT collapse into one giant segment
    frames = [_frame(1.0, ["car"]), _frame(15.0, ["car"]), _frame(25.0, ["car"])]
    boundaries = [0.0, 10.0, 20.0, 30.0]
    segments = _merge_segments(frames, boundaries=boundaries)
    assert len(segments) == 3
    assert all(s["class"] == "car" for s in segments)
    assert all(s["frame_count"] == 1 for s in segments)


def test_last_shot_boundary_is_inclusive_on_both_ends():
    # boundaries=[0,10,20,30]: the final shot is [20,30] (closed), not
    # [20,30) like the earlier ones — a hit exactly AT the video's end
    # timestamp must still merge with the rest of that last shot
    frames = [_frame(25.0, ["car"]), _frame(30.0, ["car"])]
    boundaries = [0.0, 10.0, 20.0, 30.0]
    segments = _merge_segments(frames, boundaries=boundaries)
    assert len(segments) == 1
    assert segments[0]["frame_count"] == 2


def test_timestamp_outside_all_boundaries_never_merges():
    frames = [_frame(5.0, ["car"]), _frame(50.0, ["car"])]
    boundaries = [0.0, 10.0]
    segments = _merge_segments(frames, boundaries=boundaries)
    assert len(segments) == 2


def test_no_boundaries_falls_back_to_gap_sec():
    frames = [_frame(1.0, ["car"]), _frame(5.0, ["car"]), _frame(20.0, ["car"])]
    segments = _merge_segments(frames, gap_sec=10.0, boundaries=None)
    # 1.0 -> 5.0 within 10s gap: merges. 5.0 -> 20.0 is a 15s gap: splits.
    assert len(segments) == 2
    assert segments[0]["frame_count"] == 2
    assert segments[1]["frame_count"] == 1


def test_gap_sec_boundary_is_inclusive():
    frames = [_frame(0.0, ["car"]), _frame(10.0, ["car"])]
    segments = _merge_segments(frames, gap_sec=10.0, boundaries=None)
    assert len(segments) == 1
    assert segments[0]["frame_count"] == 2


def test_distinct_nouns_produce_independent_segments():
    frames = [_frame(1.0, ["car", "tree"]), _frame(2.0, ["car"])]
    boundaries = [0.0, 10.0]
    segments = _merge_segments(frames, boundaries=boundaries)
    classes = {s["class"]: s["frame_count"] for s in segments}
    assert classes == {"car": 2, "tree": 1}


def test_detector_name_uses_short_model_name():
    frames = [_frame(1.0, ["car"])]
    segments = _merge_segments(frames, boundaries=[0.0, 10.0],
                               model_name="openbmb/minicpm-v4.5:latest")
    assert segments[0]["detector"] == "openbmb/minicpm-v4.5:car"


def test_segments_sorted_by_frame_count_desc_then_class_asc():
    frames = [
        _frame(1.0, ["zebra"]),
        _frame(1.0, ["apple"]), _frame(2.0, ["apple"]), _frame(3.0, ["apple"]),
        _frame(1.0, ["mango"]), _frame(2.0, ["mango"]),
    ]
    segments = _merge_segments(frames, boundaries=[0.0, 10.0])
    ordered = [(s["class"], s["frame_count"]) for s in segments]
    assert ordered == [("apple", 3), ("mango", 2), ("zebra", 1)]


def test_empty_frame_results_returns_empty_list():
    assert _merge_segments([], boundaries=[0.0, 10.0]) == []


def test_frame_with_no_nouns_contributes_nothing():
    frames = [_frame(1.0, []), _frame(2.0, ["car"])]
    segments = _merge_segments(frames, boundaries=[0.0, 10.0])
    assert len(segments) == 1
    assert segments[0]["class"] == "car"
