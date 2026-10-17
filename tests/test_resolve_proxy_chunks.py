"""resolve_proxy.filter_r3d_chunks — R3D ships as numbered chunks per clip
(..._001.R3D, ..._002.R3D, ...); only the first chunk's clip metadata
already references and plays through the rest, so only it should ever
reach Resolve (rule since 2026-07-13)."""
from resolve_proxy import filter_r3d_chunks


def test_keeps_only_first_chunk_per_clip():
    paths = [
        "/media/CLIP_001.R3D",
        "/media/CLIP_002.R3D",
        "/media/CLIP_003.R3D",
        "/media/CLIP_004.R3D",
    ]
    assert filter_r3d_chunks(paths) == ["/media/CLIP_001.R3D"]


def test_multiple_clips_each_keep_their_own_first_chunk():
    paths = [
        "/media/A_001.R3D",
        "/media/A_002.R3D",
        "/media/B_001.R3D",
        "/media/B_002.R3D",
        "/media/B_003.R3D",
    ]
    assert filter_r3d_chunks(paths) == ["/media/A_001.R3D", "/media/B_001.R3D"]


def test_case_insensitive_extension_and_order_preserved():
    paths = ["/media/clip_002.r3d", "/media/clip_001.r3d"]
    assert filter_r3d_chunks(paths) == ["/media/clip_001.r3d"]


def test_non_r3d_paths_pass_through_unchanged():
    paths = ["/media/clip.braw", "/media/clip.mxf", "/media/still.dng"]
    assert filter_r3d_chunks(paths) == paths


def test_mixed_r3d_and_other_formats():
    paths = [
        "/media/scene_001.R3D",
        "/media/scene_002.R3D",
        "/media/other.braw",
    ]
    assert filter_r3d_chunks(paths) == ["/media/scene_001.R3D", "/media/other.braw"]


def test_r3d_without_chunk_suffix_passes_through():
    # doesn't match the _NNN convention at all — never seen in practice,
    # but must not be silently dropped by a filter that only ever removes
    # confirmed-redundant later chunks
    paths = ["/media/single_take.R3D"]
    assert filter_r3d_chunks(paths) == paths
