"""config.video_id_from_path: the data/<video_id>/ working-dir name. unique=True
keeps same-named files from different folders (two cards' A001.mov) from
sharing one working dir and its cached cuts/frames."""
from config import video_id_from_path


def test_plain_filename_stem():
    assert video_id_from_path("/some/path/video.mp4") == "video"


def test_unique_same_stem_different_folders_never_collides():
    a = video_id_from_path("/card_a/A001.mov", unique=True)
    b = video_id_from_path("/card_b/A001.mov", unique=True)
    assert a != b
    assert a.startswith("A001-") and b.startswith("A001-")


def test_unique_is_stable_for_the_same_path():
    assert (video_id_from_path("/card_a/A001.mov", unique=True)
            == video_id_from_path("/card_a/A001.mov", unique=True))


def test_special_characters_sanitized():
    assert video_id_from_path("/path/My Video (Final)!.mp4") == "My_Video_Final_"


def test_extension_dropped():
    assert video_id_from_path("/path/clip.MOV") == "clip"


def test_multiple_dots_only_last_suffix_dropped():
    assert video_id_from_path("/path/clip.v2.final.mp4") == "clip.v2.final"


def test_run_of_invalid_characters_collapses_to_one_underscore():
    assert video_id_from_path("/path/日本語ファイル.mp4") == "_"


def test_dashes_underscores_and_dots_preserved():
    assert video_id_from_path("/path/my-video_final.v1.mp4") == "my-video_final.v1"
