"""tagger.py — the end-to-end CLI around the pipeline. The VLM, Resolve and
the pipeline are all monkeypatched: these tests cover routing (RAW fallback,
R3D chunks, stills), ordering (clean → profanity → translate) and the JSON
record shape, never inference."""
import json

import pytest
from PIL import Image

import config
import resolve_proxy
import tagger
from media_probe import UndecodableMediaError


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "FILTER_PROFANITY", False)
    tagger._resolve_state = None
    yield
    tagger._resolve_state = None


def _fake_resolve_obj():
    class FakeResolve:
        quit_called = False

        def Quit(self):
            FakeResolve.quit_called = True

    return FakeResolve()


# -- Resolve RAW fallback -----------------------------------------------------

def test_reuses_existing_proxy_without_touching_resolve(tmp_path, monkeypatch):
    out_dir = config.DATA_DIR / "clip"
    out_dir.mkdir(parents=True)
    proxy = out_dir / tagger.RESOLVE_PROXY_FILENAME
    proxy.write_bytes(b"fake proxy")
    calls = []
    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running",
                        lambda: calls.append("ensure") or (None, False))
    monkeypatch.setattr(resolve_proxy, "_render_clip", lambda *a, **k: calls.append("render"))
    monkeypatch.setattr(tagger.pipeline, "run",
                        lambda video, model=None, video_id=None: [{"class": "car"}]
                        if video == str(proxy) else pytest.fail("wrong video path"))
    assert tagger._process_raw_via_resolve(tmp_path / "s.r3d", "clip", RuntimeError()) == [{"class": "car"}]
    assert calls == []


def test_transcodes_a_720p_proxy_when_none_cached(tmp_path, monkeypatch):
    proxy = config.DATA_DIR / "clip" / tagger.RESOLVE_PROXY_FILENAME
    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running", lambda: (_fake_resolve_obj(), True))
    renders = []

    def fake_render(resolve, input_path, output_path, width=None, height=None):
        renders.append((input_path, output_path, width, height))
        return output_path

    monkeypatch.setattr(resolve_proxy, "_render_clip", fake_render)
    monkeypatch.setattr(tagger.pipeline, "run", lambda video, model=None, video_id=None: [{"class": "tree"}])
    assert tagger._process_raw_via_resolve(tmp_path / "s.braw", "clip", RuntimeError()) == [{"class": "tree"}]
    assert renders == [(str(tmp_path / "s.braw"), str(proxy), 1280, 720)]


def test_resolve_unavailable_is_its_own_error(tmp_path, monkeypatch):
    def unavailable():
        raise RuntimeError("Resolve not installed")

    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running", unavailable)
    with pytest.raises(tagger.ResolveUnavailableError):
        tagger._process_raw_via_resolve(tmp_path / "s.r3d", "clip", RuntimeError())


def test_no_resolve_flag_refuses_without_launching(tmp_path, monkeypatch):
    monkeypatch.setattr(tagger, "RESOLVE_ALLOWED", False)
    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running", lambda: pytest.fail("launched"))
    with pytest.raises(tagger.ResolveUnavailableError):
        tagger._process_raw_via_resolve(tmp_path / "s.r3d", "clip", RuntimeError())


def test_failed_transcode_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running", lambda: (_fake_resolve_obj(), True))
    monkeypatch.setattr(resolve_proxy, "_render_clip", lambda *a, **k: None)
    monkeypatch.setattr(tagger.pipeline, "run", lambda *a, **k: pytest.fail("ran pipeline"))
    assert tagger._process_raw_via_resolve(tmp_path / "s.r3d", "clip", RuntimeError()) is None


def test_get_resolve_connects_once(monkeypatch):
    calls = []
    obj = _fake_resolve_obj()
    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running", lambda: calls.append(1) or (obj, True))
    assert tagger._get_resolve() is obj and tagger._get_resolve() is obj
    assert len(calls) == 1


@pytest.mark.parametrize("ours", [True, False])
def test_shutdown_quits_only_an_instance_we_own(monkeypatch, ours):
    obj = _fake_resolve_obj()
    monkeypatch.setattr(resolve_proxy, "ensure_resolve_running", lambda: (obj, ours))
    tagger._get_resolve()
    tagger.shutdown_resolve()
    assert obj.quit_called is ours
    assert tagger._resolve_state is None


def test_shutdown_is_a_noop_when_never_connected():
    tagger.shutdown_resolve()


# -- process_file routing -----------------------------------------------------

def _run(path, **kw):
    return tagger.process_file(path, model="m", lang=kw.get("lang", "en"),
                               describe=kw.get("describe", False), force=False,
                               keep_workdir=kw.get("keep_workdir", False))


def test_non_first_r3d_chunk_skips_before_decoding(tmp_path, monkeypatch):
    f = tmp_path / "clip_002.R3D"
    f.write_bytes(b"x")
    monkeypatch.setattr(tagger.pipeline, "run", lambda *a, **k: pytest.fail("decoded"))
    assert _run(f)["status"] == "redundant-chunk"


def test_first_r3d_chunk_reaches_resolve_and_reports_unavailable(tmp_path, monkeypatch):
    f = tmp_path / "clip_001.R3D"
    f.write_bytes(b"x")

    def undecodable(*a, **k):
        raise UndecodableMediaError("no video stream")

    def unavailable(*a, **k):
        raise tagger.ResolveUnavailableError("not installed")

    monkeypatch.setattr(tagger.pipeline, "run", undecodable)
    monkeypatch.setattr(tagger, "_process_raw_via_resolve", unavailable)
    rec = _run(f)
    assert rec["status"] == "resolve-unavailable"
    assert "not installed" in rec["error"]


def test_video_record_shape_and_tag_cleanup(tmp_path, monkeypatch):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    segs = [{"class": "red car", "start_time": 4.0, "end_time": 6.0, "frame_count": 2},
            {"class": "a man", "start_time": 0.0, "end_time": 1.0, "frame_count": 1},
            {"class": "tree", "start_time": 0.0, "end_time": 9.5, "frame_count": 3}]

    def fake_run(video, model=None, force=False, video_id=None):
        work = config.DATA_DIR / video_id
        (work / "frames").mkdir(parents=True)
        (work / "frames" / "index.json").write_text(json.dumps([{"timestamp_sec": 0.0}, {"timestamp_sec": 9.5}]))
        (work / "cuts.json").write_text(json.dumps({"scene_count": 3}))
        return segs

    monkeypatch.setattr(tagger.pipeline, "run", fake_run)
    monkeypatch.setattr(tagger, "_duration_sec", lambda p: 9.5)
    rec = _run(f)
    assert rec["status"] == "ok" and rec["source"] == "native"
    assert rec["tags"] == ["car", "tree"]          # person tag dropped, color stripped
    assert rec["duration_sec"] == 9.5 and rec["shot_count"] == 3
    assert [s["tag"] for s in rec["segments"]] == ["tree", "car"]
    assert not (config.DATA_DIR / config.video_id_from_path(str(f), unique=True)).exists()


def test_profanity_filter_runs_before_translation(monkeypatch):
    monkeypatch.setattr(config, "FILTER_PROFANITY", True)
    seen = []
    monkeypatch.setattr(tagger.profanity, "filter_tags", lambda tags, lang: [t for t in tags if t != "bad"])
    monkeypatch.setattr(tagger.opus_translate, "translate_tags", lambda tags, lang: seen.append(tags) or tags)
    tags, tags_en = tagger._finish_tags([{"class": "bad"}, {"class": "dog"}], "es")
    assert seen == [["dog"]] and tags_en == ["dog"]


def test_still_image_goes_through_the_still_prompt(tmp_path, monkeypatch):
    f = tmp_path / "photo.png"
    Image.new("RGB", (2000, 1000), "white").save(f)
    import detect_objects_vlm
    calls = []

    def fake_detect(video_id, model=None, object_prompt=None):
        idx = json.loads((config.DATA_DIR / video_id / "frames" / "index.json").read_text())
        calls.append((object_prompt, Image.open(idx[0]["path"]).size))
        return [{"class": "logo", "start_time": 0.0, "end_time": 0.0, "frame_count": 1}]

    monkeypatch.setattr(detect_objects_vlm, "detect_objects_vlm", fake_detect)
    rec = _run(f)
    assert rec["type"] == "image" and rec["tags"] == ["logo"]
    assert calls == [(detect_objects_vlm.STILL_OBJECT_PROMPT, (1280, 640))]


def test_undecodable_image_is_reported(tmp_path, monkeypatch):
    f = tmp_path / "broken.jpg"
    f.write_bytes(b"not an image")
    monkeypatch.setattr(tagger.shutil, "which", lambda name: None)
    assert _run(f)["status"] == "image-undecodable"


def test_collect_inputs_filters_by_extension(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    (tmp_path / "._a.mp4").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.jpg").write_bytes(b"")
    assert [p.name for p in tagger.collect_inputs([str(tmp_path)], False)] == ["a.mp4"]
    assert sorted(p.name for p in tagger.collect_inputs([str(tmp_path)], True)) == ["a.mp4", "c.jpg"]


def test_video_id_is_unique_per_folder(tmp_path):
    a = config.video_id_from_path(str(tmp_path / "x" / "A001.mov"), unique=True)
    b = config.video_id_from_path(str(tmp_path / "y" / "A001.mov"), unique=True)
    assert a != b and a.startswith("A001-")


def test_segments_follow_the_plural_merge():
    segs = [{"class": "decorative plate", "start_time": 0.0, "end_time": 5.0, "frame_count": 1},
            {"class": "decorative plates", "start_time": 6.9, "end_time": 7.0, "frame_count": 1},
            {"class": "teapot", "start_time": 0.0, "end_time": 7.0, "frame_count": 2}]
    tags = [s["tag"] for s in tagger._clean_segments(segs)]
    assert tags == ["decorative plates", "teapot", "decorative plates"]
