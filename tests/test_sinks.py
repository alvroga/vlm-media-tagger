"""sinks: the JSON sidecar (write, task merge, staleness, read-only mirror),
the jsonl sink, plugin loading, and the library scanner built on is_done."""
import json
import os
import stat
import sys
import types

import pytest

import library
import sinks

FULL = {"tags", "description"}


def _record(**over):
    rec = {"path": "ignored", "type": "video", "status": "ok", "source": "native",
           "tags": ["car", "street"], "description": "A red car on a street.",
           "segments": [{"class": "car", "start": 0.0, "end": 1.0}],
           "model": "minicpm-v4.7", "prompt_set": "minicpm-v4.7", "language": "en",
           "tool_version": "0.1.0", "tasks": ["description", "tags"], "elapsed_sec": 3.2}
    rec.update(over)
    return rec


@pytest.fixture
def media(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"video bytes")
    return f


@pytest.fixture
def sink(tmp_path):
    return sinks.SidecarSink(mirror_root=tmp_path / "mirror")


def test_sidecar_beside_file_keeps_full_name(sink, media):
    sink.write(media, _record())
    data = json.loads((media.parent / "clip.mp4.json").read_text())
    assert data["file"] == "clip.mp4"
    assert data["source_size"] == media.stat().st_size
    assert data["source_mtime_ns"] == media.stat().st_mtime_ns
    assert data["tool_version"] == "0.1.0"
    assert data["tasks"]["tags"]["model"] == "minicpm-v4.7"
    assert data["tasks"]["description"]["prompt_set"] == "minicpm-v4.7"
    assert data["tags"] == ["car", "street"]
    assert "path" not in data and "status" not in data  # run-only fields
    assert [p.name for p in media.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_mp4_and_mov_never_collide(sink, tmp_path):
    a, b = tmp_path / "clip.mp4", tmp_path / "clip.mov"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    sink.write(a, _record(tags=["a"]))
    sink.write(b, _record(tags=["b"]))
    assert json.loads((tmp_path / "clip.mp4.json").read_text())["tags"] == ["a"]
    assert json.loads((tmp_path / "clip.mov.json").read_text())["tags"] == ["b"]


def test_failed_record_writes_nothing(sink, media):
    sink.write(media, _record(status="undecodable"))
    assert not (media.parent / "clip.mp4.json").exists()
    assert not sink.is_done(media, {"tags"})


def test_tags_only_run_keeps_earlier_description(sink, media):
    sink.write(media, _record())
    sink.write(media, _record(tags=["truck"], description="", tasks=["tags"], model="newer"))
    data = json.loads((media.parent / "clip.mp4.json").read_text())
    assert data["tags"] == ["truck"]
    assert data["description"] == "A red car on a street."
    assert data["tasks"]["tags"]["model"] == "newer"
    assert data["tasks"]["description"]["model"] == "minicpm-v4.7"


def test_is_done_needs_every_requested_task(sink, media):
    sink.write(media, _record(tasks=["tags"], description=""))
    assert sink.is_done(media, {"tags"})
    assert not sink.is_done(media, FULL)


def test_changed_file_is_stale_and_drops_old_fields(sink, media):
    sink.write(media, _record())
    assert sink.is_done(media, FULL)
    media.write_bytes(b"re-exported, different size")
    assert not sink.is_done(media, {"tags"})
    # A tags-only run on the changed file must not carry the old description.
    sink.write(media, _record(tags=["new"], description="", tasks=["tags"]))
    data = json.loads((media.parent / "clip.mp4.json").read_text())
    assert "description" not in data
    assert set(data["tasks"]) == {"tags"}


def test_touched_file_with_same_size_is_stale(sink, media):
    sink.write(media, _record())
    st = media.stat()
    os.utime(media, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert not sink.is_done(media, FULL)


def test_other_language_is_not_done(sink, media):
    sink.write(media, _record())
    assert not sink.is_done(media, FULL, language="es")


def test_model_filter_is_opt_in(sink, media):
    sink.write(media, _record())
    assert sink.is_done(media, FULL)                          # any model counts
    assert sink.is_done(media, FULL, model="minicpm-v4.7")
    assert not sink.is_done(media, FULL, model="some-newer-model")


def test_corrupt_sidecar_reads_as_not_done(sink, media):
    (media.parent / "clip.mp4.json").write_text("{ not json")
    assert not sink.is_done(media, {"tags"})
    sink.write(media, _record())  # and is simply overwritten
    assert sink.is_done(media, FULL)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_folder_mirrors_and_logs(sink, tmp_path, capsys):
    folder = tmp_path / "locked"
    folder.mkdir()
    media = folder / "a.mov"
    media.write_bytes(b"x")
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        sink.write(media, _record())
        assert not (folder / "a.mov.json").exists()
        mirror = sink.mirror_path(media)
        assert mirror.is_file() and str(mirror).startswith(str(tmp_path / "mirror"))
        assert "read-only" in capsys.readouterr().err
        assert sink.is_done(media, FULL)  # the mirror counts as state
    finally:
        folder.chmod(stat.S_IRWXU)


def test_jsonl_sink_appends(tmp_path, media):
    out = tmp_path / "out" / "records.jsonl"
    s = sinks.JsonlSink(out)
    s.write(media, _record())
    s.write(media, _record(tags=["second"]))
    s.close()
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["tags"] for r in lines] == [["car", "street"], ["second"]]
    assert lines[0]["path"] == str(media)
    assert not s.tracks_state


def test_load_builtin_and_plugin_sinks(monkeypatch):
    mod = types.ModuleType("fake_plugin")

    class Hook(sinks.OutputSink):
        name = "hook"

        def write(self, media, record):
            pass
    mod.Hook = Hook
    mod.NotASink = dict
    monkeypatch.setitem(sys.modules, "fake_plugin", mod)

    loaded = sinks.load_sinks("sidecar, fake_plugin:Hook")
    assert [s.name for s in loaded] == ["sidecar", "hook"]
    assert sinks.load_sinks("") == []
    for bad in ("nope", "fake_plugin:Missing", "no_such_module_xyz:Cls", "fake_plugin:NotASink"):
        with pytest.raises(ValueError):
            sinks.load_sinks(bad)


def test_is_done_without_state_sinks_is_never_done(media):
    assert not sinks.is_done([sinks.JsonlSink(media.parent / "x.jsonl")], media, {"tags"})


def test_scan_counts_and_skips(tmp_path, sink):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").mkdir()
    for name in ("a.mp4", "b.JPG", "sub/c.mov", "notes.txt", "._a.mp4",
                 ".hidden/d.mp4", "A001_C001_001.R3D", "A001_C001_002.R3D"):
        (tmp_path / name).write_bytes(b"x")
    sink.write(tmp_path / "a.mp4", _record())

    res = library.scan(tmp_path, FULL, [sink])
    assert res.total == 4  # a.mp4, b.JPG, sub/c.mov, the _001 chunk
    assert res.done == 1
    assert sorted(p.name for p in res.candidates) == ["A001_C001_001.R3D", "b.JPG", "c.mov"]

    # The sidecars themselves are never media.
    assert all(not p.name.endswith(".json") for p in res.candidates)


def test_sidecar_is_readable_by_others_and_keeps_its_mode(sink, media):
    sink.write(media, _record())
    side = sink.sidecar_path(media)
    assert stat.S_IMODE(side.stat().st_mode) == 0o666 & ~sinks._UMASK   # not mkstemp's 0600
    os.chmod(side, 0o640)
    sink.write(media, _record(tasks=["tags"]))
    assert stat.S_IMODE(side.stat().st_mode) == 0o640                  # a user's choice survives


def test_stale_sidecar_beside_file_does_not_hide_current_mirror(sink, media):
    sink.write(media, _record())
    side = sink.sidecar_path(media)
    stale = json.loads(side.read_text())
    stale["source_size"] = 1                     # describes an older export
    side.write_text(json.dumps(stale))
    mirror = sink.mirror_path(media)
    mirror.parent.mkdir(parents=True)
    current = json.loads(json.dumps(stale))
    current["source_size"] = media.stat().st_size
    mirror.write_text(json.dumps(current))
    assert sink.is_done(media, FULL)


@pytest.mark.parametrize("tasks", [["tags", "description"], "tags,description", {"tags": "x"}])
def test_hand_mangled_tasks_read_as_not_done_and_never_crash(sink, media, tasks):
    sink.write(media, _record())
    side = sink.sidecar_path(media)
    data = json.loads(side.read_text())
    data["tasks"] = tasks
    side.write_text(json.dumps(data))
    assert sink.is_done(media, FULL, model="minicpm-v4.7") is False
    sink.write(media, _record())                 # and the redo can write over it
    assert sink.is_done(media, FULL)
