"""tagger.main with --sidecar / --skip-done, the VLM mocked out: a second run
over the same folder processes nothing, a changed file is redone."""
import json
import sys

import pytest

import config
import tagger


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def fake_process_file(path, **kw):
        calls.append(path.name)
        return {"path": str(path), "type": "video", "status": "ok", "tags": ["car"],
                "description": "A car." if kw["describe"] else "", "elapsed_sec": 0.1}
    monkeypatch.setattr(tagger, "process_file", fake_process_file)
    monkeypatch.setattr(config, "check_vlm", lambda model: None)
    monkeypatch.setattr(tagger, "shutdown_resolve", lambda: None)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["tagger.py", *map(str, args), "-o", "/dev/null"])
        return tagger.main()
    return run, calls


def test_skip_done_resumes_and_redoes_changed(tmp_path, fake_run):
    run, calls = fake_run
    (tmp_path / "a.mp4").write_bytes(b"a")
    (tmp_path / "b.mov").write_bytes(b"b")

    assert run(tmp_path, "--sidecar", "--skip-done") == 0
    assert sorted(calls) == ["a.mp4", "b.mov"]
    side = json.loads((tmp_path / "a.mp4.json").read_text())
    assert side["tasks"].keys() == {"tags", "description"}
    assert side["description"] == "A car."

    calls.clear()
    assert run(tmp_path, "--sidecar", "--skip-done") == 0
    assert calls == []

    (tmp_path / "b.mov").write_bytes(b"changed")
    assert run(tmp_path, "--sidecar", "--skip-done") == 0
    assert calls == ["b.mov"]


def test_skip_done_needs_a_state_sink(tmp_path, fake_run):
    run, _ = fake_run
    (tmp_path / "a.mp4").write_bytes(b"a")
    with pytest.raises(SystemExit):
        run(tmp_path, "--skip-done")
