"""vlm.py request shapes, the truncated-JSON salvage in _parse_vlm_objects,
and prompt-set selection — no server needed (urlopen is faked)."""
import io
import json

import config
import prompts
import vlm
from detect_objects_vlm import _parse_vlm_objects


def test_llama_request_shape(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    monkeypatch.setattr(config, "VLM_BACKEND", "llama")
    monkeypatch.setattr(vlm.urllib.request, "urlopen", fake_urlopen)
    out = vlm.chat([{"role": "system", "content": "s"},
                    {"role": "user", "content": "p", "images": ["QUJD"]}], num_predict=77)
    assert out == "hi"
    assert sent["url"].endswith("/v1/chat/completions")
    body = sent["body"]
    assert body["max_tokens"] == 77 and body["temperature"] == 0.2
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0] == {"role": "system", "content": "s"}
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "p"}
    assert user[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_llama_transport_error_is_vlmerror(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(config, "VLM_BACKEND", "llama")
    monkeypatch.setattr(vlm.urllib.request, "urlopen", boom)
    try:
        vlm.chat([{"role": "user", "content": "p"}])
    except vlm.VLMError as e:
        assert "refused" in str(e)
    else:
        raise AssertionError("expected VLMError")


def test_truncated_tag_list_is_salvaged():
    raw = '{"objects": ["car", "road sign", "a very long headline that never closes'
    assert _parse_vlm_objects(raw) == ["car", "road sign"]


def test_unsalvageable_text_still_returns_empty():
    assert _parse_vlm_objects('{"objects": ["unterminated') == []
    assert _parse_vlm_objects("no json here") == []


def test_prompt_set_follows_config(monkeypatch):
    monkeypatch.setattr(config, "PROMPT_SET", "minicpm-v4.5")
    p45 = prompts.get("IMAGE_CAPTION_PROMPT")
    monkeypatch.setattr(config, "PROMPT_SET", "minicpm-v4.7")
    p47 = prompts.get("IMAGE_CAPTION_PROMPT")
    assert "2-4 short sentences" in p45 and "4-8 sentences" in p47
    assert "never by name unless their name is printed" in p47
    with prompts.override(IMAGE_CAPTION_PROMPT="x"):
        assert prompts.get("IMAGE_CAPTION_PROMPT") == "x"
    assert prompts.get("IMAGE_CAPTION_PROMPT") == p47


def test_47_set_keeps_the_placeholder_contract():
    # detect_objects_vlm's leak filter matches "example item" literally
    for name in ("OBJECT_PROMPT", "STILL_OBJECT_PROMPT"):
        assert "example item" in prompts.PROMPT_SETS["minicpm-v4.7"][name]
