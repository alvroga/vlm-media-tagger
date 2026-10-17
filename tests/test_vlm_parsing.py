"""_parse_vlm_objects: raw VLM text -> cleaned noun list. Covers the real
dirty-output shapes seen in practice (single quotes, {"name": ...}
dicts, <think> blocks) plus the defensive guards found by the golden-set
harness (person-noun filtering, placeholder-leak filtering, malformed
JSON shapes). No network calls, no Ollama."""
from detect_objects_vlm import _parse_vlm_objects


def test_clean_json():
    assert _parse_vlm_objects('{"objects": ["car", "tree", "fence"]}') == \
        ["car", "tree", "fence"]


def test_single_quoted_dict_falls_back_to_ast():
    assert _parse_vlm_objects("{'objects': ['car', 'bicycle']}") == \
        ["car", "bicycle"]


def test_think_block_is_stripped_before_parsing():
    text = '<think>let me look at this image carefully</think>{"objects": ["car"]}'
    assert _parse_vlm_objects(text) == ["car"]


def test_think_block_with_surrounding_prose():
    text = 'Sure!<think>reasoning here</think>\n{"objects": ["dog", "leash"]}\nDone.'
    assert _parse_vlm_objects(text) == ["dog", "leash"]


def test_name_wrapped_dict_entries():
    text = '{"objects": [{"name": "hat"}, {"name": "bicycle"}]}'
    assert _parse_vlm_objects(text) == ["hat", "bicycle"]


def test_object_and_label_dict_keys_also_supported():
    text = '{"objects": [{"object": "car"}, {"label": "tree"}]}'
    assert _parse_vlm_objects(text) == ["car", "tree"]


def test_dict_entry_with_no_usable_key_becomes_empty_string_and_is_dropped():
    text = '{"objects": [{"color": "red"}, "car"]}'
    assert _parse_vlm_objects(text) == ["car"]


def test_hedging_parenthetical_is_stripped():
    text = '{"objects": ["liquid (possibly coffee)", "cup"]}'
    assert _parse_vlm_objects(text) == ["liquid", "cup"]


def test_underscore_normalized_to_space():
    text = '{"objects": ["stop_sign"]}'
    assert _parse_vlm_objects(text) == ["stop sign"]


def test_person_nouns_dropped_exact_match():
    text = '{"objects": ["car", "man", "person", "hand", "dog"]}'
    assert _parse_vlm_objects(text) == ["car", "dog"]


def test_placeholder_leak_filtered():
    text = '{"objects": ["example item one", "example item two", "real car"]}'
    assert _parse_vlm_objects(text) == ["real car"]


def test_no_json_object_present_returns_empty():
    assert _parse_vlm_objects("I don't see anything specific here.") == []


def test_empty_string_returns_empty():
    assert _parse_vlm_objects("") == []


def test_malformed_json_unrecoverable_even_by_ast_returns_empty():
    assert _parse_vlm_objects('{"objects": [car, tree unquoted]}') == []


def test_top_level_not_a_dict_returns_empty():
    # ast.literal_eval would happily parse a bare list — must not .get() on it
    assert _parse_vlm_objects('["car", "tree"]') == []


def test_objects_key_not_a_list_returns_empty():
    assert _parse_vlm_objects('{"objects": "car"}') == []


def test_missing_objects_key_returns_empty():
    assert _parse_vlm_objects('{"detections": ["car"]}') == []


def test_entries_over_three_words_dropped():
    text = '{"objects": ["this entry has way too many words", "car"]}'
    assert _parse_vlm_objects(text) == ["car"]


def test_single_character_entries_dropped():
    text = '{"objects": ["a", "car"]}'
    assert _parse_vlm_objects(text) == ["car"]


def test_entries_with_angle_brackets_dropped():
    text = '{"objects": ["<unk>", "car"]}'
    assert _parse_vlm_objects(text) == ["car"]


def test_prose_before_and_after_json_is_ignored():
    text = 'Here is my answer:\n{"objects": ["car"]}\nLet me know if you need more.'
    assert _parse_vlm_objects(text) == ["car"]


def test_known_limitation_brace_in_trailing_prose_breaks_extraction():
    # Documents a REAL limitation, not a requirement: find("{")/rfind("}")
    # is a naive first/last match, so a stray {brace} in trailing prose
    # after the real JSON object pulls in unparseable garbage and the
    # whole response is discarded. If the extraction logic is ever made
    # smarter, this test should start asserting ["car"] instead of [] —
    # until then it protects the safe fail-empty fallback from regressing
    # into a crash.
    text = '{"objects": ["car"]} (this is a common object {like a vehicle})'
    assert _parse_vlm_objects(text) == []
