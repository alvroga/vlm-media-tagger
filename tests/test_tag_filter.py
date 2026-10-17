"""tag_filter.clean_tag/clean_tags — the last line of defense before a
noun reaches the output as a tag. Person-noun ban must hold at WORD
level (found live 2026-07-10: multi-word phrases like "a man" and
possessives like "person's arm" sailed past an exact-match blocklist,
and color-stripping could itself MANUFACTURE a bare person noun —
"red hand" -> "hand")."""
from tag_filter import clean_tag, clean_tags


def test_plain_tag_passes_through():
    assert clean_tag("car") == "car"


def test_color_prefix_stripped():
    assert clean_tag("red shirt") == "shirt"


def test_multiple_colors_and_connectors_stripped():
    assert clean_tag("black and red scarf") == "scarf"


def test_pure_color_tag_dropped():
    assert clean_tag("black") is None
    assert clean_tag("red and white") is None


def test_junk_word_dropped():
    assert clean_tag("text") is None
    assert clean_tag("background") is None


def test_color_stripping_reveals_junk_word_still_dropped():
    assert clean_tag("white text") is None


def test_person_noun_exact_match_dropped():
    assert clean_tag("man") is None
    assert clean_tag("person") is None


def test_person_noun_multiword_phrase_dropped():
    # the actual live bug: exact-match blocklists miss these
    assert clean_tag("a man") is None
    assert clean_tag("out-of-focus person's face") is None


def test_person_noun_possessive_normalized():
    assert clean_tag("person's arm") is None
    assert clean_tag("man's hand") is None


def test_person_noun_curly_apostrophe_normalized():
    assert clean_tag("person’s arm") is None


def test_color_stripping_can_manufacture_a_bare_person_noun():
    # "red hand" -> strip "red" -> "hand", which IS in PERSON_NOUNS.
    # clean_tag must catch this on the ORIGINAL word list, not just the
    # post-strip result — verify the manufactured case is still dropped.
    assert clean_tag("red hand") is None


def test_non_person_word_containing_person_substring_survives():
    # "hand truck" must NOT be dropped just because "hand" is a person
    # word when standalone — false-drops here are accepted (a missing tag
    # is cheaper than a wrong one in an index) but real objects with
    # unrelated names shouldn't be casualties unless they ARE the word
    assert clean_tag("hand truck") is None  # "hand" is still its own word — dropped by design
    assert clean_tag("handrail") == "handrail"  # not a separate "hand" word at all


def test_language_plus_texty_word_dropped():
    # the lazy-tag pattern found 2026-07-11 building the multilingual
    # golden entries: dense/unfamiliar-script scenes sometimes get a
    # generic "<language> text" tag instead of real transcription
    assert clean_tag("korean text banners") is None
    assert clean_tag("korean text signs") is None
    assert clean_tag("russian text sign") is None
    assert clean_tag("chinese characters") is None
    assert clean_tag("japanese text") is None


def test_language_plus_real_noun_survives():
    # must NOT over-match: "japanese flag"/"korean bbq" are real content,
    # not a lazy placeholder — only ALL-texty-words after the language
    # triggers the drop
    assert clean_tag("japanese flag") == "japanese flag"
    assert clean_tag("korean bbq") == "korean bbq"
    assert clean_tag("chinese lantern") == "chinese lantern"
    assert clean_tag("chinese") == "chinese"


def test_watermark_and_overlay_debris_dropped():
    assert clean_tag("text overlay") is None
    assert clean_tag("hd watermark") is None
    assert clean_tag("flight information text") is None


def test_text_prefix_suffix_dropped():
    assert clean_tag("text 1974") is None
    assert clean_tag("title text") is None


def test_real_readable_brand_text_survives():
    # the whole point of the watermark/text guards: actual readable
    # brand/title words must NOT be caught by them
    assert clean_tag("fedex") == "fedex"
    assert clean_tag("aperol spritz") == "aperol spritz"


def test_empty_and_whitespace_tags_dropped():
    assert clean_tag("") is None
    assert clean_tag("   ") is None


def test_case_normalized_to_lowercase():
    assert clean_tag("Car") == "car"
    assert clean_tag("RED SHIRT") == "shirt"


def test_clean_tags_dedups_preserving_first_occurrence_order():
    result = clean_tags(["car", "tree", "car", "red shirt", "shirt"])
    assert result == ["car", "tree", "shirt"]


def test_clean_tags_drops_junk_and_person_entries_from_a_mixed_list():
    result = clean_tags(["car", "man", "text", "a woman", "tree"])
    assert result == ["car", "tree"]


def test_clean_tags_empty_list_returns_empty():
    assert clean_tags([]) == []
