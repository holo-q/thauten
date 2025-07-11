import sys
# from pathlib import Path
# s = str(Path(__file__).parent / "errloom")
# s = str(Path(".venv/lib/python3.12/site-packages/"))
# sys.path.append(s)

import pytest

from errloom.holoware import (
    ClassSpan,
    ContextResetSpan,
    EgoSpan,
    Holoware,
    ObjSpan,
    SampleSpan,
    TextSpan,
)
from errloom.holoware_parse import (_build_class_span, _build_context_reset_span, _build_ego_or_sampler_span, _build_obj_span, filter_comments, HolowareParser, parse_span_tag)

# --- Tests for filter_comments ---

def test_filter_comments_empty():
    assert filter_comments("") == ""

def test_filter_comments_no_comments():
    content = "line 1\nline 2"
    assert filter_comments(content) == content

def test_filter_comments_only_comments():
    content = "# comment 1\n# comment 2"
    assert filter_comments(content) == "\n"

def test_filter_comments_mixed():
    content = "line 1\n# comment\nline 2"
    assert filter_comments(content) == "line 1\n\nline 2"

def test_filter_comments_with_whitespace():
    content = "  # comment with leading whitespace"
    assert filter_comments(content) == ""

def test_filter_comments_inline_content_not_filtered():
    content = "line_with_hash = '#value'"
    assert filter_comments(content) == content

# --- Tests for parse_span_tag ---

def test_parse_span_tag_simple():
    base, kargs, kwargs = parse_span_tag("MyClass")
    assert base == "MyClass"
    assert kargs == []
    assert kwargs == {}

def test_parse_span_tag_with_kargs():
    base, kargs, kwargs = parse_span_tag("MyClass arg1 arg2")
    assert base == "MyClass"
    assert kargs == ["arg1", "arg2"]
    assert kwargs == {}

def test_parse_span_tag_with_kwargs():
    base, kargs, kwargs = parse_span_tag("MyClass key1=val1 key2=val2")
    assert base == "MyClass"
    assert kargs == []
    assert kwargs == {"key1": "val1", "key2": "val2"}

def test_parse_span_tag_mixed_args():
    base, kargs, kwargs = parse_span_tag("MyClass arg1 key1=val1 arg2")
    assert base == "MyClass"
    assert kargs == ["arg1", "arg2"]
    assert kwargs == {"key1": "val1"}

def test_parse_span_tag_with_special_attr():
    base, kargs, kwargs = parse_span_tag("MyClass <>something")
    assert base == "MyClass"
    assert kargs == []
    assert kwargs == {"<>": "something"}

def test_parse_span_tag_with_empty_special_attr_raises_error():
    with pytest.raises(ValueError, match="Empty <> attribute"):
        parse_span_tag("MyClass <>")

# --- Tests for Span Builders ---

def test_build_class_span():
    out = []
    _build_class_span(out, "MyClass", ["arg"], {"kw": "val"})
    assert len(out) == 1
    span = out[0]
    assert isinstance(span, ClassSpan)
    assert span.class_name == "MyClass"
    assert span.kargs == ["arg"]
    assert span.kwargs == {"kw": "val"}

def test_build_ego_or_sampler_span_ego_only():
    out = []
    _build_ego_or_sampler_span(out, "o_o", [], {})
    assert len(out) == 1
    span = out[0]
    assert isinstance(span, EgoSpan)
    assert span.ego == "user"

def test_build_ego_or_sampler_span_with_uuid():
    out = []
    _build_ego_or_sampler_span(out, "@_@:123", [], {})
    assert len(out) == 1
    span = out[0]
    assert isinstance(span, EgoSpan)
    assert span.ego == "assistant"
    assert span.uuid == "123"

def test_build_ego_or_sampler_span_as_sampler():
    out = []
    _build_ego_or_sampler_span(out, "o_o:id", ["karg"], {"goal": "test"})
    assert len(out) == 2
    ego_span, sampler_span = out
    assert isinstance(ego_span, EgoSpan)
    assert ego_span.ego == "user"
    assert isinstance(sampler_span, SampleSpan)
    assert sampler_span.uuid == "id"
    assert sampler_span.kargs == ["karg"]
    assert sampler_span.goal == "test"
    assert sampler_span.kwargs == {}

def test_build_context_reset_span():
    out = []
    handler = _build_context_reset_span(train=True)
    handler(out, "+++", [], {})
    assert len(out) == 1
    span = out[0]
    assert isinstance(span, ContextResetSpan)
    assert span.train is True

def test_build_obj_span():
    out = []
    _build_obj_span(out, "var1|var2", ["karg"], {})
    assert len(out) == 1
    span = out[0]
    assert isinstance(span, ObjSpan)
    assert span.var_ids == ["var1", "var2"]
    assert span.kargs == ["karg"]

# --- Tests for HolowareParser ---

@pytest.mark.parametrize(
    "code, expected_details",
    [
        ("", []),
        ("   \n\t ", []),
        (
            "just text",
            [
                {"type": EgoSpan, "attrs": {"ego": "system"}},
                {"type": TextSpan, "attrs": {"text": "just text"}},
            ],
        ),
        ("<|o_o|>", [{"type": EgoSpan, "attrs": {"ego": "user"}}]),
        ("<|@_@|>", [{"type": EgoSpan, "attrs": {"ego": "assistant"}}]),
        (
            "<|o_o|>Hello",
            [
                {"type": EgoSpan, "attrs": {"ego": "user"}},
                {"type": TextSpan, "attrs": {"text": "Hello"}},
            ],
        ),
        (
            "Hello<|o_o|>",
            [
                {"type": EgoSpan, "attrs": {"ego": "system"}},
                {"type": TextSpan, "attrs": {"text": "Hello"}},
                {"type": EgoSpan, "attrs": {"ego": "user"}},
            ],
        ),
        (
            "<|o_o|>Hello<|@_@|>World",
            [
                {"type": EgoSpan, "attrs": {"ego": "user"}},
                {"type": TextSpan, "attrs": {"text": "Hello"}},
                {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                {"type": TextSpan, "attrs": {"text": "World"}},
            ],
        ),
        (
            "text <|o_o|> text",
            [
                {"type": EgoSpan, "attrs": {"ego": "system"}},
                {"type": TextSpan, "attrs": {"text": "text "}},
                {"type": EgoSpan, "attrs": {"ego": "user"}},
                {"type": TextSpan, "attrs": {"text": "text"}},
            ],
        ),
        # Should not have duplicate ego spans
        ("<|o_o|><|o_o|>", [{"type": EgoSpan, "attrs": {"ego": "user"}}]),
        ("<|+++|>", [{"type": ContextResetSpan, "attrs": {"train": True}}]),
        ("<|===|>", [{"type": ContextResetSpan, "attrs": {"train": False}}]),
        (
            "<|o_o|><|MyClass|>",
            [
                {"type": EgoSpan, "attrs": {"ego": "user"}},
                {"type": ClassSpan, "attrs": {"class_name": "MyClass"}},
            ],
        ),
        (
            "<|@_@|><|my_obj|>",
            [
                {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                {"type": ObjSpan, "attrs": {"var_ids": ["my_obj"]}},
            ],
        ),
        (
            "<|@_@|><|obj1|obj2|>",
            [
                {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                {"type": ObjSpan, "attrs": {"var_ids": ["obj1", "obj2"]}},
            ],
        ),
        (
            "<|@_@ goal=run|>",
            [
                {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                {"type": SampleSpan, "attrs": {"goal": "run"}},
            ],
        ),
    ],
)
def test_holoware_parser_basic(code, expected_details):
    spans = HolowareParser(code).parse().spans
    assert len(spans) == len(expected_details)
    for i, details in enumerate(expected_details):
        span = spans[i]
        assert isinstance(span, details["type"])
        if "attrs" in details:
            for attr, value in details["attrs"].items():
                assert getattr(span, attr) == value

def test_parser_implicit_system_ego():
    holoware = HolowareParser("some text").parse()
    assert len(holoware.spans) == 2
    assert isinstance(holoware.spans[0], EgoSpan)
    assert holoware.spans[0].ego == "system"
    assert isinstance(holoware.spans[1], TextSpan)
    assert holoware.spans[1].text == "some text"

def test_parser_context_reset_clears_ego():
    code = "<|o_o|>hello<|+++|>world"
    holoware = HolowareParser(code).parse()
    # Expected: Ego(user), Text(hello), Reset, Ego(system), Text(world)
    assert len(holoware.spans) == 5
    assert isinstance(holoware.spans[0], EgoSpan)
    assert holoware.spans[0].ego == "user"
    assert isinstance(holoware.spans[2], ContextResetSpan)
    assert isinstance(holoware.spans[3], EgoSpan)
    assert holoware.spans[3].ego == "system"
    assert isinstance(holoware.spans[4], TextSpan)
    assert holoware.spans[4].text == "world"

def test_parser_span_before_ego_raises_error():
    with pytest.raises(ValueError):
        HolowareParser("<|MyClass|>").parse()
    with pytest.raises(ValueError):
        HolowareParser("<|my_obj|>").parse()
    with pytest.raises(ValueError):
        # SampleSpan is created with EgoSpan, so this is tricky
        # The rule is on SampleSpan, ObjSpan, ClassSpan
        # This input <|goal=run|> will be parsed as an ObjSpan
        HolowareParser("<|goal=run|>").parse()

def test_parser_unclosed_tag_raises_error():
    with pytest.raises(ValueError, match="Unclosed tag"):
        HolowareParser("<|o_o").parse()

def test_parser_merges_consecutive_text_spans():
    # This is implicitly tested by other tests, but good to be explicit.
    # The parser logic merges text spans under the same ego.
    code = "<|o_o|>one two three"
    holoware = HolowareParser(code).parse()
    assert len(holoware.spans) == 2
    assert isinstance(holoware.spans[1], TextSpan)
    assert holoware.spans[1].text == "one two three"

def test_parser_handles_whitespace():
    code = "  <|o_o|>  \n  text  "
    holoware = HolowareParser(code).parse()
    # Expected: Ego(user), Text(text  )
    # Leading whitespace before tag is ignored if it's the first thing.
    # Leading whitespace of text after tag is stripped.
    # Trailing whitespace of text is preserved.
    assert len(holoware.spans) == 2
    assert isinstance(holoware.spans[0], EgoSpan)
    assert isinstance(holoware.spans[1], TextSpan)
    assert holoware.spans[1].text == "text  "

def test_parser_indented_block_simple():
    code = """<|o_o|>
<|MyClass|>
    Some indented text.
"""
    holoware = HolowareParser(code).parse()
    assert len(holoware.spans) == 2
    class_span = holoware.spans[1]
    assert isinstance(class_span, ClassSpan)
    assert class_span.class_name == "MyClass"
    assert class_span.body is not None

    body = class_span.body
    assert body is not None
    assert len(body.spans) == 2 # system ego + text
    assert isinstance(body.spans[0], EgoSpan) and body.spans[0].ego == "system"
    assert isinstance(body.spans[1], TextSpan)
    assert body.spans[1].text == "Some indented text."

def test_parser_indented_block_complex():
    code = """<|o_o|>
<|Container|>
    <|@_@|>
    <|Item|>
        Nested item
"""
    holoware = HolowareParser(code).parse()
    container_span = holoware.spans[1]
    assert isinstance(container_span, ClassSpan)
    assert container_span.class_name == "Container"

    body = container_span.body
    assert body is not None
    assert len(body.spans) == 2 # assistant ego, Item class
    assert isinstance(body.spans[0], EgoSpan) and body.spans[0].ego == "assistant"

    item_span = body.spans[1]
    assert isinstance(item_span, ClassSpan) and item_span.class_name == "Item"

    nested_body = item_span.body
    assert nested_body is not None
    assert len(nested_body.spans) == 2 # system ego, text
    assert isinstance(nested_body.spans[1], TextSpan)
    assert nested_body.spans[1].text == "Nested item"

def test_parser_indented_block_no_block():
    code = "<|o_o|><|MyClass|>\nNot indented."
    holoware = HolowareParser(code).parse()
    assert len(holoware.spans) == 3
    class_span = holoware.spans[1]
    assert isinstance(class_span, ClassSpan)
    assert class_span.body is None
    text_span = holoware.spans[2]
    assert isinstance(text_span, TextSpan)
    assert text_span.text == "Not indented."