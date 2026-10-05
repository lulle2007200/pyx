from contextlib import contextmanager
from io import StringIO
from typing import Type

from pyx.scanner import Lines, Pattern, Pos, Scanner, StrLeaf, StrMultiSegment, StrSegment


@contextmanager
def raises(exc: Type[Exception]):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def sc(s: str) -> Scanner:
    return Scanner(StringIO(s))


# ---------------------------------------------------------------- Pos / Pattern

def test_pos_arithmetic():
    assert Pos(1, 2) + Pos(3, 4) == Pos(4, 6)
    assert Pos(1, 2) - Pos(3, 4) == Pos(-2, -2)
    assert str(Pos(1, 2)) == "(1, 2)"
    with raises(TypeError):
        Pos(1, 2) + (3, 4)


def test_pattern_compiles_once():
    p = Pattern(r"a+")
    assert isinstance(p, str)
    assert p.compiled is p.compiled
    assert (m := p.compiled.match("aaab")) and m.end() == 3


# ---------------------------------------------------------------- string views

def test_leaf_slice():
    leaf = StrLeaf("hello")[1:3]
    assert leaf.text == "el" and len(leaf) == 2


def test_segment_basic_and_full_range_is_not_copied():
    leaf = StrLeaf("hello")
    assert StrSegment(leaf, slice(1, 4)).text == "ell"
    assert StrSegment(leaf).text is leaf.text


def test_segment_of_segment_flattens():
    seg = StrSegment("abcdef", slice(1, 5))      # "bcde"
    sub = seg[1:3]                               # "cd"
    assert sub.text == "cd"
    assert sub.buffer is seg.buffer
    assert (sub.start, sub.stop) == (2, 4)


def test_segment_rejects_step():
    with raises(ValueError):
        StrSegment("abc", slice(None, None, 2))


def test_segment_over_multisegment():
    multi = StrMultiSegment([StrLeaf("ab\n"), StrLeaf("cd\n")])
    assert StrSegment(multi, slice(1, 5)).text == "b\ncd"


def test_multisegment_basics():
    seg = StrMultiSegment([StrLeaf("ab\n"), StrLeaf("cd")])
    assert len(seg) == 5
    assert seg.text == "ab\ncd"
    assert seg[1:4].text == "b\nc"
    assert seg[3:].text == "cd"
    assert seg[1:4][1:2].text == "\n"            # slice of a slice


def test_offset_to_segment_boundaries():
    seg = StrMultiSegment([StrLeaf("ab\n"), StrLeaf("cd")])
    assert seg.offset_to_segment(0) == (0, 0)
    assert seg.offset_to_segment(2) == (0, 2)
    assert seg.offset_to_segment(3) == (1, 0)                # start-biased
    assert seg.offset_to_segment(3, end=True) == (0, 3)      # end-biased
    assert seg.offset_to_segment(5) == (2, 0)
    assert seg.offset_to_segment(5, end=True) == (1, 2)
    with raises(ValueError):
        seg.offset_to_segment(6)
    with raises(ValueError):
        seg.offset_to_segment(-1)


# ---------------------------------------------------------------- Lines

def test_lines_advance_commit_reset():
    ln = Lines()
    ln.append(StrLeaf("ab\n"))
    ln.append(StrLeaf("cd\n"))
    assert ln.text == "ab\ncd\n"
    assert ln.pos() == Pos(1, 0) and not ln.is_empty

    ln.advance(2)
    ln.reset()
    assert ln.cursor == 0                         # reset rewinds, drops nothing
    assert ln.lines.text == "ab\ncd\n"

    ln.advance(4)
    assert ln.pos() == Pos(2, 1)
    assert ln.lines.text == "ab\ncd\n"            # nothing dropped before commit
    ln.commit()
    assert ln.lines.text == "cd\n"
    assert ln.pos() == Pos(2, 1) and ln.cursor == 1


def test_lines_advance_bounds():
    ln = Lines()
    ln.append(StrLeaf("ab\n"))
    with raises(ValueError):
        ln.advance(4)
    with raises(ValueError):
        ln.advance(-1)


def test_lines_text_cache():
    s = "ab\n"
    ln = Lines()
    ln.append(StrLeaf(s))
    assert ln.text is s                           # single line: no copy

    ln.append(StrLeaf("cd\n"))
    joined = ln.text
    assert joined == "ab\ncd\n"
    ln.advance(1)
    ln.commit()                                   # nothing dropped: cache kept
    assert ln.text is joined
    ln.advance(3)
    ln.commit()                                   # first line dropped: cache rebuilt
    assert ln.text == "cd\n"


# ---------------------------------------------------------------- consume basics

def test_full_line_without_newline():
    s = sc("foobar")
    m = s.consume(Pattern(r"foobar"))
    assert m is not None
    assert m.text.text == "foobar" and m.len == 6
    assert (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(1, 6))   # end stays on row 1
    assert s.lines.pos() == Pos(2, 0)                           # but the line is dropped
    assert s.consume(Pattern(r".")) is None                           # EOF


def test_consume_newline_then_next_line():
    s = sc("ab\ncd\n")
    m = s.consume(Pattern(r"ab\n"))
    assert m and (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(1, 3))
    m = s.consume(Pattern(r"cd"))
    assert m and (m.pos_start, m.pos_end) == (Pos(2, 0), Pos(2, 2))
    assert s.lines.pos() == Pos(2, 2)


def test_sequential_consume_within_line():
    s = sc("foo bar\n")
    assert (m := s.consume(Pattern(r"foo"))) and (m.text.text == "foo")
    assert s.lines.pos() == Pos(1, 3)
    assert s.lines.rest.text == " bar\n"
    assert (m := s.consume(Pattern(r" "))) and (m.text.text == " ")
    m = s.consume(Pattern(r"bar"))
    assert m and m.pos_start == Pos(1, 4) and m.pos_end == Pos(1, 7)


def test_failed_consume_does_not_move():
    s = sc("abc\n")
    assert s.consume(Pattern(r"x")) is None
    assert s.lines.pos() == Pos(1, 0) and s.lines.cursor == 0
    assert (m := s.consume(Pattern(r"abc"))) and (m.text.text == "abc")    # still works afterwards


def test_first_matching_pattern_wins():
    s = sc("abc")
    m = s.consume(Pattern(r"x"), Pattern(r"ab"), Pattern(r"abc"))
    assert m and m.text.text == "ab"


def test_empty_input():
    assert sc("").consume(Pattern(r".+")) is None
    assert sc("").consume(Pattern(r".")) is None
    m = sc("").consume(Pattern(r".*"))                  # empty match at EOF
    assert m is not None and m.len == 0 and m.text.text == ""
    assert (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(1, 0))


# ---------------------------------------------------------------- groups / Match

def test_named_groups():
    s = sc("foo-12")
    m = s.consume(Pattern(r"(?P<word>\w+?)-(?P<num>\d+)"))
    assert m
    assert {k: v.text for k, v in m.groups.items()} == {"word": "foo", "num": "12"}


def test_group_offsets_after_partial_consume():
    s = sc("xx foo\n")
    s.consume(Pattern(r"xx "))
    m = s.consume(Pattern(r"(?P<w>foo)"))
    assert m and m.groups["w"].text == "foo"


def test_optional_group_not_participating():
    s = sc("foo")
    m = s.consume(Pattern(r"(?P<a>\w+)(?P<b>-\d+)?"))
    assert m and set(m.groups) == {"a"}
    assert m.groups["a"].text == "foo"


def test_match_str():
    s = sc("foo bar")
    m = s.consume(Pattern(r"(?P<w>foo)"))
    assert str(m) == "('foo', {'w'='foo'}, (1, 0), (1, 3))"


def test_match_survives_later_commits():
    s = sc("foo\nbar\n")
    m1 = s.consume(Pattern(r"(?P<w>foo)\n"))
    s.consume(Pattern(r"bar"))                          # drops lines after m1 was made
    assert m1
    assert m1.text.text == "foo\n"
    assert m1.groups["w"].text == "foo"
    assert m1.lines.text == "foo\n"


# ---------------------------------------------------------------- match (peek)

def test_match_does_not_consume():
    s = sc("foo bar\n")
    m = s.match(Pattern(r"foo"))
    assert m and m.text.text == "foo"
    assert (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(1, 3))
    assert s.lines.pos() == Pos(1, 0)             # cursor unchanged
    m2 = s.consume(Pattern(r"foo"))                     # same text is still there
    assert m2 and m2.text.text == "foo"


def test_match_multiline_peek_keeps_lines_loaded():
    s = sc("a\nEND\nrest\n")
    m = s.match(Pattern(r"(?s).*?END"), multiline=True)
    assert m and m.text.text == "a\nEND"
    assert s.lines.pos() == Pos(1, 0)
    assert (m := s.consume(Pattern(r"a\n"))) and (m.text.text == "a\n")     # uses the already loaded lines
    assert s.lines.pos() == Pos(2, 0)


# ---------------------------------------------------------------- multiline

def test_multiline_spans_lines():
    s = sc("a\nEND\nrest\n")
    assert s.consume(Pattern(r"(?s).*?END")) is None    # non-multiline only sees "a\n"
    m = s.consume(Pattern(r"(?s).*?END"), multiline=True)
    assert m and m.text.text == "a\nEND"
    assert (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(2, 3))
    assert s.lines.pos() == Pos(2, 3)
    assert (m := s.consume(Pattern(r"\n"))) and (m.text.text == "\n")


def test_multiline_hits_eof():
    s = sc("a\nb\n")
    assert s.consume(Pattern(r"zzz"), multiline=True) is None
    assert s.lines.pos() == Pos(1, 0)             # rewound
    assert (m := s.consume(Pattern(r"a\n"))) and (m.text.text == "a\n")     # nothing was lost


def test_consume_needs_multiline_to_see_next_line():
    s = sc("foo\nbar\nbaz\n")
    s.consume(Pattern("foo\n"))
    s.consume(Pattern("bar"))
    assert s.consume(Pattern("\nbaz")) is None
    m = s.consume(Pattern("\nbaz"), multiline=True)
    assert m and (m.pos_start, m.pos_end) == (Pos(2, 3), Pos(3, 3))


# ---------------------------------------------------------------- many

def test_many_spans_lines():
    s = sc("ab\ncd\n}")
    m = s.consume(Pattern(r"[^}]+"), many=True)
    assert m and m.text.text == "ab\ncd\n" and m.len == 6
    assert (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(2, 3))
    m = s.consume(Pattern(r"}"))
    assert m and (m.pos_start, m.pos_end) == (Pos(3, 0), Pos(3, 1))


def test_many_stops_when_pattern_stops():
    s = sc("aaab")
    m = s.consume(Pattern(r"a"), many=True)
    assert m and m.text.text == "aaa" and m.len == 3
    assert (m := s.consume(Pattern(r"b"))) and (m.pos_start == Pos(1, 3))


def test_many_until_eof():
    s = sc("ab\ncd\n")
    m = s.consume(Pattern(r"[a-z\n]+"), many=True)
    assert m and m.text.text == "ab\ncd\n"
    assert (m.pos_start, m.pos_end) == (Pos(1, 0), Pos(2, 3))
    assert s.consume(Pattern(r".")) is None


def test_many_with_possibly_empty_pattern_terminates():
    s = sc("aab")
    m = s.consume(Pattern(r"a*"), many=True)
    assert m and m.text.text == "aa"

    s = sc("b")
    m = s.consume(Pattern(r"a*"), many=True)
    assert m is not None and m.len == 0 and m.text.text == ""
    assert s.lines.pos() == Pos(1, 0)


def test_many_group_keeps_last_occurrence():
    s = sc("abc")
    m = s.consume(Pattern(r"(?P<c>[a-z])"), many=True)
    assert m and m.text.text == "abc"
    assert m.groups["c"].text == "c"


def test_many_no_match_returns_none():
    s = sc("xyz")
    assert s.consume(Pattern(r"a"), many=True) is None
    assert s.lines.pos() == Pos(1, 0)


# ---------------------------------------------------------------- advance

def test_advance_within_line():
    s = sc("foo bar\n")
    s.consume(Pattern(r"foo"))
    s.advance(1)
    m = s.consume(Pattern(r"bar"))
    assert m and m.pos_start == Pos(1, 4)


def test_advance_to_end_of_line_drops_it():
    s = sc("ab\ncd\n")
    s.consume(Pattern(r"a"))
    s.advance(2)                                  # eats "b\n"
    assert s.lines.pos() == Pos(2, 0)
    assert (m := s.consume(Pattern(r"cd"))) and (m.pos_start == Pos(2, 0))


def test_advance_beyond_loaded_lines_raises():
    s = sc("ab\n")
    with raises(ValueError):
        s.advance(1)                              # nothing loaded yet
    s.consume(Pattern(r"a"))
    with raises(ValueError):
        s.advance(5)


# ---------------------------------------------------------------- integration

def test_tokenize_two_lines():
    ws, nl, name, num, eq = Pattern(r"[ \t]+"), Pattern(r"\n"), Pattern(r"[a-z]+"), Pattern(r"\d+"), Pattern(r"=")
    s = sc("x = 12\ny = 3\n")
    got = []
    while m := s.consume(ws, nl, name, num, eq):
        got.append((m.text.text, tuple(m.pos_start), tuple(m.pos_end)))
    assert got == [
        ("x", (1, 0), (1, 1)), (" ", (1, 1), (1, 2)), ("=", (1, 2), (1, 3)),
        (" ", (1, 3), (1, 4)), ("12", (1, 4), (1, 6)), ("\n", (1, 6), (1, 7)),
        ("y", (2, 0), (2, 1)), (" ", (2, 1), (2, 2)), ("=", (2, 2), (2, 3)),
        (" ", (2, 3), (2, 4)), ("3", (2, 4), (2, 5)), ("\n", (2, 5), (2, 6)),
    ]