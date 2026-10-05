from dataclasses import dataclass
from typing import Protocol, TextIO, override, Optional, NamedTuple, Sequence, cast
from functools import reduce
import re

@dataclass
class Match:
    pos_start: Pos
    pos_end: Pos

    groups: dict[str, StrRef]
    lines: StrRef

    len: int

    text: StrRef

    def __str__(self) -> str:
        groups = ", ".join(f"{k!r}={v.text!r}" for k, v in self.groups.items())
        return f"({self.text.text!r}, {{{groups}}}, {self.pos_start}, {self.pos_end})"
    
class Pattern(str):
    @property
    def compiled(self) -> re.Pattern[str]:
        try:
            return self._compiled
        except AttributeError:
            self._compiled = re.compile(cast(str, self))
            return self._compiled

class Pos(NamedTuple):
    row: int
    col: int

    @override
    def __add__(self, other: object) -> Pos:
        if isinstance(other, Pos):
            return Pos(self.row + other.row,self.col + other.col)
        return NotImplemented

    def __sub__(self, other: object) -> Pos:
        if isinstance(other, Pos):
            return Pos(self.row - other.row,self.col - other.col)
        return NotImplemented

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"

class StrRef(Protocol):
    @property
    def text(self) -> str:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key: slice[Optional[int]], /) -> StrRef:
        ...

class StrLeaf:
    def __init__(self, buffer: str, /):
        self._buffer = buffer

    @property
    def buffer(self) -> str:
        return self._buffer

    @property
    def text(self) -> str:
        return self.buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, key: slice[Optional[int]], /) -> StrRef:
        return StrLeaf(self.buffer[key])

class StrSegment:
    def __init__(self, buffer: StrRef | str, range: Optional[slice[Optional[int]]]=None, /):
        if isinstance(buffer, str):
            buffer = StrLeaf(buffer)
        if range is not None:
            start, stop, step = range.indices(len(buffer))
            if step != 1:
                raise ValueError()
            self._start = start
            self._stop = stop
        else:
            self._start = 0
            self._stop = len(buffer)
        self._buffer = buffer

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def buffer(self) -> StrRef:
        return self._buffer

    @property
    def text(self) -> str:
        if self.start == 0 and self.stop == len(self.buffer):
            return self.buffer.text
        else:
            return self._buffer[self.start:self.stop].text

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, key: slice[Optional[int]], /) -> StrSegment:
        start, stop, step = key.indices(len(self))
        return StrSegment(self.buffer, slice(self.start + start, self.start + stop, step))

class StrMultiSegment:
    def __init__(self, buffer: Sequence[StrRef], range: Optional[slice[Optional[int]]]=None, /):
        total_len = reduce(lambda total_len, segment: total_len + len(segment), buffer, 0)
        if range is not None:
            start, stop, step = range.indices(total_len)
            if step != 1:
                raise ValueError()
            self._start = start
            self._stop = stop
        else:
            self._start = 0
            self._stop = total_len
        self._total_len = total_len
        self._segments = tuple(buffer)

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def segments(self) -> Sequence[StrRef]:
        return self._segments

    @property
    def text(self) -> str:
        if len(self.segments) == 1:
            text = self.segments[0].text
        else:
            text = "".join(segment.text for segment in self.segments)
        if self.start != 0 or self.stop != self._total_len:
            return text[self.start:self.stop]

        return text

    def offset_to_segment(self, index: int, end: bool=False) -> tuple[int, int]:
        if not 0 <= index <= len(self):
            raise ValueError()
        index += self.start
        for i, segment in enumerate(self.segments):
            n = len(segment)
            if index < n or (end and index == n):
                return (i, index)
            else:
                index -= n
        return (len(self.segments), index)

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, key: slice[Optional[int]], /) -> StrMultiSegment:
        start, stop, step = key.indices(len(self))
        return StrMultiSegment(self.segments, slice(self.start + start, self.start + stop, step))

class Lines:
    def __init__(self) -> None:
        self.lines = StrMultiSegment(())
        self._row = 1
        self._cached: Optional[str] = None

        self._start = 0
        self._cursor: int = 0

    def pos(self, offset: int=0, end: bool=False) -> Pos:
        row, col = self.lines.offset_to_segment(self.cursor + offset, end=end)
        return Pos(self._row + row, col)

    @property
    def cursor(self):
        return self._cursor

    @property
    def rest(self) -> StrRef:
        return self.lines[self._start:]

    @property
    def text(self) -> str:
        if self._cached is None:
            self._cached = self.lines.text
        return self._cached

    @property
    def is_empty(self) -> bool:
        return self.cursor == len(self.lines)

    def advance(self, count: int) -> None:
        if not 0 <= self.cursor + count <= len(self.lines):
            raise ValueError()
        self._cursor += count

    def commit(self) -> None:
        dropped, new_start = self.lines.offset_to_segment(self.cursor)
        if dropped:
            self.lines = StrMultiSegment(self.lines.segments[dropped:])
            self._row += dropped
            self._cached = None
        self._start = self._cursor = new_start

    def reset(self) -> None:
        self._cursor = self._start

    def append(self, line: StrRef) -> None:
        self.lines = StrMultiSegment((*self.lines.segments, line))
        self._cached = None

class Scanner:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self.lines = Lines()

    def load_next_line(self) -> bool:
        line = self.read_line()
        if line is None:
            return False
        self.lines.append(line)
        return  True

    def read_line(self) -> Optional[StrRef]:
        line = self.stream.readline()
        if not line:
            return None
        return StrLeaf(line)

    def advance(self, count: int) -> None:
        self.lines.advance(count)
        self.lines.commit()

    def _match_single(self, *pattern: Pattern) -> Optional[re.Match[str]]:
        text, cursor = self.lines.text, self.lines.cursor
        for p in pattern:
            if m := p.compiled.match(text, cursor):
                return m
        return None

    def _match(self, *pattern: Pattern, multiline: bool=False, many: bool=False, commit: bool=False) -> Optional[Match]:
        if self.lines.is_empty:
            self.load_next_line()

        start_pos, start_offset  = self.lines.pos(end=True), self.lines.cursor
        spans: dict[str, tuple[int, int]] = {}
        matched = False

        while True:
            if self.lines.is_empty and matched:
                if not self.load_next_line():
                    break

            if m := self._match_single(*pattern):
                matched = True
                for name, index in m.re.groupindex.items():
                    if m.start(index) != -1:
                        spans[name] = m.span(index)
                empty_match = m.end() == self.lines.cursor
                self.lines.advance(m.end() - self.lines.cursor)

                if not many or empty_match:
                    break
            else:
                if matched or not multiline or not self.load_next_line():
                    break
                continue

        if not matched:
            self.lines.reset()
            return None

        end_offset = self.lines.cursor
        end_pos = self.lines.pos(end=True)
        n = end_offset - start_offset
        lines = self.lines.lines
        res =  Match(start_pos, end_pos,
                     {name: StrSegment(lines, slice(*span)) for name, span in spans.items()},
                     self.lines.lines,
                     n,
                     StrSegment(self.lines.lines, slice(start_offset, end_offset)))

        if commit:
            self.lines.commit()
        else:
            self.lines.reset()

        return res

    def match(self, *pattern: Pattern, multiline: bool=False, many: bool=False) -> Optional[Match]:
        if m := self._match(*pattern, multiline=multiline, many=many, commit=False):
            return m

    def consume(self, *pattern: Pattern, multiline: bool=False, many: bool=False) -> Optional[Match]:
        if m := self._match(*pattern, multiline=multiline, many=many, commit=True):
            return m

