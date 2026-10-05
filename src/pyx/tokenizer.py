from io import StringIO
from typing import Iterator
from typing import TYPE_CHECKING, Iterable, override
from collections import deque
from pathlib import Path
import sys
import token
import itertools
import re
import argparse
from enum import (
	IntEnum, 
	auto
)
from dataclasses import dataclass
from pegen.tokenizer import BaseTokenizer, TokenInfo, TokenGenerator
import token

from pyx.scanner import Scanner, Match, Pattern, Pos, StrRef

if TYPE_CHECKING:
	from typing import (
		Optional,
		Self,
		Set,
		TextIO,
		List,
	)

__all__ = [
	"tokenize",
	"untokenize",
	"PyxTokenType",
	"PyxTokenInfo",
	"PyxTokenGenerator"
]

class TokenizeError(Exception):
	pass

def group(*choices: str, name: Optional[str]=None) -> str:
	if name:
		return f"(?P<{name}>" + "|".join(choices) + ")"
	else:
		return "(?:" + "|".join(choices) + ")"

def maybe(*choices: str) -> str:
	return group(*choices) + "?"


Eof = Pattern(r"$")
FullLine = Pattern(r"(?:[^\\\r\n]|\\\r\n|\\[^\r])*\r?\n")
Newline = Pattern(r"(?:\n|\r\n)")
Whitespace = Pattern(r"[ \f\t]*")
Comment = Pattern(r"#[^\r\n]*")
LineCont = Pattern(r"\\(?:\r\n|\n)")

Name = Pattern(r"\w+")

HexNum = Pattern(r"0[xX](?:_?[0-9a-fA-F])+")
BinNum = Pattern(r"0[bB](?:_?[01])+")
OctNum = Pattern(r"0[oO](?:_?[0-7])+")
DecNum = Pattern(r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)")
IntNum = Pattern(group(HexNum, BinNum, OctNum, DecNum))

Exponent = Pattern(r"[eE][-+]?[0-9](?:_?[0-9])*")
PointFloat = Pattern(group(r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?",
                           r"\.[0-9](?:_?[0-9])*") + \
                     maybe(Exponent))
ExpFloat = Pattern(r"[0-9](?:_?[0-9])*" + Exponent)
FloatNum = Pattern(group(PointFloat, ExpFloat))

ImagNum = Pattern(group(r"[0-9](?:_?[0-9])*[jJ]", FloatNum + r"[jJ]"))

Number = Pattern(group(ImagNum, FloatNum, IntNum))

def all_string_prefixes(letters: str) -> Set[str]:
	result: Set[str] = set()
	for t in itertools.permutations(letters):
		for u in itertools.product(*[(c, c.upper()) for c in t]):
			result.add("".join(u))
	return result

FStringPrefixes = {
	p for p in (all_string_prefixes("f")  | 
                all_string_prefixes("fr") | 
                all_string_prefixes("t")  | 
                all_string_prefixes("tr")) 
    if p.lower().count("f") == 1 or p.lower().count("t") == 1
}

PlainStringPrefixes = ((
	all_string_prefixes("")  | 
	all_string_prefixes("r") | 
	all_string_prefixes("b") | 
	all_string_prefixes("u") | 
	all_string_prefixes("br")
) - FStringPrefixes) | {""}

FstringPrefix = Pattern(group(*sorted(FStringPrefixes, key=len, reverse=True)))
PlainStringPrefix = Pattern(group(*sorted(PlainStringPrefixes, key=len, reverse=True)))

SingleQuoteStringStart1 = Pattern(r"'")
SingleQuoteStringStart3 = Pattern(r"'''")
DoubleQuoteStringStart1 = Pattern(r'"')
DoubleQuoteStringStart3 = Pattern(r'"""')

PlainStringStart1 = Pattern(PlainStringPrefix + group(SingleQuoteStringStart1,
                                                      DoubleQuoteStringStart1,
                                                      name="quot"))

PlainStringStart3 = Pattern(PlainStringPrefix + group(SingleQuoteStringStart3,
                                                      DoubleQuoteStringStart3,
                                                      name="quot"))

PlainStringStart = Pattern(PlainStringPrefix + group(SingleQuoteStringStart3, 
                                                     DoubleQuoteStringStart3, 
                                                     SingleQuoteStringStart1, 
                                                     DoubleQuoteStringStart1, 
                                                     name="quot"))

SingleQuoteStringMid1 = Pattern(r"(?:\\\r\n|\\[\s\S]|[^'\\\n\r])+")
DoubleQuoteStringMid1 = Pattern(r'(?:\\\r\n|\\[\s\S]|[^"\\\n\r])+')
SingleQuoteStringMid3 = Pattern(r"(?:\\[\s\S]|'?'?(?:[^'\\]|$))+")
DoubleQuoteStringMid3 = Pattern(r'(?:\\[\s\S]|"?"?(?:[^"\\]|$))+')

string_mid = {
	"'": SingleQuoteStringMid1,
	"'''": SingleQuoteStringMid3,
	'"': DoubleQuoteStringMid1,
	'"""': DoubleQuoteStringMid3,
}

SingleQuoteStringEnd1 = Pattern(r"'")
SingleQuoteStringEnd3 = Pattern(r"'''")
DoubleQuoteStringEnd1 = Pattern(r'"')
DoubleQuoteStringEnd3 = Pattern(r'"""')

string_end = {
	"'": SingleQuoteStringEnd1,
	"'''": SingleQuoteStringEnd3,
	'"': DoubleQuoteStringEnd1,
	'"""': DoubleQuoteStringEnd3,
}

PlainString1 = Pattern(group(
	SingleQuoteStringStart1 + maybe(SingleQuoteStringMid1) + SingleQuoteStringEnd1,
	DoubleQuoteStringStart1 + maybe(DoubleQuoteStringMid1) + DoubleQuoteStringEnd1,
))

PlainString3 = Pattern(group(
	SingleQuoteStringStart3 + maybe(SingleQuoteStringMid3) + SingleQuoteStringEnd3,
	DoubleQuoteStringStart3 + maybe(DoubleQuoteStringMid3) + DoubleQuoteStringEnd3,
))

PlainString = Pattern(group(PlainString1, PlainString3))

FStringStart = Pattern(FstringPrefix + group(SingleQuoteStringStart3,
                                             DoubleQuoteStringStart3,
                                             SingleQuoteStringStart1,
                                             DoubleQuoteStringStart1,
                                             name="quot"))

SingleQuoteFStringMid1 = Pattern(r"(?:\\\r\n|\\[^{}]|{{|}}|\\(?={|})|[^'\\{}\n\r])+")
DoubleQuoteFStringMid1 = Pattern(r'(?:\\\r\n|\\[^{}]|{{|}}|\\(?={|})|[^"\\{}\n\r])+')
SingleQuoteFStringMid3 = Pattern(r"(?:\\[^{}]|{{|}}|\\(?={|})|'?'?(?:[^'\\{}]|$))+")
DoubleQuoteFStringMid3 = Pattern(r'(?:\\[^{}]|{{|}}|\\(?={|})|"?"?(?:[^"\\{}]|$))+')

fstring_mid = {
	"'": SingleQuoteFStringMid1,
	"'''": SingleQuoteFStringMid3,
	'"': DoubleQuoteFStringMid1,
	'"""': DoubleQuoteFStringMid3,
}

fstring_end = {
	"'": SingleQuoteStringEnd1,
	"'''": SingleQuoteStringEnd3,
	'"': DoubleQuoteStringEnd1,
	'"""': DoubleQuoteStringEnd3,
}

operators = (
	"**=", "//=", ">>=", "<<=", "...", "!=", "->",
	":=", "<=", ">=", "==",
	"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "@=",
	"**", "//", "<<", ">>",
	"+", "-", "*", "/", "%", "@", "&", "|", "^", "~",
	"<", ">", "(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "=", "!",
)
Operator = Pattern(group(*[re.escape(o) for o in operators]))

pyx_operators = (
	"...", "=", "{", "}", ".", ":"
)

PyxOperator = Pattern(group(*[re.escape(o) for o in pyx_operators]))
PyxSpreadOperator = Pattern(re.escape("..."))
PyxInterpStartOperator = Pattern(re.escape("{"))
PyxInterpEndOperator = Pattern(re.escape("}"))

PyxTagStart = Pattern(r"<")
PyxCloseStart = Pattern(r"</")
PyxTagEnd = Pattern(r">")
PyxTagSelfClose = Pattern(r"/>")

PyxIdentifier = Pattern(r"[a-zA-Z](?:[a-zA-Z0-9_-]*[a-zA-Z0-9])?")

PyxWhitespace = Pattern(r"[\s]+")

# NOTE: No backslash escape sequences in favour of HTML character references
# PyxText = Pattern(r"(?:[^\\{}<>]|\\[\s\S])+")
PyxText = Pattern(r"(?:[^{}<>])+")

SingleQuotePyxString = Pattern(r"'[^']*'")
DoubleQuotePyxString = Pattern(r'"[^"]*"')

PyxString = Pattern(group(SingleQuotePyxString, DoubleQuotePyxString))

PyxOpenLookahead = Pattern(r"<[a-zA-Z>]")

PyxCommentStart = Pattern(r"<!--")
PyxCommentMid = Pattern(r"[\s\S]+")
PyxCommentEnd = Pattern(r"-->")
PyxComment = Pattern(PyxCommentStart + maybe(PyxCommentMid) + PyxCommentEnd)

LBrace = Pattern("{")

class PyxTokenType(IntEnum):
	NAME                = auto()
	NUMBER              = auto()
	STRING              = auto()
	FSTRING_END         = auto()
	FSTRING_MIDDLE      = auto()
	FSTRING_START       = auto()
	OP                  = auto()
	TYPE_COMMENT        = auto()
	DEDENT              = auto()
	INDENT              = auto()
	ENDMARKER           = auto()
	NEWLINE             = auto()
	ASYNC               = auto()
	AWAIT               = auto()
	SOFT_KEYWORD        = auto()
	AMPER               = auto()
	AMPEREQUAL          = auto()
	AT                  = auto()
	ATEQUAL             = auto()
	CIRCUMFLEX          = auto()
	CIRCUMFLEXEQUAL     = auto()
	COLON               = auto()
	COLONEQUAL          = auto()
	COMMA               = auto()
	DOT                 = auto()
	DOUBLESLASH         = auto()
	DOUBLESLASHEQUAL    = auto()
	DOUBLESTAR          = auto()
	DOUBLESTAREQUAL     = auto()
	ELLIPSIS            = auto()
	EQEQUAL             = auto()
	EQUAL               = auto()
	ERRORTOKEN          = auto()
	GREATER             = auto()
	GREATEREQUAL        = auto()
	ISEOF               = auto()
	ISNONTERMINAL       = auto()
	ISTERMINAL          = auto()
	LBRACE              = auto()
	LEFTSHIFT           = auto()
	LEFTSHIFTEQUAL      = auto()
	LESS                = auto()
	LESSEQUAL           = auto()
	LPAR                = auto()
	LSQB                = auto()
	MINEQUAL            = auto()
	MINUS               = auto()
	NOTEQUAL            = auto()
	NT_OFFSET           = auto()
	N_TOKENS            = auto()
	PERCENT             = auto()
	PERCENTEQUAL        = auto()
	PLUS                = auto()
	PLUSEQUAL           = auto()
	RARROW              = auto()
	RBRACE              = auto()
	RIGHTSHIFT          = auto()
	RIGHTSHIFTEQUAL     = auto()
	RPAR                = auto()
	RSQB                = auto()
	SEMI                = auto()
	SLASH               = auto()
	SLASHEQUAL          = auto()
	STAR                = auto()
	STAREQUAL           = auto()
	TILDE               = auto()
	TYPE_IGNORE         = auto()
	VBAR                = auto()
	VBAREQUAL           = auto()
	tok_name            = auto()
	ENCODING            = auto()
	NL                  = auto()
	COMMENT             = auto()
	EXCLAMATION         = auto()
	EXACT_TOKEN_TYPES   = auto()
	TSTRING_START       = auto()
	TSTRING_MIDDLE      = auto()
	TSTRING_END         = auto()

	WHITESPACE          = auto()
	LINECONT            = auto()

	PYX_WHITESPACE      = auto()
	PYX_IDENTIFIER      = auto()
	PYX_STRING          = auto()
	PYX_COMMENT         = auto()
	PYX_TEXT            = auto()
	PYX_INTERP_START    = auto()
	PYX_INTERP_END      = auto()
	PYX_OP_EQ           = auto()
	PYX_OP_SPREAD       = auto()
	PYX_OP_DOT          = auto()
	PYX_OP_COLON        = auto()
	PYX_OP              = auto()
	PYX_TAG_END         = auto()
	PYX_TAG_START       = auto()
	PYX_CLOSE_TAG_START = auto()
	PYX_TAG_SELFCLOSE   = auto()

expr_start_keywords = {
	"return", "yield", "raise",
	"assert", 
	"if", "elif", "while", "else", "for",
	"and", "or", "not",
	"lambda", "await",
	"in", "is", 
	"del", "with", "as", "from", "import", "global", "nonlocal", "print"
}

closing_brackets = {
	")", "]", "}"
}

opening_brackets = {
	"(", "[", "{"
}

bracket_pairs = {
	"(": ")",
	"[": "]",
	"{": "}"
}

exact_token_types = {
	string: PyxTokenType[name] for string, name in [(string, token.tok_name[value]) for string, value in  token.EXACT_TOKEN_TYPES.items()]
}

pyx_exact_token_types = {
	"=": PyxTokenType.PYX_OP_EQ,
	"...": PyxTokenType.PYX_OP_SPREAD,
	"{": PyxTokenType.PYX_INTERP_START,
	"}": PyxTokenType.PYX_INTERP_END,
	".": PyxTokenType.PYX_OP_DOT,
	":": PyxTokenType.PYX_OP_COLON,
}

def is_pyx_token(token: PyxTokenInfo) -> bool:
	return token.type.name.startswith("PYX")

def get_exact_token_type(tok: PyxTokenInfo) -> Optional[PyxTokenType]:
	if is_pyx_token(tok):
		if tok.string in pyx_exact_token_types.keys():
			return pyx_exact_token_types[tok.string]
	elif tok.string in exact_token_types.keys():
		return exact_token_types[tok.string]
	return None

@dataclass
class PyxTokenInfo:
	type: PyxTokenType
	_line: StrRef
	_text: StrRef

	start: Pos
	end: Pos

	@property
	def line(self) -> str:
		return self._line.text

	@property
	def string(self) -> str:
		return self._text.text

	@property
	def exact_type(self) -> Optional[PyxTokenType]:
		return get_exact_token_type(self)

	@classmethod
	def from_match(cls, type: PyxTokenType, match: Match) -> Self:
		return cls(type, match.lines, match.text,
	           	   match.pos_start, match.pos_end)

	def __eq__(self, other: object) -> bool:
		if isinstance(other, IntEnum):
			return other == self.type
		return NotImplemented

	def __str__(self) -> str:
		return f"({self.type.name}, {self.string!r}, {self.start}, {self.end})"  

class Tokenizer:
	def __init__(self, ctx: PyxTokenGenerator) -> None:
		self.ctx = ctx
		self.scanner = ctx.scanner

	def scan_one(self) -> PyxTokenInfo:
		raise NotImplementedError()

class PyTokenizerMode(IntEnum):
	Normal = auto()
	FString = auto()
	Pyx = auto()

class PyTokenizerImpl(Tokenizer):
	def __init__(self, ctx: PyxTokenGenerator, indent: str="", mode: PyTokenizerMode=PyTokenizerMode.Normal) -> None:
		super().__init__(ctx)
		self.indent_stack = [""]
		self.bracket_stack = []
		self.mode = mode

	@property
	def is_start_of_line(self) -> bool:
		return not self.ctx.prev or self.ctx.prev in (PyxTokenType.NEWLINE, PyxTokenType.NL)

	@property
	def is_top_level(self) -> bool:
		return not self.bracket_stack and self.mode == PyTokenizerMode.Normal

	@property
	def maybe_start_of_expr(self) -> bool:
		prev = self.ctx.prev if self.ctx.prev != PyxTokenType.WHITESPACE else self.ctx.prevprev
		assert prev != PyxTokenType.WHITESPACE
		if prev is None or prev in (PyxTokenType.NEWLINE, PyxTokenType.INDENT, PyxTokenType.DEDENT):
			return True
		if prev == PyxTokenType.OP:
			return prev.string not in closing_brackets
		if prev == PyxTokenType.NAME:
			return prev.string in expr_start_keywords
		return False

	@property
	def cur_indent(self) -> str:
		return self.indent_stack[-1]

	def handle_indent(self, match: Match) -> Optional[PyxTokenInfo]:
		indent = match.text.text
		if indent != self.cur_indent:
			if indent.startswith(self.cur_indent):
				self.indent_stack.append(indent)
				return PyxTokenInfo.from_match(PyxTokenType.INDENT, match)
			elif indent in self.indent_stack:
				index = self.indent_stack.index(indent)
				dedent_count = len(self.indent_stack) - index - 1
				for _ in range(dedent_count):
					self.indent_stack.pop()
					self.ctx.pending.appendleft(PyxTokenInfo.from_match(PyxTokenType.DEDENT, match))
				return self.ctx.pending.pop()
			else:
				return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, match)

	def handle_operator(self, match: Match) -> PyxTokenInfo:
		operator = match.text.text

		if operator in opening_brackets:
			self.bracket_stack.append(operator)
		elif operator in closing_brackets:
			if self.mode in (PyTokenizerMode.FString, PyTokenizerMode.Pyx) and operator == "}" and not self.bracket_stack:
				self.ctx.pop()
				if self.mode == PyTokenizerMode.Pyx:
					return PyxTokenInfo.from_match(PyxTokenType.PYX_INTERP_END, match)
			elif self.bracket_stack and operator == bracket_pairs[self.bracket_stack[-1]]:
				self.bracket_stack.pop()
			else:
				return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, match)

		return PyxTokenInfo.from_match(PyxTokenType.OP, match)

	# def scan_string(self, match: Match) -> PyxTokenInfo:
	# 	text = match.text

	# 	quot = match.groups["quot"].text
	# 	mid = string_mid[quot]
	# 	end = string_end[quot]

	# 	while True:
	# 		if m := self.scanner.consume(mid):
	# 			text += m.text
	# 		elif m := self.scanner.consume(end):
	# 			text += m.text
	# 			return PyxTokenInfo.from_text(PyxTokenType.STRING, text, m.offset_start, m.pos_start)
	# 		elif self.scanner.eof or self.scanner.has_data:
	# 			rest = self.scanner.consume(len(self.scanner))
	# 			assert rest is not None
	# 			text += rest.string
	# 			return PyxTokenInfo.from_text(PyxTokenType.ERRORTOKEN, text, match.offset_start, match.pos_start)

	def _scan_one(self) -> Optional[PyxTokenInfo]:
		if m := self.scanner.consume(Whitespace):
			if self.is_start_of_line and self.is_top_level and not self.scanner.match(Newline, Comment, Eof):
				if indent := self.handle_indent(m):
					return indent
			elif m.len != 0:
				return PyxTokenInfo.from_match(PyxTokenType.WHITESPACE, m)
		if m := self.scanner.consume(LineCont):
			return PyxTokenInfo.from_match(PyxTokenType.LINECONT, m)
		elif m := self.scanner.consume(Newline):
			if (self.is_top_level and 
			   		(self.ctx.prev not in (PyxTokenType.COMMENT, PyxTokenType.NEWLINE, PyxTokenType.NL) or 
			   		(self.ctx.prev == PyxTokenType.COMMENT and self.ctx.prevprev and self.ctx.prevprev not in (PyxTokenType.COMMENT, PyxTokenType.NEWLINE, PyxTokenType.NL)))
			 	):
				return PyxTokenInfo.from_match(PyxTokenType.NEWLINE, m)
			else:
				return PyxTokenInfo.from_match(PyxTokenType.NL, m)
		elif m := self.scanner.consume(Comment):
			return PyxTokenInfo.from_match(PyxTokenType.COMMENT, m)
		elif m := self.scanner.consume(FStringStart):
			self.ctx.push(FStringTokenizer(self.ctx, m.groups["quot"].text))
			return PyxTokenInfo.from_match(PyxTokenType.FSTRING_START, m)
		elif self.scanner.match(PlainStringStart3) and (m := self.scanner.consume(PlainString3, multiline=True)):
			return PyxTokenInfo.from_match(PyxTokenType.STRING, m)
		# elif m := self.scanner.consume(PlainStringStart3):
		# 	return self.scan_string(m)
		elif m := self.scanner.consume(PlainString1):
			return PyxTokenInfo.from_match(PyxTokenType.STRING, m)
		# elif m := self.scanner.consume(PlainStringStart1):
		# 	return self.scan_string(m)
		elif m := self.scanner.consume(Number):
			return PyxTokenInfo.from_match(PyxTokenType.NUMBER, m)
		elif m := self.scanner.consume(Name):
			return PyxTokenInfo.from_match(PyxTokenType.NAME, m)
		elif self.maybe_start_of_expr and (m := self.scanner.consume(PyxTagStart)):
			self.ctx.push(PyxTokenizerImpl(self.ctx))
			return PyxTokenInfo.from_match(PyxTokenType.PYX_TAG_START, m)
		elif m := self.scanner.consume(Operator):
			return self.handle_operator(m)
		elif m := self.scanner.consume(Eof):
			# NOTE: Ensure to always emit a newline token and pending dedents before the endmarker.
			if self.ctx.prev and self.ctx.prev.type not in (PyxTokenType.NEWLINE, PyxTokenType.NL):
				self.ctx.pending.appendleft(PyxTokenInfo.from_match(PyxTokenType.NEWLINE, m))
			if indent := self.handle_indent(m):
				self.ctx.pending.append(indent)
			self.ctx.pending.appendleft(PyxTokenInfo.from_match(PyxTokenType.ENDMARKER, m))
			return self.ctx.pending.pop()
		elif m := self.scanner.advance(1):
			return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, m)
		else:
			raise TokenizeError()

	@override
	def scan_one(self) -> PyxTokenInfo:
		t = None
		while t is None:
			t = self._scan_one()
		return t

class FStringTokenizer(Tokenizer):
	def __init__(self, ctx: PyxTokenGenerator, quot: str) -> None:
		super().__init__(ctx)
		self.end = fstring_end[quot]
		self.mid = fstring_mid[quot]

	@override
	def scan_one(self) -> PyxTokenInfo:
		while True:
			if m := self.scanner.consume(self.end):
				self.ctx.pop()
				return PyxTokenInfo.from_match(PyxTokenType.FSTRING_END, m)
			elif m := self.scanner.consume(LBrace):
				self.ctx.push(PyTokenizerImpl(self.ctx, mode=PyTokenizerMode.FString))
				return PyxTokenInfo.from_match(PyxTokenType.OP, m)
			elif m := self.scanner.consume(self.mid, many=True):
				return PyxTokenInfo.from_match(PyxTokenType.FSTRING_MIDDLE, m)
			# elif m := self.scanner.consume(self.mid):
			# 	text = m.text
			# 	while n := self.scanner.consume(self.mid):
			# 		text += n.string
			# 	return PyxTokenInfo.from_text(PyxTokenType.FSTRING_MIDDLE, text, m.offset_start, m.pos_start)
			elif m := self.scanner.advance(1):
				return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, m)
			else:
				raise TokenizeError()

class PyxTokenizerMode(IntEnum):
	Tag = auto()
	CloseTagStart = auto()
	Element = auto()

class PyxTokenizerImpl(Tokenizer):
	def __init__(self, ctx: PyxTokenGenerator) -> None:
		super().__init__(ctx)
		self.mode_stack = [PyxTokenizerMode.Tag]

	@property
	def mode(self) -> Optional[PyxTokenizerMode]:
		return self.mode_stack[-1] if self.mode_stack else None

	def handle_operator(self, match: Match) -> PyxTokenInfo:
		if pyx_exact_token_types[match.text.text] == PyxTokenType.PYX_INTERP_START:
			self.ctx.push(PyTokenizerImpl(self.ctx, mode=PyTokenizerMode.Pyx))
			self.ctx.pending.appendleft(PyxTokenInfo.from_match(PyxTokenType.PYX_INTERP_START, match))
			if m := self.scanner.consume(PyxWhitespace):
				self.ctx.pending.appendleft(PyxTokenInfo.from_match(PyxTokenType.PYX_WHITESPACE, m))
			if m := self.scanner.consume(PyxSpreadOperator):
				self.ctx.pending.appendleft(PyxTokenInfo.from_match(PyxTokenType.PYX_OP_SPREAD, m))
			return self.ctx.pending.pop()
		elif pyx_exact_token_types[match.text.text] == PyxTokenType.PYX_INTERP_END:
			return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, match)
		return PyxTokenInfo.from_match(PyxTokenType.PYX_OP, match)

	# def scan_comment(self, match: Match) -> PyxTokenInfo:
	# 	text = match.text
	# 	while True:
	# 		if m := self.scanner.consume(PyxCommentEnd):
	# 			text += m.text
	# 			return PyxTokenInfo.from_text(PyxTokenType.PYX_COMMENT, text, match.offset_start, match.pos_start)
	# 		elif m := self.scanner.consume(PyxCommentMid):
	# 			text += m.text
	# 		elif self.scanner.has_data or self.scanner.eof:
	# 			rest = self.scanner.consume(len(self.scanner))
	# 			assert rest
	# 			text += rest.string
	# 			return PyxTokenInfo.from_text(PyxTokenType.ERRORTOKEN, text, match.offset_start, match.pos_start)

	def handle_element(self) -> PyxTokenInfo:
		if m := self.scanner.consume(PyxCloseStart):
			self.mode_stack.pop()
			self.mode_stack.append(PyxTokenizerMode.CloseTagStart)
			return PyxTokenInfo.from_match(PyxTokenType.PYX_CLOSE_TAG_START, m)
		elif self.scanner.match(PyxCommentStart) and (m := self.scanner.consume(PyxComment, multiline=True)):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_COMMENT, m)
		# elif m := self.scanner.consume(PyxCommentStart):
		# 	return self.scan_comment(m)
		elif m := self.scanner.consume(PyxTagStart):
			self.mode_stack.append(PyxTokenizerMode.Tag)
			return PyxTokenInfo.from_match(PyxTokenType.PYX_TAG_START, m)
		elif m := self.scanner.consume(PyxText, many=True):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_TEXT, m)
		# elif m := self.scanner.consume(PyxText):
		# 	text = m.text
		# 	while n := self.scanner.consume(PyxText):
		# 		text += n.string
		# 	return PyxTokenInfo.from_text(PyxTokenType.PYX_TEXT, text, m.offset_start, m.pos_start)
		elif m := self.scanner.consume(PyxOperator):
			return self.handle_operator(m)
		elif m := self.scanner.advance(1):
			return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, m)
		else:
			raise TokenizeError()

	def handle_close_tag(self) -> PyxTokenInfo:
		if m := self.scanner.consume(PyxWhitespace):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_WHITESPACE, m)
		elif m := self.scanner.consume(PyxIdentifier):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_IDENTIFIER, m)
		elif m := self.scanner.consume(PyxTagEnd):
			self.mode_stack.pop()
			if not self.mode:
				self.ctx.pop()
			return PyxTokenInfo.from_match(PyxTokenType.PYX_TAG_END, m)
		elif m := self.scanner.advance(1):
			return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, m)
		else:
			raise TokenizeError()

	def handle_tag(self) -> PyxTokenInfo:
		if m := self.scanner.consume(PyxWhitespace):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_WHITESPACE, m)
		elif m := self.scanner.consume(PyxIdentifier):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_IDENTIFIER, m)
		elif m := self.scanner.consume(PyxOperator):
			return self.handle_operator(m)
		elif m := self.scanner.consume(PyxString):
			return PyxTokenInfo.from_match(PyxTokenType.PYX_STRING, m)
		elif m := self.scanner.consume(PyxTagSelfClose):
			self.mode_stack.pop()
			if not self.mode:
				self.ctx.pop()
			return PyxTokenInfo.from_match(PyxTokenType.PYX_TAG_SELFCLOSE, m)
		elif m := self.scanner.consume(PyxTagEnd):
			self.mode_stack.pop()
			self.mode_stack.append(PyxTokenizerMode.Element)
			return PyxTokenInfo.from_match(PyxTokenType.PYX_TAG_END, m)
		elif m := self.scanner.advance(1):
			return PyxTokenInfo.from_match(PyxTokenType.ERRORTOKEN, m)
		else:
			raise TokenizeError()

	@override
	def scan_one(self) -> PyxTokenInfo:
		if self.mode == PyxTokenizerMode.Element:
			return self.handle_element()
		elif self.mode == PyxTokenizerMode.CloseTagStart:
			return self.handle_close_tag()
		elif self.mode == PyxTokenizerMode.Tag:
			return self.handle_tag()
		else:
			raise TokenizeError()


class PyxTokenGenerator(TokenGenerator[PyxTokenType]):
	def __init__(self, file: str | Scanner | TextIO) -> None:
		self.scanner: Scanner
		if isinstance(file, str):
			self.scanner = Scanner(StringIO(file))
		elif isinstance(file, Scanner):
			self.scanner = file
		else:
			self.scanner = Scanner(file)
		self.pending: deque[PyxTokenInfo] = deque()
		self.prev_stack: deque[PyxTokenInfo] = deque(maxlen=2)
		self.tokenizer_stack: List[Tokenizer] = [PyTokenizerImpl(self)]

	@property
	def prev(self) -> Optional[PyxTokenInfo]:
		return self.prev_stack[0] if self.prev_stack else None

	@property
	def prevprev(self) -> Optional[PyxTokenInfo]:
		return self.prev_stack[1] if len(self.prev_stack) >= 2 else None

	@property
	def tokenizer(self) -> Tokenizer:
		return self.tokenizer_stack[-1]

	def pop(self) -> None:
		self.tokenizer_stack.pop()

	def push(self, tokenizer: Tokenizer) -> None:
		self.tokenizer_stack.append(tokenizer)

	def next(self) -> PyxTokenInfo:
		if self.pending:
			t = self.pending.pop()
		else:
			t = self.tokenizer.scan_one()
		self.prev_stack.appendleft(t)
		return t

	@override
	def __iter__(self) -> Self:
		return self

	@override
	def __next__(self) -> PyxTokenInfo:
		if self.prev == PyxTokenType.ENDMARKER:
			raise StopIteration
		return self.next()

def tokenize(source: str | TextIO, /) -> Iterator[TokenInfo[PyxTokenType]]:
	ctx = PyxTokenGenerator(source)
	return ctx

def untokenize(tokens: Iterable[PyxTokenInfo]) -> str:
	res = ""
	for t in tokens:
		if t.string:
			res += t.string
	return res

class PyxTokenizer(BaseTokenizer[PyxTokenType]):
	def __init__(self,
	             tokengen: PyxTokenGenerator,
	             *,
	             path: str="",
	             verbose: bool=False):
		super().__init__(tokengen,
		                 PyxTokenType.ENDMARKER,
		                 skip_tokens=(PyxTokenType.COMMENT, PyxTokenType.WHITESPACE, PyxTokenType.LINECONT, PyxTokenType.NL),
						 collapse_tokens=(PyxTokenType.NEWLINE,),
						 non_syntactic_tokens=(PyxTokenType.NEWLINE, PyxTokenType.INDENT, PyxTokenType.DEDENT, PyxTokenType.LINECONT),
						 path=path,
						 verbose=verbose)

	@override
	def should_skip(self, tok: TokenInfo[PyxTokenType]) -> bool:
		if super().should_skip(tok):
			return True
		elif (tok.type == PyxTokenType.ERRORTOKEN and tok.string == " "):
			return True
		else:
			return False

def main():
	p = argparse.ArgumentParser()

	def file_path(string: str | Path):
		path = Path(string)
		if path.is_file():
			return path
		else:
			raise FileNotFoundError(string)

	p.add_argument("filename", type=file_path, nargs="?")
	args = p.parse_args()

	if args.filename is not None:
		interactive = False
		f = open(args.filename, "r")
		scanner = Scanner(f)
	else:
		interactive = True
		scanner = Scanner(sys.stdin)

	ctx = PyxTokenGenerator(scanner)

	if interactive:
		tokens: deque[TokenInfo[PyxTokenType]] = deque()
		flush_tokens = (
			PyxTokenType.NEWLINE, PyxTokenType.ENDMARKER, 
		    PyxTokenType.PYX_TAG_SELFCLOSE, PyxTokenType.PYX_TAG_END, PyxTokenType.PYX_INTERP_START, PyxTokenType.PYX_INTERP_END,
		    PyxTokenType.PYX_TEXT,
		    PyxTokenType.ERRORTOKEN
		)

		for t in ctx:
			print(t)
			continue
			tokens.appendleft(t)

			if t in flush_tokens:
				while tokens:
					print(tokens.pop())
	else:
		for t in ctx:
			print(t)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
