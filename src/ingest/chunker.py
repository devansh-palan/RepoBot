"""AST-based chunking.

One chunk per function, one per class header, and size-based chunks for
everything else. The guiding rule is that no line of a supported file is ever
dropped: whatever the AST walk does not claim is swept up by gap filling, and
anything too large to embed is split into overlapping windows.

All language-specific behaviour comes from config.LanguageSpec, so this module
never mentions Python by name.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from tree_sitter import Language, Node, Parser

from src import config
from src.config import LanguageSpec

from .models import MODULE_SYMBOL, Chunk

# A 0-based, inclusive row range plus what lives there.
_Span = tuple[int, int, str, str]  # (start_row, end_row, kind, symbol)


class MissingGrammarError(RuntimeError):
    """A registry entry names a grammar that is not installed or has no entry point."""


@lru_cache(maxsize=None)
def get_parser(grammar_module: str, entrypoint: str) -> Parser:
    """Load and cache a tree-sitter parser for a grammar wheel.

    Raises MissingGrammarError rather than ModuleNotFoundError/AttributeError so
    callers can skip one misconfigured language instead of losing the whole run.
    """
    try:
        module = importlib.import_module(grammar_module)
    except ModuleNotFoundError as exc:
        raise MissingGrammarError(
            f"grammar module {grammar_module!r} is not installed; "
            f"add it to requirements.txt"
        ) from exc

    factory = getattr(module, entrypoint, None)
    if factory is None:
        available = sorted(n for n in dir(module) if n.startswith("language"))
        raise MissingGrammarError(
            f"{grammar_module!r} has no {entrypoint!r}; available: {available}"
        )
    return Parser(Language(factory()))


def chunk_file(path: str | Path, repo_root: str | Path) -> list[Chunk]:
    """Chunk one file on disk. Returns [] for unsupported languages."""
    path = Path(path)
    spec = config.spec_for_path(path)
    if spec is None:
        return []

    data = path.read_bytes()
    rel = Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    return chunk_source(data, rel, spec)


def chunk_source(data: bytes, file_path: str, spec: LanguageSpec) -> list[Chunk]:
    """Chunk source bytes. Split out from chunk_file so tests need no disk."""
    # errors="replace" keeps the line structure intact on stray bytes, and
    # normalising CRLF does not change the newline count tree-sitter counts by.
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    lines = text.split("\n")

    tree = get_parser(spec.grammar_module, spec.grammar_entrypoint).parse(data)

    spans: list[_Span] = []
    class_extents: list[tuple[int, int, str]] = []
    _visit(tree.root_node, "", spec, spans, class_extents)
    gaps = _gap_spans(spans, lines, class_extents)
    spans = _absorb_structural_gaps(spans, gaps, lines)
    spans.sort(key=lambda s: (s[0], s[1]))

    chunks: list[Chunk] = []
    for start, end, kind, symbol in spans:
        chunks.extend(_emit(lines, start, end, kind, symbol, file_path, spec))
    return chunks


# --------------------------------------------------------------------------
# AST walk
# --------------------------------------------------------------------------


def _visit(
    node: Node,
    prefix: str,
    spec: LanguageSpec,
    out: list[_Span],
    class_extents: list[tuple[int, int, str]],
) -> None:
    """Record a span for every function and class reachable from `node`.

    Descends through unrecognised nodes (an `if` or `try` at module level can
    hold definitions) and into class bodies, but never into a function body —
    closures and helpers stay with the function that owns them.
    """
    for child in node.named_children:
        span_node, definition, chain = _unwrap(child, spec)
        if definition is None or span_node is None:
            _visit(child, prefix, spec, out, class_extents)
            continue

        name = _symbol_name(definition, chain, spec)
        qualified = prefix + _receiver_prefix(definition, spec) + name

        if definition.type in spec.type_nodes:
            # Interfaces, enums and plain structs own no methods, so there is
            # nothing to recurse into — emit the declaration whole.
            out.append(
                (span_node.start_point[0], span_node.end_point[0], "type", qualified)
            )
        elif definition.type in spec.class_nodes:
            nested: list[_Span] = []
            body = definition.child_by_field_name("body")
            if body is not None:
                _visit(body, qualified + ".", spec, nested, class_extents)

            # The class chunk is the signature, docstring, and attributes down
            # to the first method — not the whole body, which would duplicate
            # every method and blow past the size limit on large classes.
            first_member = min((s[0] for s in nested), default=None)
            header_end = (
                first_member - 1 if first_member is not None else span_node.end_point[0]
            )
            out.append((span_node.start_point[0], header_end, "class", qualified))
            out.extend(nested)
            class_extents.append(
                (span_node.start_point[0], span_node.end_point[0], qualified)
            )
        else:
            out.append(
                (span_node.start_point[0], span_node.end_point[0], "function", qualified)
            )


_MAX_UNWRAP_DEPTH = 8


def _unwrap(
    node: Node, spec: LanguageSpec
) -> tuple[Node | None, Node | None, tuple[Node, ...]]:
    """Resolve a node to (span_node, definition, wrappers).

    Wrappers are unwrapped in a chain, because a definition can sit several
    levels down: `export const f = () => {}` is
    export_statement > lexical_declaration > variable_declarator > arrow_function.
    The span stays the outermost node so `export` and decorators travel with the
    symbol, while kind comes from the definition and the name may come from any
    link in the chain.

    Returns (None, None, ()) when the node is not a symbol at all.
    """
    symbol_nodes = spec.symbol_nodes()

    chain: list[Node] = []
    current = node
    for _ in range(_MAX_UNWRAP_DEPTH):
        if current.type not in spec.wrapper_nodes:
            break
        chain.append(current)
        following = next(
            (
                child
                for child in current.named_children
                if child.type in symbol_nodes or child.type in spec.wrapper_nodes
            ),
            None,
        )
        if following is None:
            return None, None, ()
        current = following

    if current.type in symbol_nodes:
        return node, current, tuple(chain)
    return None, None, ()


def _symbol_name(
    definition: Node, chain: tuple[Node, ...], spec: LanguageSpec
) -> str:
    """The declared name, searching the definition then its wrappers inward-out.

    The fallback to wrappers is what names an anonymous arrow function after
    the variable it is assigned to.
    """
    for node in (definition, *reversed(chain)):
        for field in spec.name_fields:
            candidate = node.child_by_field_name(field)
            if candidate is None:
                continue
            resolved = _resolve_declarator(candidate)
            if resolved.text:
                return resolved.text.decode("utf-8", errors="replace")
    return "<anonymous>"


def _receiver_prefix(definition: Node, spec: LanguageSpec) -> str:
    """Qualify a method by its receiver type, for languages that declare
    methods at top level.

    Go's `func (a *Account) Interest()` is a sibling of the type it belongs to,
    not a child, so without this every method would chunk as a bare `Interest`
    and collide with the same method name on any other type.
    """
    if spec.receiver_field is None:
        return ""

    receiver = definition.child_by_field_name(spec.receiver_field)
    if receiver is None:
        return ""

    # The type sits under a pointer or generic wrapper; the first type
    # identifier in the receiver is the name in every form.
    named = _first_descendant(receiver, "type_identifier")
    if named is None or not named.text:
        return ""
    return named.text.decode("utf-8", errors="replace") + "."


def _first_descendant(node: Node, node_type: str) -> Node | None:
    """Depth-first search for the first named descendant of a given type."""
    for child in node.named_children:
        if child.type == node_type:
            return child
        found = _first_descendant(child, node_type)
        if found is not None:
            return found
    return None


def _resolve_declarator(node: Node) -> Node:
    """Follow C/C++ declarator nesting down to the identifier.

    A C function's name lives at the bottom of
    function_definition > function_declarator > identifier, with pointers and
    arrays adding further levels.
    """
    for _ in range(_MAX_UNWRAP_DEPTH):
        inner = node.child_by_field_name("declarator")
        if inner is None:
            return node
        node = inner
    return node


# --------------------------------------------------------------------------
# Gap filling — the guarantee that nothing is dropped
# --------------------------------------------------------------------------


def _gap_spans(
    spans: list[_Span],
    lines: list[str],
    class_extents: list[tuple[int, int, str]],
) -> list[_Span]:
    """Spans covering every line no symbol claimed: imports, constants, scripts.

    Entirely blank runs are skipped — they carry nothing to retrieve — but any
    gap with content becomes a chunk.
    """
    covered = bytearray(len(lines))
    for start, end, _, _ in spans:
        for row in range(start, min(end + 1, len(lines))):
            covered[row] = 1

    gaps: list[_Span] = []
    for run_start, run_end in _uncovered_runs(covered):
        # Cut at class edges first: a run can start inside a class body and
        # continue past its end, and each side wants a different symbol name.
        for seg_start, seg_end in _split_at_class_edges(
            run_start, run_end, class_extents
        ):
            trimmed = _trim_blank_edges(lines, seg_start, seg_end)
            if trimmed is None:
                continue
            start, end = trimmed
            kind, symbol = _enclosing_class(start, end, class_extents)
            gaps.append((start, end, kind, symbol))
    return gaps


def _split_at_class_edges(
    start: int, end: int, class_extents: list[tuple[int, int, str]]
) -> Iterator[tuple[int, int]]:
    """Cut an inclusive row range wherever a class begins or ends."""
    cuts = sorted(
        {
            row
            for cls_start, cls_end, _ in class_extents
            for row in (cls_start, cls_end + 1)
            if start < row <= end
        }
    )
    left = start
    for cut in cuts:
        yield left, cut - 1
        left = cut
    yield left, end


def _trim_blank_edges(
    lines: list[str], start: int, end: int
) -> tuple[int, int] | None:
    """Shrink a range to its content, or None when it is blank throughout.

    Without this, the empty string after a file's final newline becomes a chunk
    of its own, and trailing blank lines drag gaps past the class they belong to.
    """
    while start <= end and not lines[start].strip():
        start += 1
    while end >= start and not lines[end].strip():
        end -= 1
    return (start, end) if start <= end else None


# Lines made only of these carry no meaning on their own: they are the tail of
# the construct above them. `end` covers Ruby, where blocks close with a word.
_STRUCTURAL_CHARS = set("{}()[];,")
_STRUCTURAL_WORDS = {"end", "});", "})"}


def _absorb_structural_gaps(
    spans: list[_Span], gaps: list[_Span], lines: list[str]
) -> list[_Span]:
    """Fold closing-brace-only gaps into the chunk they close.

    Without this, every brace-language class ends up with a second chunk whose
    entire content is `}` — pure noise in the index. Folding rather than
    dropping keeps the guarantee that no line goes missing.
    """
    result = sorted(spans, key=lambda s: (s[0], s[1]))
    ends = {span[1]: index for index, span in enumerate(result)}

    leftover: list[_Span] = []
    for gap in sorted(gaps, key=lambda s: (s[0], s[1])):
        start, end, _, _ = gap
        index = ends.get(start - 1)
        if index is None or not _is_structural_only(lines, start, end):
            leftover.append(gap)
            continue

        previous = result[index]
        result[index] = (previous[0], end, previous[2], previous[3])
        del ends[start - 1]
        ends[end] = index

    return result + leftover


def _is_structural_only(lines: list[str], start: int, end: int) -> bool:
    """True when a range holds nothing but block terminators."""
    for row in range(start, end + 1):
        stripped = lines[row].strip()
        if not stripped:
            continue
        if stripped in _STRUCTURAL_WORDS:
            continue
        if set(stripped) <= _STRUCTURAL_CHARS:
            continue
        return False
    return True


def _uncovered_runs(covered: bytearray) -> Iterator[tuple[int, int]]:
    """Yield inclusive (start, end) runs of zero bytes."""
    row = 0
    while row < len(covered):
        if covered[row]:
            row += 1
            continue
        start = row
        while row < len(covered) and not covered[row]:
            row += 1
        yield start, row - 1


def _enclosing_class(
    start: int, end: int, class_extents: list[tuple[int, int, str]]
) -> tuple[str, str]:
    """Attribute a gap to the innermost class containing it, if any.

    Trailing class attributes declared after the last method land here; naming
    them after their class keeps the citation meaningful.
    """
    best: tuple[int, str] | None = None
    for cls_start, cls_end, name in class_extents:
        if cls_start <= start and end <= cls_end:
            size = cls_end - cls_start
            if best is None or size < best[0]:
                best = (size, name)
    if best is None:
        return "module", MODULE_SYMBOL
    return "class", best[1]


# --------------------------------------------------------------------------
# Emitting, with size-based splitting
# --------------------------------------------------------------------------


def _emit(
    lines: list[str],
    start: int,
    end: int,
    kind: str,
    symbol: str,
    file_path: str,
    spec: LanguageSpec,
) -> list[Chunk]:
    """Turn one span into one chunk, or several if it exceeds the size limits."""
    windows = list(_windows(lines, start, end))
    total = len(windows)
    return [
        Chunk(
            code="\n".join(lines[w_start : w_end + 1]),
            language=spec.name,
            file_path=file_path,
            symbol=symbol,
            kind=kind,
            start_line=w_start + 1,  # rows are 0-based, chunk lines are 1-based
            end_line=w_end + 1,
            part=index,
            part_count=total,
        )
        for index, (w_start, w_end) in enumerate(windows, start=1)
    ]


def _windows(lines: list[str], start: int, end: int) -> Iterator[tuple[int, int]]:
    """Split an inclusive row range into windows that fit the size limits.

    Greedy: take lines until either budget would be exceeded, then step back by
    CHUNK_OVERLAP_LINES so a construct straddling a boundary still appears
    whole in one window.
    """
    row = start
    while True:
        stop = row
        chars = 0
        while stop <= end:
            width = len(lines[stop]) + 1  # +1 for the newline we rejoin with
            too_many_lines = stop - row + 1 > config.MAX_CHUNK_LINES
            too_many_chars = chars + width > config.MAX_CHUNK_CHARS
            # `stop > row` guarantees progress: a single over-long line still
            # forms a window of its own rather than looping forever.
            if stop > row and (too_many_lines or too_many_chars):
                break
            chars += width
            stop += 1

        yield row, stop - 1

        if stop > end:
            return
        row = max(stop - config.CHUNK_OVERLAP_LINES, row + 1)
