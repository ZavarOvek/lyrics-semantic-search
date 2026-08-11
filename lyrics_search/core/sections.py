"""SPEC-02 §4.1-4.3: section detection and the four-level chunking cascade.

Pure functions, no I/O. `chunk_song` is the entry point used by the
preprocess runner; everything else here is a building block for it or
for tests.

Cascade semantics (SPEC-02 §4.3): "each subsequent level fires only
where the previous one gave no boundary" is applied *positionally*
within a song, not as an all-or-nothing choice per song. A song can
legitimately mix real `[Verse 1]` tags with a bracket-less `INTRO HOOK`
line elsewhere -- both fire, at their own position, each keeping its
own split_by. Blank-line splitting then further refines any segment
(from any level) that still has internal paragraph breaks. Level 4
(length) is the unconditional final pass over everything.

Token counts are approximated by whitespace-word counts. This is a
deliberate, documented heuristic (see notes/) rather than a real
tokenizer call: core/ has zero dependency on torch/transformers per
SPEC-02 §1, and word count is a cheap, good-enough proxy to keep the
hard ceiling far below any embedder's real tokenizer max_length (the
embed runner separately double-checks this against each model's *real*
tokenizer -- see SPEC-02-PATCH item 5/notes/reports/spec02-patch-report.md).

SPEC-02-PATCH item 6: `split_by` records the *structural* origin of a
segment's boundary ("bracket_tag" | "plain_label" | "blank_line" |
"none" -- no boundary of any kind was found) and is never overwritten.
`force_split` is a separate bool: whether level-4 word-count packing
*additionally* had to cut this piece down because it was still too long
after whatever structural split produced it. The two are independent:
a bracket_tag-delimited verse that's individually too long is
`split_by="bracket_tag", force_split=True`; a whole song with no
markup at all that happens to be short is `split_by="none",
force_split=False`. Previously these were conflated into a single
`split_by` value ("length" meant both), which made it impossible to
tell, from the field alone, whether length-packing had actually run.

EVAL-PREP §3: the cascade above is one *chunking strategy*, and there are
now two. `whole_song` emits exactly one segment per song, which is only
possible because bge-m3's real max_seq_length is 8192 -- no other embedder
here can hold a whole song in one vector. Having both selectable from
config makes chunked and unchunked retrieval directly comparable on the
same model, which answers a question the project has not yet asked: does
chunking earn its keep? `chunk_song_with()` is the dispatch point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lyrics_search.core.text import collapse_whitespace

CANONICAL_SECTIONS = [
    "Verse",
    "Chorus",
    "Bridge",
    "Hook",
    "Outro",
    "Pre-Chorus",
    "Intro",
    "Post-Chorus",
    "Refrain",
    "Interlude",
    "Instrumental",
]
_SINGLE_WORD_CANON = {
    "verse": "Verse",
    "chorus": "Chorus",
    "bridge": "Bridge",
    "hook": "Hook",
    "outro": "Outro",
    "intro": "Intro",
    "refrain": "Refrain",
    "interlude": "Interlude",
    "instrumental": "Instrumental",
}
_COMPOUND_CANON = {
    ("pre", "chorus"): "Pre-Chorus",
    ("post", "chorus"): "Post-Chorus",
}

BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_LEADING_WORD_RE = re.compile(r"^\s*([A-Za-z]+)")
_ALPHA_RE = re.compile(r"[A-Za-z]+")

# Target/ceiling in whitespace-word count (see module docstring).
TARGET_MIN_WORDS = 40
TARGET_MAX_WORDS = 80
HARD_CEILING_WORDS = 120


def _match_canonical(word: str) -> str | None:
    return _SINGLE_WORD_CANON.get(word.lower())


def classify_bracket(content: str) -> tuple[str, str | None]:
    """Classify the inner text of a `[...]` tag.

    Returns (kind, section) where kind is one of:
    - "canonical": recognized section type, `section` is the canonical name
    - "credits":   looks like a performer/producer credit, no section type
    - "noise":     placeholder/garbled content, not a boundary at all
    """
    stripped = content.strip()
    if not stripped or not _ALPHA_RE.search(stripped):
        return "noise", None

    m = _LEADING_WORD_RE.match(stripped)
    if m:
        canon = _match_canonical(m.group(1))
        if canon:
            return "canonical", canon
        rest_tokens = _ALPHA_RE.findall(stripped)
        if len(rest_tokens) >= 2:
            pair = (rest_tokens[0].lower(), rest_tokens[1].lower())
            if pair in _COMPOUND_CANON:
                return "canonical", _COMPOUND_CANON[pair]

    return "credits", None


def classify_plain_label_line(line: str) -> str | None:
    """SPEC-02 §4.2: bracket-less section labels on their own line.

    Conservative on purpose: only fires on short lines (<=2 alpha tokens,
    no brackets -- those are handled by classify_bracket) so ordinary
    lyric lines that merely mention "chorus" in a sentence are left alone.
    """
    raw = line.strip()
    if not raw or "[" in raw or "]" in raw:
        return None
    core = raw.rstrip(":").strip()
    tokens = _ALPHA_RE.findall(core)
    if not tokens or len(tokens) > 2:
        return None
    lower_tokens = [t.lower() for t in tokens]

    if len(lower_tokens) == 2:
        pair = (lower_tokens[0], lower_tokens[1])
        if pair in _COMPOUND_CANON:
            return _COMPOUND_CANON[pair]

    for t in lower_tokens:
        canon = _match_canonical(t)
        if canon:
            return canon
    return None


@dataclass(frozen=True)
class Segment:
    text: str
    section: str | None
    split_by: str  # "bracket_tag" | "plain_label" | "blank_line" | "none"
    force_split: bool = False  # level-4 length packing applied on top


@dataclass(frozen=True)
class _Boundary:
    start: int  # char offset where the tag/label itself starts
    end: int  # char offset right after it (body starts here)
    section: str | None
    split_by: str


def _strip_noise_brackets(text: str) -> str:
    def _sub(m: re.Match) -> str:
        kind, _ = classify_bracket(m.group(1))
        return "" if kind == "noise" else m.group(0)

    return BRACKET_RE.sub(_sub, text)


def _find_bracket_boundaries(text: str) -> list[_Boundary]:
    boundaries = []
    for m in BRACKET_RE.finditer(text):
        kind, section = classify_bracket(m.group(1))
        if kind == "noise":
            continue
        boundaries.append(_Boundary(m.start(), m.end(), section, "bracket_tag"))
    return boundaries


def _find_plain_label_boundaries(text: str) -> list[_Boundary]:
    boundaries = []
    pos = 0
    for line in text.split("\n"):
        section = classify_plain_label_line(line)
        if section:
            boundaries.append(_Boundary(pos, pos + len(line), section, "plain_label"))
        pos += len(line) + 1  # +1 for the '\n' consumed by split
    return boundaries


def _outer_segments(text: str) -> list[Segment]:
    """Levels 1+2 combined positionally (see module docstring)."""
    text = _strip_noise_brackets(text)
    boundaries = sorted(
        _find_bracket_boundaries(text) + _find_plain_label_boundaries(text),
        key=lambda b: b.start,
    )

    if not boundaries:
        stripped = text.strip()
        return [Segment(stripped, None, "none")] if stripped else []

    segments: list[Segment] = []
    lead = text[: boundaries[0].start].strip()
    if lead:
        segments.append(Segment(lead, None, "none"))

    for i, b in enumerate(boundaries):
        next_start = boundaries[i + 1].start if i + 1 < len(boundaries) else len(text)
        body = text[b.end : next_start].strip()
        if body:
            segments.append(Segment(body, b.section, b.split_by))
    return segments


def _refine_with_blank_lines(seg: Segment) -> list[Segment]:
    """Level 3, applied within any level-1/2 segment (or the whole song)
    that still has internal blank-line paragraph breaks.

    SPEC-02-PATCH item 6: a single-part result keeps whatever split_by it
    already had (including "none" for a whole song/segment with no
    structural boundary at all) -- it is *not* relabeled "length" here.
    Whether length-packing (level 4) subsequently has to act on it is a
    separate, independent fact recorded by `force_split` in chunk_song.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", seg.text) if p.strip()]
    if len(parts) <= 1:
        return [Segment(seg.text, seg.section, seg.split_by)]
    return [Segment(p, seg.section, "blank_line") for p in parts]


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_line(line: str, hard_ceiling: int) -> list[str]:
    """Fallback for a single line/sentence that alone exceeds the ceiling.
    Splits on sentence punctuation first, then on whitespace as a last
    resort. Never breaks a word.
    """
    sentences = re.split(r"(?<=[.!?])\s+", line)
    if len(sentences) == 1:
        words = line.split()
        pieces, cur = [], []
        for w in words:
            if cur and len(cur) + 1 > hard_ceiling:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(w)
        if cur:
            pieces.append(" ".join(cur))
        return pieces
    return [s for s in sentences if s.strip()]


def _pack_by_length(text: str, target_max: int, hard_ceiling: int) -> list[str]:
    """Greedily pack lines into pieces around target_max words, never
    exceeding hard_ceiling, never splitting a word.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    pieces: list[str] = []
    cur_lines: list[str] = []
    cur_words = 0

    def flush() -> None:
        nonlocal cur_lines, cur_words
        if cur_lines:
            pieces.append("\n".join(cur_lines))
        cur_lines, cur_words = [], 0

    for line in lines:
        w = _word_count(line)
        if w > hard_ceiling:
            flush()
            pieces.extend(_split_long_line(line, hard_ceiling))
            continue
        if cur_words + w > hard_ceiling:
            flush()
        cur_lines.append(line)
        cur_words += w
        if cur_words >= target_max:
            flush()
    flush()
    return pieces


def chunk_song(
    text: str,
    *,
    target_max: int = TARGET_MAX_WORDS,
    hard_ceiling: int = HARD_CEILING_WORDS,
    warn: "callable | None" = None,
) -> list[Segment]:
    """SPEC-02 §4.3 cascade. `warn(original_word_count)` is called once per
    segment that hits level-4 length packing -- i.e. once per
    `force_split` event, *not* once per output piece (a single oversized
    segment can produce several force_split=True pieces from one warn()
    call). This is a plain instrumentation hook, not a "something is
    wrong" signal (SPEC-02-PATCH item 5): forced length-splitting is the
    cascade working as designed, not silent truncation. Silent
    truncation -- a chunk still exceeding a model's *real* tokenizer
    max_length after this cascade -- is checked for separately, for real,
    downstream in runners/embed.py, where an actual tokenizer is
    available (core/ has none, per SPEC-02 §1).
    """
    outer = _outer_segments(text)

    refined: list[Segment] = []
    for seg in outer:
        refined.extend(_refine_with_blank_lines(seg))

    final: list[Segment] = []
    for seg in refined:
        wc = _word_count(seg.text)
        if wc <= hard_ceiling:
            final.append(seg)
            continue
        if warn:
            warn(wc)
        for piece in _pack_by_length(seg.text, target_max, hard_ceiling):
            piece = collapse_whitespace(piece)
            if piece:
                final.append(Segment(piece, seg.section, seg.split_by, force_split=True))
    return final


def whole_song_segments(text: str) -> list[Segment]:
    """EVAL-PREP §3: one segment per song, or none if the song is empty.

    The body is assembled from `_outer_segments()` rather than from `text`
    directly, so structural markup (`[Verse 1]`, bracket-less labels,
    noise brackets) is stripped exactly as it is in the sections
    strategy. Otherwise the two arms would differ in *two* ways --
    chunking and tag removal -- and any measured difference could not be
    attributed to chunking alone, which is the whole point of having the
    arm.

    `split_by` is "none" and `section` is None because a whole-song
    segment has, by construction, no boundary and no single section type.
    This makes eval's `split_by` x `force_split` slice degenerate for
    this arm -- correctly, and visibly, rather than by carrying a label
    inherited from whichever tag happened to come first.
    """
    body = "\n".join(seg.text for seg in _outer_segments(text))
    body = body.strip()
    if not body:
        return []
    return [Segment(body, None, "none", force_split=False)]


CHUNKING_SECTIONS = "sections"
CHUNKING_WHOLE_SONG = "whole_song"
CHUNKING_STRATEGIES = (CHUNKING_SECTIONS, CHUNKING_WHOLE_SONG)


def chunk_song_with(
    text: str,
    strategy: str = CHUNKING_SECTIONS,
    *,
    warn: "callable | None" = None,
) -> list[Segment]:
    """Dispatch to a chunking strategy by name (EVAL-PREP §3).

    `warn` is only meaningful for the sections cascade -- whole_song
    never force-splits, so it never fires. It is accepted for both so the
    preprocess runner does not have to branch on the strategy.
    """
    if strategy == CHUNKING_SECTIONS:
        return chunk_song(text, warn=warn)
    if strategy == CHUNKING_WHOLE_SONG:
        return whole_song_segments(text)
    raise ValueError(
        f"unknown chunking strategy {strategy!r}; expected one of {', '.join(CHUNKING_STRATEGIES)}"
    )
