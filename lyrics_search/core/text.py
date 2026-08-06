"""Text normalization helpers. Pure functions, no I/O, no third-party deps.

Used by:
- contracts.RawSong.song_id generation (`normalize_for_id` + `make_song_id`)
- core.dedupe (`normalize_for_dedupe`)
- core.sections (whitespace collapsing)
- embedders.fasttext_avg and retrievers.lexical (`tokenize_words`) -- SPEC-03
  §3 needs FastTextAvgEmbedder's TF-IDF weighting and LexicalRetriever's
  BM25 to use identical word tokenization, so it lives here once rather
  than as two independently-drifting copies.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokenization: runs of ASCII letters/apostrophes."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace (incl. newlines) to a single space, strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_id(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Used to build song_id.

    Deliberately does NOT strip punctuation -- "Rock 'n' Roll" and
    "Rock n Roll" should be allowed to hash differently; only the
    *presentation* differences (case, accents, spacing) are normalized.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return collapse_whitespace(text.lower())


def make_song_id(artist: str, title: str) -> str:
    """SPEC-02 §2: sha1(f"{norm(artist)}|{norm(title)}")[:16]."""
    key = f"{normalize_for_id(artist)}|{normalize_for_id(title)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def normalize_for_dedupe(text: str) -> str:
    """Aggressive normalization for duplicate-block detection: case,
    punctuation and whitespace differences should not prevent a match.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return collapse_whitespace(text)


def hash_block(text: str) -> str:
    return hashlib.sha1(normalize_for_dedupe(text).encode("utf-8")).hexdigest()
