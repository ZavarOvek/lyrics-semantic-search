"""SPEC-02-PATCH item 2: prove the song-level filter is actually wired up.

The full-corpus run showed `songs passed / rejected: 30000 / 0`. That is
consistent both with "the corpus is genuinely clean" and with "the filter
silently never fires" -- the two are indistinguishable from that number
alone. This test feeds `run_preprocess` a handful of deliberately bad
records through the real code path (not just the pure `_song_reject_reason`
helper in isolation) and asserts each lands in rejects.jsonl with
level="song" and the expected reason.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from lyrics_search.contracts import RawSong
from lyrics_search.core.rejects import RejectReason
from lyrics_search.core.text import make_song_id
from lyrics_search.runners.preprocess import _song_reject_reason, run_preprocess

GOOD_TEXT = (
    "[Verse 1]\nWalking down this empty street tonight\n"
    "Thinking about the way you used to hold me tight\n"
    "[Chorus]\nAnd I keep coming back to you\n"
    "Every single thing I do reminds me of the truth\n"
)


def _song(artist: str, title: str, text_raw: str) -> RawSong:
    return RawSong(
        song_id=make_song_id(artist, title),
        artist=artist,
        title=title,
        text_raw=text_raw,
        source="test",
    )


BAD_SONGS = {
    "empty": (_song("Test Artist", "Empty Page", ""), RejectReason.SONG_EMPTY_TEXT),
    "whitespace_only": (
        _song("Test Artist", "Whitespace Page", "   \n\n  \t  "),
        RejectReason.SONG_EMPTY_TEXT,
    ),
    "too_short": (
        _song("Test Artist", "Stub Page", "just three words"),
        RejectReason.SONG_TOO_SHORT,
    ),
    "instrumental_bracket": (
        _song("Test Artist", "Interlude", "[Instrumental]"),
        RejectReason.SONG_INSTRUMENTAL_ONLY,
    ),
    "instrumental_bare": (
        _song("Test Artist", "Interlude 2", "  instrumental  "),
        RejectReason.SONG_INSTRUMENTAL_ONLY,
    ),
}


# --- Unit level: the pure decision function in isolation -------------------


@pytest.mark.parametrize("name", BAD_SONGS.keys())
def test_song_reject_reason_pure(name):
    song, expected_reason = BAD_SONGS[name]
    assert _song_reject_reason(song) == expected_reason


def test_song_reject_reason_pure_good_song_passes():
    song = _song("Real Artist", "Real Song", GOOD_TEXT)
    assert _song_reject_reason(song) is None


# --- Integration level: through the real run_preprocess() I/O path ---------


def test_run_preprocess_rejects_bad_songs_at_song_level(tmp_path):
    good = _song("Real Artist", "Real Song", GOOD_TEXT)
    all_songs = [good] + [s for s, _ in BAD_SONGS.values()]

    raw_path = tmp_path / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for s in all_songs:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    out_dir = tmp_path / "out"
    run_preprocess(raw_path, out_dir)

    rejects = [json.loads(line) for line in open(out_dir / "rejects.jsonl", encoding="utf-8")]
    song_rejects = {r["song_id"]: r for r in rejects if r["level"] == "song"}

    # Every bad song must appear, exactly once, with the right reason.
    assert len(song_rejects) == len(BAD_SONGS)
    for name, (song, expected_reason) in BAD_SONGS.items():
        assert song.song_id in song_rejects, f"{name} ({song.title!r}) missing from rejects.jsonl"
        assert song_rejects[song.song_id]["reason"] == expected_reason.value, name

    # The good song must survive and produce at least one chunk.
    chunks = [json.loads(line) for line in open(out_dir / "chunks.jsonl", encoding="utf-8")]
    assert any(c["song_id"] == good.song_id for c in chunks)

    # And no bad song's id should appear as a chunk's song_id.
    bad_ids = {s.song_id for s, _ in BAD_SONGS.values()}
    assert not any(c["song_id"] in bad_ids for c in chunks)
