"""EVAL-PREP §3: the `whole_song` chunking arm.

Three things need to hold for the arm to be a fair comparison rather than
a second pipeline: it emits exactly one segment per non-empty song, it
strips structural markup exactly as the sections cascade does (so
chunking is the only variable between the arms), and it reaches all the
way through build+search selected from config alone.

The fourth thing is a *negative* result that EVAL-PREP asks about
explicitly -- whether aggregation needed changing. It did not, and
test_whole_song_aggregation_needs_no_special_casing pins that: the same
`aggregate_chunks_to_songs` used by the chunked arm returns one hit per
song when handed one chunk per song.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lyrics_search.config import ExperimentConfig, RetrievalConfig
from lyrics_search.contracts import Chunk
from lyrics_search.core.aggregate import aggregate_chunks_to_songs
from lyrics_search.core.sections import (
    CHUNKING_SECTIONS,
    CHUNKING_WHOLE_SONG,
    chunk_song,
    chunk_song_with,
    whole_song_segments,
)
from lyrics_search.runners.build import run_build
from lyrics_search.runners.search import run_search
from lyrics_search.sources.jsonl_file import JsonlFileSource

GOLDEN = Path(__file__).parent / "golden" / "spec03_mini_corpus.jsonl"

TAGGED = """[Intro]
one two three

[Verse 1]
four five six
seven eight nine

[Chorus]
ten eleven twelve
"""


# --- whole_song_segments ----------------------------------------------


def test_whole_song_emits_exactly_one_segment():
    segments = whole_song_segments(TAGGED)
    assert len(segments) == 1


def test_whole_song_emits_nothing_for_empty_text():
    assert whole_song_segments("") == []
    assert whole_song_segments("   \n\n  ") == []


def test_whole_song_strips_structural_markup():
    text = whole_song_segments(TAGGED)[0].text
    assert "[" not in text and "]" not in text
    assert "Verse" not in text and "Chorus" not in text


def test_whole_song_keeps_every_lyric_word_from_the_chunked_arm():
    """The arms must differ in *segmentation only*: the union of the
    chunked arm's words is exactly the whole-song segment's words."""
    chunked_words = " ".join(s.text for s in chunk_song(TAGGED)).split()
    whole_words = whole_song_segments(TAGGED)[0].text.split()
    assert whole_words == chunked_words


def test_whole_song_labels_are_degenerate_not_inherited():
    seg = whole_song_segments(TAGGED)[0]
    assert seg.section is None
    assert seg.split_by == "none"
    assert seg.force_split is False


def test_whole_song_ignores_the_hard_ceiling():
    long_text = " ".join(f"w{i}" for i in range(1000))
    assert len(whole_song_segments(long_text)) == 1
    assert len(chunk_song(long_text)) > 1


# --- chunk_song_with dispatch ------------------------------------------


def test_dispatch_sections_matches_chunk_song():
    assert chunk_song_with(TAGGED, CHUNKING_SECTIONS) == chunk_song(TAGGED)


def test_dispatch_whole_song_matches_whole_song_segments():
    assert chunk_song_with(TAGGED, CHUNKING_WHOLE_SONG) == whole_song_segments(TAGGED)


def test_dispatch_default_is_sections():
    assert chunk_song_with(TAGGED) == chunk_song(TAGGED)


def test_dispatch_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="whole_song"):
        chunk_song_with(TAGGED, "paragraphs")


# --- aggregation, the question EVAL-PREP asks --------------------------


def test_whole_song_aggregation_needs_no_special_casing():
    chunk_scores = [
        (
            Chunk(
                chunk_id="a:0",
                song_id="a",
                section=None,
                text="x",
                position=0,
                split_by="none",
                force_split=False,
            ),
            0.9,
        ),
        (
            Chunk(
                chunk_id="b:0",
                song_id="b",
                section=None,
                text="y",
                position=0,
                split_by="none",
                force_split=False,
            ),
            0.4,
        ),
    ]
    hits = aggregate_chunks_to_songs(chunk_scores)
    assert [h.song_id for h in hits] == ["a", "b"]
    assert [h.score for h in hits] == [0.9, 0.4]
    assert [h.best_chunk.chunk_id for h in hits] == ["a:0", "b:0"]


# --- config -> build -> search -----------------------------------------


def test_config_selects_the_arm_end_to_end(tmp_path):
    config = ExperimentConfig(
        corpus="demo",
        chunking="whole_song",
        embedder="tfidf",
        index="numpy",
        retrieval=RetrievalConfig(mode="dense", top_k=10, return_n=5),
    )
    work_dir = run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    assert work_dir == tmp_path / "demo" / "whole_song"

    result, _timing = run_search(config, "mountain climbing summit", data_root=tmp_path)
    assert result.status == "ok"
    assert len(result.hits) > 0


def test_the_two_arms_coexist_on_disk(tmp_path):
    """SPEC-04 §0.1's lesson applied to chunking: building one arm must
    not evict the other's artifacts."""
    source = JsonlFileSource(GOLDEN)
    base = {"corpus": "demo", "embedder": "tfidf", "index": "numpy"}
    sections_dir = run_build(
        ExperimentConfig(chunking="sections", **base), data_root=tmp_path, source=source
    )
    whole_dir = run_build(
        ExperimentConfig(chunking="whole_song", **base), data_root=tmp_path, source=source
    )

    assert sections_dir != whole_dir
    for d in (sections_dir, whole_dir):
        assert (d / "chunks.jsonl").exists()
        assert (d / "indexes" / "tfidf" / "numpy").is_dir()

    n_sections = len(
        (sections_dir / "chunks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    n_whole = len((whole_dir / "chunks.jsonl").read_text(encoding="utf-8").strip().splitlines())
    assert n_whole < n_sections

    # ingest is chunking-independent, so raw.jsonl is shared, not duplicated
    assert (tmp_path / "demo" / "raw.jsonl").exists()
    assert not (sections_dir / "raw.jsonl").exists()
    assert not (whole_dir / "raw.jsonl").exists()
