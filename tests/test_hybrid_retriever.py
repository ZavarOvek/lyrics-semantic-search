"""SPEC-03 §3: HybridRetriever tests -- RRF fusion of dense+lexical
branches, branch_statuses visibility, best_chunk resolution (dense wins
on overlap), and combined status handling when one or both branches
degrade to empty_query/query_out_of_vocabulary."""
from __future__ import annotations

from lyrics_search.contracts import Chunk, Hit, SearchResult
from lyrics_search.retrievers.hybrid import HybridRetriever

CHUNK_A = Chunk(chunk_id="a:0", song_id="a", section=None, text="dense chunk a", position=0, split_by="none")
CHUNK_B = Chunk(chunk_id="b:0", song_id="b", section=None, text="dense chunk b", position=0, split_by="none")
CHUNK_C_LEX = Chunk(chunk_id="c:0", song_id="c", section=None, text="lexical chunk c", position=0, split_by="none")


class FakeRetriever:
    def __init__(self, result: SearchResult):
        self._result = result

    def search(self, query: str, top_k: int) -> SearchResult:
        return self._result


def test_hybrid_fuses_both_branches_via_rrf():
    dense = FakeRetriever(SearchResult(
        hits=[Hit(song_id="a", score=0.9, best_chunk=CHUNK_A), Hit(song_id="b", score=0.5, best_chunk=CHUNK_B)],
        status="ok", query_norm=1.0,
    ))
    lexical = FakeRetriever(SearchResult(
        hits=[Hit(song_id="b", score=3.0, best_chunk=CHUNK_B), Hit(song_id="a", score=1.0, best_chunk=CHUNK_A)],
        status="ok", query_norm=2.0,
    ))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("query", top_k=5)
    assert result.status == "ok"
    assert {h.song_id for h in result.hits} == {"a", "b"}
    assert result.branch_statuses == {"dense": "ok", "lexical": "ok"}


def test_hybrid_song_appearing_first_in_both_ranks_first():
    dense = FakeRetriever(SearchResult(
        hits=[Hit(song_id="a", score=0.9, best_chunk=CHUNK_A), Hit(song_id="b", score=0.5, best_chunk=CHUNK_B)],
        status="ok", query_norm=1.0,
    ))
    lexical = FakeRetriever(SearchResult(
        hits=[Hit(song_id="a", score=3.0, best_chunk=CHUNK_A), Hit(song_id="b", score=1.0, best_chunk=CHUNK_B)],
        status="ok", query_norm=2.0,
    ))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("query", top_k=5)
    assert result.hits[0].song_id == "a"


def test_hybrid_dense_wins_best_chunk_on_overlap():
    dense_only_chunk = Chunk(chunk_id="a:1", song_id="a", section=None, text="dense-picked chunk", position=1, split_by="none")
    dense = FakeRetriever(SearchResult(
        hits=[Hit(song_id="a", score=0.9, best_chunk=dense_only_chunk)], status="ok", query_norm=1.0,
    ))
    lexical = FakeRetriever(SearchResult(
        hits=[Hit(song_id="a", score=3.0, best_chunk=CHUNK_A)], status="ok", query_norm=2.0,
    ))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("query", top_k=5)
    assert result.hits[0].best_chunk.chunk_id == "a:1"  # dense's pick, not lexical's


def test_hybrid_lexical_only_song_uses_lexical_best_chunk():
    dense = FakeRetriever(SearchResult(hits=[], status="ok", query_norm=1.0))
    lexical = FakeRetriever(SearchResult(
        hits=[Hit(song_id="c", score=2.0, best_chunk=CHUNK_C_LEX)], status="ok", query_norm=2.0,
    ))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("query", top_k=5)
    assert result.hits[0].song_id == "c"
    assert result.hits[0].best_chunk.chunk_id == "c:0"


def test_hybrid_both_empty_query_gives_empty_query_status():
    dense = FakeRetriever(SearchResult(hits=[], status="empty_query", query_norm=0.0))
    lexical = FakeRetriever(SearchResult(hits=[], status="empty_query", query_norm=0.0))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("   ", top_k=5)
    assert result.status == "empty_query"
    assert result.hits == []


def test_hybrid_both_oov_gives_oov_status():
    dense = FakeRetriever(SearchResult(hits=[], status="query_out_of_vocabulary", query_norm=0.0))
    lexical = FakeRetriever(SearchResult(hits=[], status="query_out_of_vocabulary", query_norm=0.0))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("zzzqqq", top_k=5)
    assert result.status == "query_out_of_vocabulary"


def test_hybrid_partial_degradation_visible_in_branch_statuses():
    """Dense OOV (e.g. tfidf query had no vocab match) but lexical still
    found keyword hits -- overall status should be "ok", with the
    degraded branch visible in branch_statuses, not silently hidden."""
    dense = FakeRetriever(SearchResult(hits=[], status="query_out_of_vocabulary", query_norm=0.0))
    lexical = FakeRetriever(SearchResult(
        hits=[Hit(song_id="c", score=2.0, best_chunk=CHUNK_C_LEX)], status="ok", query_norm=1.0,
    ))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("query", top_k=5)
    assert result.status == "ok"
    assert result.branch_statuses == {"dense": "query_out_of_vocabulary", "lexical": "ok"}
    assert result.hits[0].song_id == "c"


def test_hybrid_query_norm_is_dense_branch_norm():
    dense = FakeRetriever(SearchResult(hits=[], status="ok", query_norm=0.42))
    lexical = FakeRetriever(SearchResult(hits=[], status="ok", query_norm=7.0))
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("query", top_k=5)
    assert result.query_norm == 0.42
