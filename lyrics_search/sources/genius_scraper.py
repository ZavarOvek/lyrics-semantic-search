"""SPEC-02 §3: Genius.com scraper source -- STUB, not implemented.

Deliberately out of scope for SPEC-02. A real implementation would need,
at minimum:

- Respecting robots.txt and Genius's terms of service / rate limits --
  this is not a bulk-download API; scraping at 30k-song scale needs
  explicit throttling, and the official Genius API (metadata only, no
  lyrics text) plus a licensed lyrics provider is the compliant path,
  not raw HTML scraping.
- Local on-disk caching of fetched pages so repeated ingest runs don't
  re-hit the network for songs already fetched. That caching is also
  what "byte-identical raw.jsonl on repeated runs" (the ingest
  acceptance check) would depend on for this source -- without it,
  determinism against a live, changing website is not achievable.
- Retry/backoff and dead-link/removed-page handling, since scraped raw
  HTML is far less reliable than a curated HF dataset dump.

None of this is implemented here. This class exists only so the Source
protocol has a named placeholder and callers get a clear, loud
NotImplementedError instead of a silently missing source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from lyrics_search.contracts import RawSong


@dataclass
class GeniusScraperSource:
    artist_or_query: str

    def iter_songs(self) -> Iterator[RawSong]:
        raise NotImplementedError(
            "GeniusScraperSource is a stub -- live scraping is out of "
            "SPEC-02 scope. See this module's docstring for what a real "
            "implementation would require (robots.txt/rate-limit "
            "compliance, local caching for reproducibility, retry/backoff)."
        )
        yield  # pragma: no cover -- keeps this a generator function
