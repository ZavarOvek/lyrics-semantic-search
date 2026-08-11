"""SPEC-02: full-corpus dedup check (one-off analysis script, not part of the pipeline).

Checks data/full/raw.jsonl (30000 songs) for:
1. normalized (artist, title) collisions -- cross-check against song_id,
   which is already sha1(norm(artist)|norm(title))[:16] per contracts.py.
2. exact text_raw duplicates -- same lyrics text under different song_id
   (different artist/title metadata), which song_id dedup would NOT catch.

Read-only analysis; writes nothing back into data/.
"""

from __future__ import annotations

import json
from collections import defaultdict

from lyrics_search.core.text import normalize_for_id, hash_block

RAW_PATH = "data/full/raw.jsonl"


def main() -> None:
    by_song_id: dict[str, list[dict]] = defaultdict(list)
    by_norm_artist_title: dict[str, list[dict]] = defaultdict(list)
    by_text_hash: dict[str, list[dict]] = defaultdict(list)

    n = 0
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            by_song_id[rec["song_id"]].append(rec)
            key = f"{normalize_for_id(rec['artist'])}|{normalize_for_id(rec['title'])}"
            by_norm_artist_title[key].append(rec)
            th = hash_block(rec["text_raw"])
            by_text_hash[th].append(rec)

    print(f"Total songs read: {n}")

    dupe_ids = {k: v for k, v in by_song_id.items() if len(v) > 1}
    print(f"\n1. song_id collisions (normalized artist|title, same as ingest's own key):")
    print(
        f"   {len(dupe_ids)} colliding song_id groups covering {sum(len(v) for v in dupe_ids.values())} records"
    )
    for k, v in list(dupe_ids.items())[:5]:
        print(f"   - {k}: {[(r['artist'], r['title']) for r in v]}")

    dupe_at = {k: v for k, v in by_norm_artist_title.items() if len(v) > 1}
    print(f"\n2. normalized (artist,title) collisions (independent cross-check):")
    print(
        f"   {len(dupe_at)} colliding groups covering {sum(len(v) for v in dupe_at.values())} records"
    )
    assert len(dupe_at) == len(dupe_ids), "song_id and normalized artist|title dedup disagree!"

    dupe_text = {k: v for k, v in by_text_hash.items() if len(v) > 1}
    total_dupe_text_records = sum(len(v) for v in dupe_text.values())
    print(f"\n3. exact text_raw duplicates (different metadata, identical lyrics):")
    print(f"   {len(dupe_text)} colliding text groups covering {total_dupe_text_records} records")
    print(
        f"   ({total_dupe_text_records - len(dupe_text)} 'extra' records that a text-based dedup would remove)"
    )
    for k, v in list(dupe_text.items())[:15]:
        pairs = [(r["artist"], r["title"], r["song_id"]) for r in v]
        print(f"   - {pairs}")


if __name__ == "__main__":
    main()
