# faiss `write_index`/`read_index` fail on non-ASCII Windows paths

**Status:** found and fixed during SPEC-03 Phase C. Recorded here as its own
file per SPEC-03-PATCH §6, because it's a concrete platform trap worth
citing directly in a README later, not a detail worth re-deriving from
`git log` after the fact.

## Symptom

`faiss.write_index(index, path)` and `faiss.read_index(path)` -- the
library's standard, documented save/load API -- silently fail (or raise an
opaque `RuntimeError: Error: 'blah blah' failed` from inside faiss's C++
layer) whenever `path` contains a character outside the machine's active
ANSI codepage, most commonly any non-Latin script.

## Why it hit this project specifically

The root directory of the checkout contains non-ASCII (Cyrillic)
characters. Every absolute path faiss would ever be asked to open in this
repo -- every `data/<corpus>/<chunking>/indexes/<embedder>/faiss/index.faiss`,
any test tmp dir faiss touched via a path derived from the repo root --
passes through that non-ASCII segment.

So this wasn't a contrived edge case discovered by deliberately testing
exotic paths. It's the *default* condition of every real save/load faiss
would ever perform in this repo. Left unfixed, it would have surfaced the
first time anyone actually tried to persist a faiss index here -- most
likely well into an eval phase or a later demo, far from where the root
cause (a file-I/O detail three layers down in a C++ dependency) would be
obvious.

## Root cause

faiss's C++ `write_index`/`read_index` call `fopen()` (or the Windows
equivalent, `_wfopen`/ANSI `fopen` depending on build) directly on the given
path string. On Windows, this file-open path does not reliably use UTF-8 or
UTF-16 for arbitrary faiss builds -- non-ASCII bytes in the path get
mangled or rejected before the OS-level file API ever sees a valid path.
This is a faiss/Windows interop gap, not something fixable from the Python
side of the call.

## Fix

`lyrics_search/indexes/faiss_index.py`'s `save()`/`load()` never call
`faiss.write_index()`/`read_index()`. Instead:

- `save()`: `faiss.serialize_index(self._index).tobytes()` turns the index
  into an in-memory `bytes` object entirely inside faiss's C++ layer (no
  filesystem access at all on faiss's side), then plain Python `open(path, "wb")`
  writes those bytes to disk. Python's own file I/O handles Unicode paths
  correctly on Windows.
- `load()`: the mirror image -- `open(path, "rb").read()` via Python, then
  `faiss.deserialize_index(np.frombuffer(raw_bytes, dtype=np.uint8))`
  reconstructs the index from the in-memory bytes.

This sidesteps the bug entirely: faiss's C++ code never sees a path string,
so it never has the opportunity to mishandle one.

## Test coverage

`tests/test_indexes.py::test_faiss_index_save_load_roundtrip` builds a small
index, saves it to a tmp directory, reloads it, and checks search results
match before/after the roundtrip. Uses `pytest.importorskip("faiss")` so it
skips (not fails) in any environment without faiss installed -- which is
why it showed as 2 skipped tests rather than 2 passing tests whenever the
suite was run against the wrong (non-project) Python interpreter, see
`notes/reports/spec03-patch-report.md` §2. Confirmed passing against a real faiss
install (both during Phase C and again in SPEC-03-PATCH's environment
re-verification, this time via `.venv`, which does have faiss).
