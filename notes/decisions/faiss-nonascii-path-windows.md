# faiss `write_index`/`read_index` fail on non-ASCII Windows paths

> **Generalisation, added after a second incident of the same class.** This
> is not a faiss trap. It is a property of *any* tool that hands a path
> straight to the native Windows API instead of going through Python's
> Unicode-aware file layer. Read "Not only faiss" at the end of this file
> before concluding that some new tool is safe on this checkout.

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

## Not only faiss

A second incident, during SUBAGENTS.md, showed this is a class of bug
rather than one dependency's defect. Creating a directory junction so
Claude Code could see `.claude/agents/` inside the repo:

```
cmd /c mklink /J "D:\...\<non-ASCII>\.claude\agents" "D:\...\<non-ASCII>\lyrics-semantic-search\.claude\agents"
```

failed with a mojibake error message -- the message itself came back
with the path already corrupted, which is the tell. The same operation
through PowerShell succeeded on the identical path:

```
powershell -NoProfile -Command "New-Item -ItemType Junction -Path ... -Target ..."
```

`mklink` is a `cmd.exe` builtin, and `cmd.exe` resolves path arguments
through the console's active ANSI codepage before the filesystem call.
PowerShell passes UTF-16 through to the wide-character Windows API.
Nothing about faiss is involved; the failure mode is identical because
the cause is identical.

### What to conclude from this

The rule is about the *interface*, not about any particular library. A
tool is at risk here whenever a path string crosses out of Python (or
another Unicode-correct runtime) into native Windows code:

- C/C++ extensions that call `fopen()` on a path they were given
  (faiss -- and by the same argument, any future index or model format
  that persists itself from native code)
- `cmd.exe` builtins and anything invoked through `cmd /c`
- command-line tools that take a path as an argument and were built
  without a Unicode manifest

Tools that are *not* at risk: anything doing its file I/O through Python
`open()`/`pathlib`, and anything given bytes rather than a path.

So the check to run on a new dependency is narrow and cheap: does it ever
receive a path that it opens itself? If yes, exercise it once on this
checkout before trusting it, rather than on a tmp dir that may sit under
an all-ASCII path. If it does, prefer the same fix used for faiss --
serialise in memory, let Python touch the disk.

A useful corollary: a mangled or mojibake *error message* is itself
evidence of this bug, because it shows the path was already re-encoded
before the failure was reported.
