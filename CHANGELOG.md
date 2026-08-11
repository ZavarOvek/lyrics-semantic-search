# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- GitHub Actions CI: a `ruff` lint job (`check` + `format --check`) and a
  `pytest` matrix on Python 3.11 and 3.12, installing only
  `requirements.txt` — no torch, no faiss, no model downloads.
- CI enforces `--cov-fail-under=100` on `lyrics_search/core`, so the
  coverage claim in the README is checked rather than asserted.
- `tests/test_fit_state_cross_process_demo.py`: cross-process determinism of
  a fitted embedder state, on the synthetic demo corpus. The existing
  real-data test can only run on a machine with the 2 GiB fastText vectors
  and a built `data/dev`, so in CI it skips; determinism is a property of
  the code, so 200 synthetic songs pin it just as well. The fixture builds
  the demo corpus when absent rather than skipping.
- `test_vector_norm_handles_a_dense_row`: the sparse branch of `vector_norm`
  was pinned, the dense one was not. `core/` is now genuinely at 100%.
- Ruff configuration in `pyproject.toml`; `SLF001` disabled for `tests/`.

### Fixed
- `tests/test_fitted_state_persistence.py` hardcoded
  `.venv/Scripts/python.exe`, a Windows-only path. On Linux and macOS the
  interpreter check failed first, so both tests skipped silently and for the
  wrong reason, reporting a missing interpreter instead of missing data.
  Now uses `sys.executable`.
- `tests/test_config.py` asserted `pytest.raises(Exception)` on three
  validation tests; they now expect `pydantic.ValidationError`. As written,
  a test named "rejects unknown field" would also have passed if the config
  class had failed to construct at all.
- Four `pytest.raises(match=...)` patterns contained unescaped regex
  metacharacters (`match="vectors.np"` also matches `vectorsXnp`); wrapped
  in `re.escape`.
- Files opened without a context manager in `phase0.py`, `scripts/` and
  three test modules.

### Changed
- `zip()` calls are explicit about `strict=True` where the operands are
  equal-length by construction.
- Adopted `ruff format`; the tree was reformatted in a single dedicated
  commit (see `.git-blame-ignore-revs`).
- `RejectReason` keeps `(str, Enum)` with an explicit `noqa: UP042`:
  `StrEnum` would change `str()` from `RejectReason.X` to `x`, and those
  values are written into `rejects.jsonl`.
- The unreachable `idcg == 0.0` guard in `core/metrics.py` is marked
  `# pragma: no cover` — the guards above it leave it with no path through
  the public API, and faking one would test a mock.
