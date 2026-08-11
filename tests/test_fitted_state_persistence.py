"""SPEC-02-PATCH item 3 acceptance check, promoted to a real pytest test
(SPEC-04 notes refactor) -- was notes/verify_fitted_state_persistence.py.

"Encode the same query in two separate processes and get an identical
vector, for both fit-dependent embedders."

Deliberately split into two subprocess invocations of this same file
(--phase X --out Y) rather than two in-process objects, so "separate
processes" is actually true -- no shared Python state, module cache, or
object identity between the two encodes. Each subprocess:
  1. constructs a *fresh* embedder instance,
  2. calls load_fit() on the fitted_state.joblib produced by a real
     data/dev/sections/embeddings/<name>/ build (never calls fit() itself),
  3. encodes the same fixed query string,
  4. saves the resulting vector to a temp .npy file.

EVAL-PREP §1: `tfidf` now encodes to `scipy.sparse`, so the transport
between subprocess and test goes through `as_dense_vector()`. That is a
change to how the vector is *carried between processes*, not to what is
being asserted -- the property under test is still "the same fitted state
produces a bit-identical vector in two separate processes". Densifying is
safe here because it is one query row, bounded by vocabulary size, not by
corpus size.

Requires a real `data/dev` build (tfidf + fasttext-avg embedded) and the
real fasttext vectors file to exist on disk -- both are real, large
artifacts deliberately excluded from git (see the README's "Data &
copyright": `data/` is never committed), so this test skips cleanly,
rather than failing, when they're absent (a fresh clone, or CI without
the real corpus built).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lyrics_search.core.vectors import as_dense_vector

QUERY = "walking alone in the rain thinking about you"
REPO_ROOT = Path(__file__).resolve().parent.parent
# The subprocess must run under the *same* interpreter as pytest. A hardcoded
# ".venv/Scripts/python.exe" is Windows-only, so on Linux/macOS — and in CI —
# the path never exists and these tests skip silently even when the real data
# is present. A silent skip in a green run is indistinguishable from a pass.
PY = sys.executable

FASTTEXT_VECTORS = REPO_ROOT / "data" / "models" / "fasttext" / "cc.en.300.vec"
_DEV_EMBEDDINGS = REPO_ROOT / "data" / "dev" / "sections" / "embeddings"
FIT_PATHS = {
    "tfidf": _DEV_EMBEDDINGS / "tfidf" / "fitted_state.joblib",
    "fasttext-avg": _DEV_EMBEDDINGS / "fasttext-avg" / "fitted_state.joblib",
}


def _phase(name: str, out_path: Path) -> None:
    if name == "tfidf":
        from lyrics_search.embedders.tfidf import TfidfEmbedder

        embedder = TfidfEmbedder()
    elif name == "fasttext-avg":
        from lyrics_search.embedders.fasttext_avg import FastTextAvgEmbedder

        embedder = FastTextAvgEmbedder(str(FASTTEXT_VECTORS), limit=50_000)
    else:
        raise SystemExit(f"unknown embedder {name}")

    # The whole point: load_fit(), never fit(). If this were silently
    # re-fitting instead, two processes started at different times / with
    # different corpora would NOT reliably agree.
    embedder.load_fit(FIT_PATHS[name])
    vec = embedder.encode([QUERY], is_query=True)[0]
    np.save(out_path, as_dense_vector(vec))


def _run_subprocess(embedder_name: str, out_path: Path) -> None:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [str(PY), __file__, "--phase", embedder_name, "--out", str(out_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"subprocess for {embedder_name} failed:\n{result.stdout}\n{result.stderr}"
    )


def _missing_real_data_reason() -> str | None:
    if not FASTTEXT_VECTORS.exists():
        return f"missing real fasttext vectors file: {FASTTEXT_VECTORS}"
    for name, path in FIT_PATHS.items():
        if not path.exists():
            return f"missing real data/dev embed artifact for {name!r}: {path}"
    return None


@pytest.mark.parametrize("embedder_name", ["tfidf", "fasttext-avg"])
def test_cross_process_fit_state_identical_vectors(embedder_name: str, tmp_path: Path) -> None:
    reason = _missing_real_data_reason()
    if reason:
        pytest.skip(f"requires a real data/dev build -- {reason}")

    vec_a_path = tmp_path / "a.npy"
    vec_b_path = tmp_path / "b.npy"
    _run_subprocess(embedder_name, vec_a_path)
    _run_subprocess(embedder_name, vec_b_path)
    vec_a = np.load(vec_a_path)
    vec_b = np.load(vec_b_path)

    # SPEC-03 §0.1: vec[:5] is a misleading display for a high-dimensional
    # sparse-ish vector -- report nnz/norm/sum instead on failure, since
    # they actually distinguish a genuinely-empty (OOV) vector from a
    # nonzero one.
    assert np.array_equal(vec_a, vec_b), (
        f"{embedder_name}: cross-process vectors differ "
        f"(nnz={np.count_nonzero(vec_a)}/{np.count_nonzero(vec_b)}, "
        f"norm={np.linalg.norm(vec_a):.6f}/{np.linalg.norm(vec_b):.6f})"
    )


if __name__ == "__main__":
    # Subprocess entrypoint only -- pytest never executes this block, it
    # imports the module and calls the test function above directly.
    idx = sys.argv.index("--phase")
    name = sys.argv[idx + 1]
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    _phase(name, out)
