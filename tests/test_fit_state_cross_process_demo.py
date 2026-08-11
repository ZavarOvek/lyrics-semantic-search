"""Cross-process determinism of a fitted embedder state, on the demo corpus.

`test_fitted_state_persistence.py` pins the same property on the *real*
`data/dev` build and the 2 GiB fastText vectors, so it can only ever run on a
machine that has them -- in CI it skips. But determinism of `load_fit()` is a
property of the code, not of the data: 200 synthetic songs exercise it just as
well as 30k real ones, and they build in seconds with no download.

So this file covers the property where it can actually run, and the other one
keeps covering it on the real artifacts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = REPO_ROOT / "data_demo" / "demo" / "sections"
FIT_PATH = DEMO_ROOT / "embeddings" / "tfidf" / "fitted_state.joblib"
GENERATOR = REPO_ROOT / "scripts" / "generate_demo_corpus.py"

# Deliberately made of words the synthetic generator actually uses, so the
# encoded vector is non-degenerate -- see the nnz assertion below.
QUERY = "walking alone in the rain thinking about you"


@pytest.fixture(scope="session")
def demo_fit_state() -> Path:
    """Build the demo corpus once if its fitted state is not on disk.

    Building rather than skipping is the point: a skipped test in a green run
    is indistinguishable from a passing one, and this build costs seconds and
    downloads nothing.
    """
    if not FIT_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"demo corpus build failed:\n{result.stdout}\n{result.stderr}"
        )
    assert FIT_PATH.exists(), f"demo build did not produce {FIT_PATH}"
    return FIT_PATH


def _encode_in_subprocess(out_path: Path) -> np.ndarray:
    # pytest's `pythonpath` setting does not reach a plain subprocess, so the
    # package root has to be handed over explicitly.
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, __file__, "--out", str(out_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"encode subprocess failed:\n{result.stdout}\n{result.stderr}"
    return np.load(out_path)


def test_two_processes_encode_the_query_identically(demo_fit_state, tmp_path):
    vec_a = _encode_in_subprocess(tmp_path / "a.npy")
    vec_b = _encode_in_subprocess(tmp_path / "b.npy")

    # Guard against the test quietly degenerating into "0 == 0": if the demo
    # corpus or the query ever drift apart, an all-OOV query would encode to a
    # zero vector and the comparison below would pass while proving nothing.
    assert np.count_nonzero(vec_a) > 0, "query encoded to an all-zero vector"
    assert vec_a.shape == vec_b.shape
    assert np.array_equal(vec_a, vec_b), (
        "two processes disagreed on the same query: "
        f"nnz={np.count_nonzero(vec_a)}/{np.count_nonzero(vec_b)}, "
        f"norm={np.linalg.norm(vec_a):.6f}/{np.linalg.norm(vec_b):.6f}"
    )


def _phase(out_path: Path) -> None:
    from lyrics_search.core.vectors import as_dense_vector
    from lyrics_search.embedders.tfidf import TfidfEmbedder

    embedder = TfidfEmbedder()
    # load_fit(), never fit(): re-fitting would make two processes agree only
    # by accident, which is exactly what this test exists to rule out.
    embedder.load_fit(FIT_PATH)
    vector = embedder.encode([QUERY], is_query=True)[0]
    np.save(out_path, as_dense_vector(vector))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    _phase(Path(parser.parse_args().out))
