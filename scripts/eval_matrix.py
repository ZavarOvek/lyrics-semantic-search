"""EVAL-AUTO §5: run the nine-arm matrix and log GPU state around each arm.

    ./.venv/Scripts/python.exe scripts/eval_matrix.py \
        --eval-set data/eval/auto/queries.jsonl --out-dir data/eval/auto/results

## Why nine arms and not twelve

Four dense + four hybrid + one lexical. BM25 depends on neither the
embedder nor the chunking arm: `build_retriever` resolves no embedder in
lexical mode, and the corpus it scores is `text_raw` from `raw.jsonl`,
shared across arms. The four nominal lexical cells would return
bit-identical numbers, so three of them are not run rather than reported
as if they were independent evidence.

## Why two runs per arm

`--stratum main` gives the headline table: `structure` and `translation`
are deliberately over-sampled, so an `(overall)` row across all three
describes no real population. The second run takes the whole set purely
for its `stratum` slice, which carries the over-sampled arms' numbers with
their own `n` and interval. Their rows are read from there and reported
apart from the headline, never merged into it.

## Why the GPU is sampled between arms and not once at the end

The arms run back to back on a shared laptop GPU. If the card heats and
clocks down over the sequence, every later arm is measured under worse
conditions than every earlier one, and the ranking picks up a drift term
that has nothing to do with retrieval quality. A single before/after pair
would prove drift happened without saying where, so the card is sampled
around every run and the log is reported alongside the table. The check is
cheap and it is the only thing standing between a thermal artifact and a
conclusion about embedders.

Two of the nine arms (`fasttext-avg`, `tfidf`) never touch the GPU at all,
which makes them a natural control: if their timings move across the
sequence too, the cause is not the card.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ARMS: list[tuple[str, str]] = [
    ("dense-bge-m3", "configs/full_dense_faiss.yaml"),
    ("dense-e5-base", "configs/full_dense_faiss_e5base.yaml"),
    ("dense-fasttext-avg", "configs/full_dense_faiss_fasttext.yaml"),
    ("dense-tfidf", "configs/full_dense_numpy_tfidf.yaml"),
    ("hybrid-bge-m3", "configs/full_hybrid_faiss.yaml"),
    ("hybrid-e5-base", "configs/full_hybrid_faiss_e5base.yaml"),
    ("hybrid-fasttext-avg", "configs/full_hybrid_faiss_fasttext.yaml"),
    ("hybrid-tfidf", "configs/full_hybrid_numpy_tfidf.yaml"),
    ("lexical-bm25", "configs/full_lexical.yaml"),
]

GPU_FIELDS = (
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "clocks.sm",
    "clocks.max.sm",
    "power.draw",
)


def sample_gpu(label: str) -> dict:
    """One nvidia-smi reading. Absence of the tool is recorded, not fatal:
    two of the nine arms are CPU-only and must still be runnable on a box
    without a card."""
    sample = {"at": time.strftime("%H:%M:%S"), "label": label}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        sample["error"] = str(exc)
        return sample
    values = [v.strip() for v in out.split(",")]
    for name, value in zip(GPU_FIELDS, values, strict=True):
        sample[name] = float(value) if value.replace(".", "", 1).isdigit() else value
    return sample


def run_one(
    config: str, eval_set: str, json_out: Path, log_out: Path, stratum: str | None
) -> float:
    cmd = [
        sys.executable,
        "-m",
        "lyrics_search.runners.eval",
        "--config",
        config,
        "--eval-set",
        eval_set,
        "--json-out",
        str(json_out),
    ]
    if stratum:
        cmd += ["--stratum", stratum]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log_out.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"{config} (stratum={stratum}) failed:\n{result.stdout}\n{result.stderr}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="data/eval/auto/queries.jsonl")
    parser.add_argument("--out-dir", default="data/eval/auto/results")
    parser.add_argument("--only", default=None, help="run one arm by name")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    arms = [a for a in ARMS if args.only is None or a[0] == args.only]
    if not arms:
        raise SystemExit(f"--only {args.only!r} matched no arm of {[a for a, _ in ARMS]}")

    telemetry = [sample_gpu("before-matrix")]
    timings = []
    for i, (name, config) in enumerate(arms, 1):
        print(f"[{i}/{len(arms)}] {name}", flush=True)
        telemetry.append(sample_gpu(f"before/{name}"))
        main_s = run_one(
            config,
            args.eval_set,
            out_dir / f"{name}.main.json",
            log_dir / f"{name}.main.log",
            "main",
        )
        telemetry.append(sample_gpu(f"mid/{name}"))
        all_s = run_one(
            config, args.eval_set, out_dir / f"{name}.all.json", log_dir / f"{name}.all.log", None
        )
        telemetry.append(sample_gpu(f"after/{name}"))
        timings.append({"arm": name, "main_s": round(main_s, 1), "all_s": round(all_s, 1)})
        print(f"      main {main_s:.0f}s   all {all_s:.0f}s", flush=True)

    telemetry.append(sample_gpu("after-matrix"))

    (out_dir / "gpu.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in telemetry) + "\n", encoding="utf-8"
    )
    (out_dir / "timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    temps = [s["temperature.gpu"] for s in telemetry if "temperature.gpu" in s]
    if temps:
        print(f"\nGPU temp over the sequence: {min(temps):.0f}C .. {max(temps):.0f}C")
        print(f"  first sample {temps[0]:.0f}C, last sample {temps[-1]:.0f}C")
    print(f"wrote {out_dir}/gpu.jsonl, {out_dir}/timings.json and {2 * len(arms)} tables")


if __name__ == "__main__":
    main()
