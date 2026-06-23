"""Build one master (model, method, benchmark) table across the whole runs/ tree.

evaluate_run.py and aggregate_results.py both assume a single run_id holds
the full model x method matrix for a benchmark. In this project each run_id
is scoped to one model (see each run's config.yaml header), so the 5-model x
8-method x 2-benchmark matrix is spread across many run folders. This script
discovers the right rows for each cell directly from the data rather than
from filenames or run_ids:

  - keep only prompt_version == "v1" (excludes prompt-ablation runs, which
    use v2/v3 and are a separate study)
  - keep only rows whose model_name is one of the five real models (excludes
    DummyBackend smoke-test rows, whose model_name is "dummy://")
  - keep only rows whose method_name is one of the eight registered methods
    (excludes "twostage_semantic_match", a retired method with complete
    leftover data for two models that predates the current text_extraction /
    abcd runners)
  - dedupe by (model, method, benchmark, question_id), keeping the latest
    timestamp; this absorbs harmless 5-row tiny-smoke-test overlaps without
    needing to know which run folder was the smoke test

A combo whose surviving question_id set doesn't exactly equal the canonical
1000-ID split for that benchmark is marked incomplete and excluded from any
written output.

Usage:
    python scripts/build_master_table.py             # print only, write nothing
    python scripts/build_master_table.py --write      # also write reports/master_table.csv
                                                        # (refuses if any combo is incomplete)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_run import (  # noqa: E402
    BOOTSTRAP_SEED,
    OPTIONS,
    _bootstrap_ci_mean_abs_deviation,
    _clopper_pearson_ci,
    rematch_abcd_rows,
    rematch_additional_option_rows,
    rematch_text_extraction_rows,
    reparse_run,
)
from modelgen.config.paths import REPORTS_DIR, RUNS_DIR  # noqa: E402

EXPECTED_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]
EXPECTED_METHODS = [
    "baseline",
    "two_prompt",
    "cyclic",
    "pride",
    "calibration",
    "additional_option",
    "text_extraction",
    "abcd",
    "independent_hypothesis",
]
EXPECTED_BENCHMARKS = ["mmlu", "arc_challenge"]

# Steady-state backend calls per question, confirmed against the runner
# source (src/modelgen/runners/*.py), not inferred from row counts:
#   - two_prompt: 2 real generate() calls (stage1 free-text, stage2 matching).
#     fallback_on_parse_failure defaults False and is False in every run
#     config actually used, so no conditional 3rd call occurs in practice.
#   - cyclic: 4 cyclic-permutation generate() calls.
#   - pride / calibration: 1 score_options() call per eval question. Each
#     also has a one-time setup cost NOT included in this per-question
#     number: pride fits its prior from calibration_n (default 50) calibration
#     questions x 4 permutations = 200 calls, amortized ONCE per
#     (model, benchmark) via a JSON sidecar cache reused across reruns;
#     calibration makes 1 neutral-prompt call, but has no sidecar, so it is
#     repeated every run (not amortized across reruns).
#   - text_extraction / abcd: 1 call; stage 2 is deterministic embedding
#     cosine similarity, not a second model call.
#   - independent_hypothesis: 1 generate() call per real option (usually 4;
#     3 for the rare ARC-Challenge question with no D choice).
CALLS_PER_QUESTION = {
    "baseline": 1,
    "two_prompt": 2,
    "cyclic": 4,
    "pride": 1,
    "calibration": 1,
    "additional_option": 1,
    "text_extraction": 1,
    "abcd": 1,
    "independent_hypothesis": 4,
}

CANONICAL_IDS = {
    "mmlu": set(json.loads((_ROOT / "data/splits/benchmark/robustness_ids.json").read_text())),
    "arc_challenge": set(
        json.loads((_ROOT / "data/splits/arc_challenge/robustness_ids.json").read_text())
    ),
}


def discover_rows(runs_dir: Path) -> pd.DataFrame:
    """Load and dedupe every relevant row across all run folders. See module docstring."""
    frames = []
    for path in sorted(runs_dir.glob("*/*.csv")):
        df = pd.read_csv(path, low_memory=False)
        if "model_name" not in df.columns:
            continue
        df = df[
            (df["prompt_version"] == "v1")
            & (df["model_name"].isin(EXPECTED_MODELS))
            & (df["method_name"].isin(EXPECTED_METHODS))
        ]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("timestamp_utc").drop_duplicates(
        subset=["model_name", "method_name", "benchmark", "question_id"], keep="last"
    )
    return full.reset_index(drop=True)


def build_master_table(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    for model in EXPECTED_MODELS:
        for method in EXPECTED_METHODS:
            for bench in EXPECTED_BENCHMARKS:
                group = df[
                    (df["model_name"] == model)
                    & (df["method_name"] == method)
                    & (df["benchmark"] == bench)
                ]
                canon = CANONICAL_IDS[bench]
                present_ids = set(group["question_id"])
                complete = present_ids == canon

                total = len(group)
                correct = int(group["is_correct"].eq(True).sum())

                if total > 0:
                    e2e = correct / total
                    e2e_lo, e2e_hi = _clopper_pearson_ci(correct, total)
                else:
                    e2e, e2e_lo, e2e_hi = float("nan"), float("nan"), float("nan")

                scored_group = group[group["parsed_choice"].notna()]
                if total > 0 and len(scored_group) > 0:
                    gt_counts = group["correct_option"].value_counts()
                    pred_counts = scored_group["parsed_choice"].value_counts()
                    devs = [
                        abs(
                            pred_counts.get(o, 0) / len(scored_group) * 100
                            - gt_counts.get(o, 0) / total * 100
                        )
                        for o in OPTIONS
                    ]
                    mad = sum(devs) / len(devs)
                    mad_lo, mad_hi = _bootstrap_ci_mean_abs_deviation(group, rng)
                else:
                    mad, mad_lo, mad_hi = float("nan"), float("nan"), float("nan")

                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "benchmark": bench,
                        "complete": complete,
                        "n_rows": total,
                        "missing_ids": len(canon - present_ids),
                        "end_to_end_accuracy": e2e,
                        "end_to_end_accuracy_ci_lower": e2e_lo,
                        "end_to_end_accuracy_ci_upper": e2e_hi,
                        "mean_abs_deviation": mad,
                        "mean_abs_deviation_ci_lower": mad_lo,
                        "mean_abs_deviation_ci_upper": mad_hi,
                        "calls_per_question": CALLS_PER_QUESTION[method],
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the master (model, method, benchmark) table across all of runs/."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write reports/master_table.csv. Refuses if any combo is incomplete.",
    )
    args = parser.parse_args()

    df = discover_rows(RUNS_DIR)
    n_combos = df[["model_name", "method_name", "benchmark"]].drop_duplicates().shape[0]
    print(f"[master] Loaded {len(df)} rows across {n_combos} (model, method, benchmark) combos")

    print("[master] Re-parsing with current parser...")
    df = reparse_run(df)

    if (df["method_name"] == "abcd").any():
        print("[master] Re-matching abcd responses with current embedding config...")
        df = rematch_abcd_rows(df)

    if (df["method_name"] == "text_extraction").any():
        print("[master] Re-matching text_extraction responses with current embedding config...")
        df = rematch_text_extraction_rows(df)

    if (df["method_name"] == "additional_option").any():
        print("[master] Re-matching additional_option responses with current Jaccard config...")
        df = rematch_additional_option_rows(df)

    table = build_master_table(df)

    incomplete = table[~table["complete"]]
    if not incomplete.empty:
        print(f"\n[master] WARNING: {len(incomplete)} / {len(table)} combos incomplete:")
        print(
            incomplete[["model", "method", "benchmark", "n_rows", "missing_ids"]].to_string(
                index=False
            )
        )

    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))

    if args.write:
        if not incomplete.empty:
            print(
                "\n[master] Refusing to write reports/master_table.csv: "
                f"{len(incomplete)} combo(s) still incomplete."
            )
            return
        out_path = REPORTS_DIR / "master_table.csv"
        table.to_csv(out_path, index=False)
        print(f"\n[master] Wrote {out_path}")


if __name__ == "__main__":
    main()
