"""Prepare checkpoints so a `run_experiment.py --run-id X --config Y --yes`
resume regenerates only specific question_ids instead of redoing a whole job.

Background: 3 ARC-Challenge robustness-split questions (79e8c959bbeb74a0,
ad6b5d46ae54842c, c30e75b011696a95) have only 3 real answer options (no
choice_d). Before the prompt_builder fix, every runner rendered a phantom
"D. nan" option for these 3 questions. Two methods (abcd, text_extraction)
additionally crashed on them, which silently dropped their whole 50-question
checkpoint batch for 2 models (Qwen-7B, Llama-8B), leaving those 4 combos at
850/1000. The other 36 (model, method) ARC-Challenge combos completed
"successfully" but with the 3 contaminated rows baked in.

`run_experiment.py` already resumes correctly from a checkpoint (skips
question_ids in `completed_ids`, regenerates the rest, overwrites the run's
CSV with the merged result). The gap: checkpoints are deleted when a job
finishes, even if it finished with contaminated or missing rows, so there is
nothing left to resume from. This script reconstructs that checkpoint state
directly from each run's existing CSV:

  - PATCHES: combo is already 1000/1000. Drop the 3 contaminated rows so
    they regenerate with the fixed code; the other 997 rows are carried over
    unchanged into the checkpoint as already-completed.
  - FILLS: combo never finished (850/1000). No rows are dropped — the
    existing 850 rows are frozen as-is into the checkpoint so the resume
    fills in exactly the missing 150 (which include the 3 contaminated IDs,
    generated correctly for the first time).

Usage:
    python scripts/prepare_patch_checkpoints.py             # print plan only
    python scripts/prepare_patch_checkpoints.py --apply      # write checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modelgen.infra.checkpoint import CheckpointManager  # noqa: E402

RUNS_DIR = ROOT / "runs"
CHECKPOINTS_DIR = ROOT / "checkpoints"

CONTAMINATED_IDS = ["79e8c959bbeb74a0", "ad6b5d46ae54842c", "c30e75b011696a95"]

# (run_id, model, method, benchmark) — combo already complete; drop the 3
# contaminated rows so they regenerate with the fixed prompt_builder.
PATCHES = [
    ("20260529_144439", "Qwen/Qwen2.5-7B-Instruct", "baseline", "arc_challenge"),
    ("20260529_144439", "Qwen/Qwen2.5-7B-Instruct", "calibration", "arc_challenge"),
    ("20260529_144439", "Qwen/Qwen2.5-7B-Instruct", "additional_option", "arc_challenge"),
    ("20260529_145812", "Qwen/Qwen2.5-7B-Instruct", "cyclic", "arc_challenge"),
    ("20260529_145812", "Qwen/Qwen2.5-7B-Instruct", "pride", "arc_challenge"),
    ("20260529_145812", "Qwen/Qwen2.5-7B-Instruct", "two_prompt", "arc_challenge"),
    ("20260529_180747", "meta-llama/Llama-3.1-8B-Instruct", "baseline", "arc_challenge"),
    ("20260529_180747", "meta-llama/Llama-3.1-8B-Instruct", "calibration", "arc_challenge"),
    ("20260529_180747", "meta-llama/Llama-3.1-8B-Instruct", "additional_option", "arc_challenge"),
    ("20260529_180748", "meta-llama/Llama-3.1-8B-Instruct", "cyclic", "arc_challenge"),
    ("20260529_180748", "meta-llama/Llama-3.1-8B-Instruct", "pride", "arc_challenge"),
    ("20260529_180748", "meta-llama/Llama-3.1-8B-Instruct", "two_prompt", "arc_challenge"),
    ("20260602_135654", "Qwen/Qwen2.5-32B-Instruct", "baseline", "arc_challenge"),
    ("20260602_135654", "Qwen/Qwen2.5-32B-Instruct", "calibration", "arc_challenge"),
    ("20260602_135654", "Qwen/Qwen2.5-32B-Instruct", "additional_option", "arc_challenge"),
    ("20260602_154857", "Qwen/Qwen2.5-32B-Instruct", "cyclic", "arc_challenge"),
    ("20260602_154857", "Qwen/Qwen2.5-32B-Instruct", "pride", "arc_challenge"),
    ("20260602_154857", "Qwen/Qwen2.5-32B-Instruct", "two_prompt", "arc_challenge"),
    ("20260608_080107", "Qwen/Qwen2.5-32B-Instruct", "text_extraction", "arc_challenge"),
    ("20260608_085259", "Qwen/Qwen2.5-32B-Instruct", "abcd", "arc_challenge"),
    ("20260602_010405", "Qwen/Qwen2.5-72B-Instruct", "baseline", "arc_challenge"),
    ("20260602_010405", "Qwen/Qwen2.5-72B-Instruct", "calibration", "arc_challenge"),
    ("20260602_010405", "Qwen/Qwen2.5-72B-Instruct", "additional_option", "arc_challenge"),
    ("20260602_125909", "Qwen/Qwen2.5-72B-Instruct", "cyclic", "arc_challenge"),
    ("20260602_125909", "Qwen/Qwen2.5-72B-Instruct", "pride", "arc_challenge"),
    ("20260602_125909", "Qwen/Qwen2.5-72B-Instruct", "two_prompt", "arc_challenge"),
    ("20260608_162341", "Qwen/Qwen2.5-72B-Instruct", "text_extraction", "arc_challenge"),
    ("20260608_180141", "Qwen/Qwen2.5-72B-Instruct", "abcd", "arc_challenge"),
    ("20260602_135956", "meta-llama/Llama-3.1-70B-Instruct", "baseline", "arc_challenge"),
    ("20260602_135956", "meta-llama/Llama-3.1-70B-Instruct", "calibration", "arc_challenge"),
    ("20260602_135956", "meta-llama/Llama-3.1-70B-Instruct", "additional_option", "arc_challenge"),
    ("20260602_162031", "meta-llama/Llama-3.1-70B-Instruct", "cyclic", "arc_challenge"),
    ("20260602_162031", "meta-llama/Llama-3.1-70B-Instruct", "pride", "arc_challenge"),
    ("20260602_162031", "meta-llama/Llama-3.1-70B-Instruct", "two_prompt", "arc_challenge"),
    ("20260608_193121", "meta-llama/Llama-3.1-70B-Instruct", "text_extraction", "arc_challenge"),
    ("20260608_200419", "meta-llama/Llama-3.1-70B-Instruct", "abcd", "arc_challenge"),
]

# (run_id, model, method, benchmark) — combo never finished (850/1000); no
# rows dropped, just freeze the existing partial CSV as the checkpoint.
FILLS = [
    ("20260617_162624", "Qwen/Qwen2.5-7B-Instruct", "abcd", "arc_challenge"),
    ("20260617_162624", "Qwen/Qwen2.5-7B-Instruct", "text_extraction", "arc_challenge"),
    ("20260617_162624", "meta-llama/Llama-3.1-8B-Instruct", "abcd", "arc_challenge"),
    ("20260617_162624", "meta-llama/Llama-3.1-8B-Instruct", "text_extraction", "arc_challenge"),
]


def _csv_path(run_id: str, method: str, model: str, benchmark: str) -> Path:
    safe_model = model.replace("/", "_")
    return RUNS_DIR / run_id / f"{run_id}_{method}_{safe_model}_{benchmark}.csv"


def _rows_to_jsonable(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to plain-Python dicts safe for json.dump (no numpy scalars)."""
    return json.loads(df.to_json(orient="records"))


def prepare_one(
    run_id: str,
    model: str,
    method: str,
    benchmark: str,
    drop_ids: list[str] | None,
    apply: bool,
) -> str:
    path = _csv_path(run_id, method, model, benchmark)
    if not path.exists():
        return f"SKIP  {run_id}  {method:<18} {model:<35} — CSV not found: {path}"

    df = pd.read_csv(path, low_memory=False)
    before = len(df)
    if drop_ids:
        df = df[~df["question_id"].isin(drop_ids)]
    after = len(df)

    msg = (
        f"{run_id}  {method:<18} {model:<35} "
        f"{before} -> {after} completed, {before - after} dropped"
    )

    if apply:
        mgr = CheckpointManager(
            checkpoint_dir=CHECKPOINTS_DIR,
            run_id=run_id,
            condition=method,
            model=model,
            benchmark=benchmark,
        )
        mgr.save(
            completed_ids=df["question_id"].tolist(),
            results=_rows_to_jsonable(df),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        msg += "  [checkpoint written]"

    return msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct checkpoints for the 3-ID patch / 150-ID fill operations."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write checkpoint files (default: print plan only)"
    )
    args = parser.parse_args()

    print("── 3-ID patch (drop contaminated rows, keep the rest) ──")
    for run_id, model, method, benchmark in PATCHES:
        print(" ", prepare_one(run_id, model, method, benchmark, CONTAMINATED_IDS, args.apply))

    print("\n── 150-ID fill (freeze existing partial CSV as-is) ──")
    for run_id, model, method, benchmark in FILLS:
        print(" ", prepare_one(run_id, model, method, benchmark, None, args.apply))

    if not args.apply:
        print("\n--apply not passed — no checkpoints written. Re-run with --apply to write them.")


if __name__ == "__main__":
    main()
