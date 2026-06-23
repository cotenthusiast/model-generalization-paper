"""Local model experiment runner.

Usage:
    python scripts/run_experiment.py [options]

Options:
    --config PATH     Path to YAML config (default: config/default.yaml)
    --run-id ID       Explicit run ID; timestamp auto-generated if omitted
    --dry-run         Print preflight estimate and exit without running
    --yes             Skip the preflight confirmation prompt
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modelgen.backends import HFCausalLMBackend, LocalGenerationConfig
from modelgen.backends.dummy import DummyBackend
from modelgen.infra.checkpoint import CheckpointManager
from modelgen.io.readers import read_normalized_questions, read_split_ids
from modelgen.io.writers import write_run_results
from modelgen.runners.abcd import ABCDRunner
from modelgen.runners.additional_option import AdditionalOptionRunner
from modelgen.runners.calibration import AnswerCalibrationRunner
from modelgen.runners.direct_mcq import DirectMCQRunner
from modelgen.runners.independent_hypothesis import IndependentHypothesisRunner
from modelgen.runners.permutation import PermutationRunner
from modelgen.runners.pride import PriDeRunner
from modelgen.runners.text_extraction import TextExtractionRunner
from modelgen.runners.two_stage import TwoStageRunner

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METHOD_TO_RUNNER = {
    "baseline": DirectMCQRunner,
    "two_prompt": TwoStageRunner,
    "cyclic": PermutationRunner,
    "pride": PriDeRunner,
    "calibration": AnswerCalibrationRunner,
    "additional_option": AdditionalOptionRunner,
    "text_extraction": TextExtractionRunner,
    "abcd": ABCDRunner,
    "independent_hypothesis": IndependentHypothesisRunner,
}

# Backend calls per question for each method (used for preflight estimates).
# ``pride`` is handled specially in ``preflight_estimate`` (calibration + inference).
_CALLS_PER_QUESTION = {
    "baseline": 1,
    "two_prompt": 2,           # stage 1 free-text + stage 2 matching
    "cyclic": 4,               # 4 cyclic permutations
    "pride": 1,                # 1 score_options call per eval question
    "calibration": 1,          # 1 score_options call per eval question (+ 1 setup call total)
    "additional_option": 1,
    "text_extraction": 1,      # stage 1 only; stage 2 is deterministic similarity matching
    "abcd": 1,                 # stage 1 only; stage 2 is deterministic similarity matching
    "independent_hypothesis": 4,  # 1 independent generate() call per option (usually 4)
}

# Maps (benchmark, split) to the artifact_group subdirectory used by read_split_ids.
_BENCHMARK_ARTIFACT_GROUP = {
    "mmlu": {
        "robustness": "benchmark",
        "review": "faithfulness",
    },
    "arc_challenge": {
        "robustness": "arc_challenge",
    },
}

# Maps benchmark name to its normalized CSV filename under data_processed_dir.
_BENCHMARK_NORMALIZED_FILE = {
    "mmlu": "mmlu_normalized.csv",
    "arc_challenge": "arc_challenge_normalized.csv",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_paths(paths_cfg: dict) -> dict[str, Path]:
    return {key: ROOT / val for key, val in paths_cfg.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_artifact_group(benchmark: str, split: str) -> str:
    benchmark_splits = _BENCHMARK_ARTIFACT_GROUP.get(benchmark)
    if benchmark_splits is None:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")
    group = benchmark_splits.get(split)
    if group is None:
        raise ValueError(f"Unknown split {split!r} for benchmark {benchmark!r}")
    return group


def load_questions(
    benchmark: str,
    split: str,
    paths: dict[str, Path],
    max_questions: int | None = None,
) -> list[dict]:
    if benchmark not in _BENCHMARK_NORMALIZED_FILE:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")
    normalized_file = _BENCHMARK_NORMALIZED_FILE[benchmark]
    artifact_group = _resolve_artifact_group(benchmark, split)
    df = read_normalized_questions(normalized_file, paths["data_processed_dir"])
    split_ids = read_split_ids(split, paths["data_splits_dir"], artifact_group)
    df = df[df["question_id"].isin(split_ids)].drop_duplicates(subset="question_id")
    if max_questions is not None:
        df = df.head(max_questions)
    return df.to_dict(orient="records")


def load_calibration_questions(
    benchmark: str,
    eval_question_ids: set[str],
    paths: dict[str, Path],
) -> list[dict]:
    """Return questions from the full normalized CSV that are NOT in the eval split."""
    if benchmark not in _BENCHMARK_NORMALIZED_FILE:
        logger.warning(
            "Unknown benchmark %r — returning empty PriDe calibration pool", benchmark
        )
        return []
    normalized_file = _BENCHMARK_NORMALIZED_FILE[benchmark]
    try:
        df = read_normalized_questions(normalized_file, paths["data_processed_dir"])
    except FileNotFoundError as exc:
        logger.warning("Cannot load PriDe calibration questions: %s", exc)
        return []
    df = df[~df["question_id"].isin(eval_question_ids)].drop_duplicates(
        subset="question_id"
    )
    candidates = df.to_dict(orient="records")
    logger.info(
        "PriDe calibration pool for %s: %d questions (%d eval excluded)",
        benchmark,
        len(candidates),
        len(eval_question_ids),
    )
    return candidates


def count_split_questions(benchmark: str, split: str, paths: dict[str, Path]) -> int:
    artifact_group = _resolve_artifact_group(benchmark, split)
    return len(read_split_ids(split, paths["data_splits_dir"], artifact_group))


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def build_backend(model_name: str, model_cfg: dict) -> HFCausalLMBackend | DummyBackend:
    """Construct a local backend from a model config dict."""
    if model_cfg.get("family") == "dummy":
        return DummyBackend()
    return HFCausalLMBackend(
        model_path=model_cfg.get("model_path", model_name),
        family=model_cfg.get("family", "unknown"),
        size_label=model_cfg.get("size_label"),
        device=model_cfg.get("device"),
    )


# ---------------------------------------------------------------------------
# Preflight estimation
# ---------------------------------------------------------------------------

def preflight_estimate(config: dict, paths: dict[str, Path]) -> None:
    """Print a call count and wall-clock estimate without running."""
    run_cfg = config["run"]
    models_cfg = config["models"]
    jobs = run_cfg["jobs"]

    print("\n── Preflight estimate ─────────────────────────────────────────")

    total_calls = 0

    for job in jobs:
        model_name = job["model"]
        methods = job["methods"]
        benchmark = job["benchmark"]
        split = job["split"]

        try:
            n_questions = count_split_questions(benchmark, split, paths)
        except Exception:
            n_questions = 1000
            print(f"  (could not read split file for {benchmark}/{split} — using 1000)")

        max_q = run_cfg.get("max_questions") or None
        if max_q is not None:
            n_questions = min(n_questions, int(max_q))

        job_calls = 0
        for m in methods:
            if m == "pride":
                calib_k = int(run_cfg.get("pride_calibration_n", 50))
                job_calls += n_questions + 4 * calib_k
            else:
                job_calls += _CALLS_PER_QUESTION.get(m, 1) * n_questions

        total_calls += job_calls
        print(f"  {model_name:<40}  {job_calls:>6} calls")

    print(f"  {'─'*60}")
    print(f"  {'Total':<40}  {total_calls:>6} calls")
    print(f"  Checkpoint every: {run_cfg.get('checkpoint_every_n', 50)} questions")
    print("───────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def run_single_method(
    backend: HFCausalLMBackend,
    model_name: str,
    method: str,
    benchmark: str,
    split: str,
    questions: list[dict],
    run_id: str,
    run_cfg: dict,
    paths: dict[str, Path],
) -> dict:
    """Run one (model, method, benchmark, split) job with checkpointing."""
    runner_cls = _METHOD_TO_RUNNER[method]
    gen_config = LocalGenerationConfig(
        max_new_tokens=run_cfg.get("max_tokens", 500),
        temperature=run_cfg.get("temperature", 0.0),
        seed=run_cfg.get("seed", 42),
    )
    common_kw = dict(
        backend=backend,
        method_name=method,
        split_name=split,
        prompt_version=run_cfg.get("prompt_version", "v1"),
        prompts_dir=paths["prompts_dir"],
        run_id=run_id,
        generation_config=gen_config,
    )
    if method == "pride":
        eval_ids = {q["question_id"] for q in questions}
        calib_candidates = load_calibration_questions(benchmark, eval_ids, paths)
        runner = runner_cls(
            **common_kw,
            calibration_n=int(run_cfg.get("pride_calibration_n", 50)),
            calibration_seed=int(run_cfg.get("pride_calibration_seed", 42)),
            calibration_benchmark=benchmark,
            calibration_runs_dir=paths["runs_dir"],
            calibration_questions=calib_candidates,
        )
    elif method == "two_prompt":
        runner = runner_cls(
            **common_kw,
            fallback_on_parse_failure=run_cfg.get("fallback_on_parse_failure", False),
        )
    elif method in ("text_extraction", "abcd"):
        runner = runner_cls(
            **common_kw,
            similarity_threshold=run_cfg.get("similarity_threshold", 0.1),
        )
    else:
        runner = runner_cls(**common_kw)

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=paths["checkpoints_dir"],
        run_id=run_id,
        condition=method,
        model=model_name,
        benchmark=benchmark,
    )

    state = checkpoint_mgr.load()
    completed_ids: list[str] = state["completed_ids"] if state else []
    accumulated: list[dict] = state["results"] if state else []
    started_at: str = (
        state["started_at"] if state else datetime.now(timezone.utc).isoformat()
    )
    for row in accumulated:
        row.setdefault("benchmark", benchmark)

    completed_set = set(completed_ids)
    remaining = [q for q in questions if q["question_id"] not in completed_set]

    tag = f"[{model_name}] {method}"

    if not remaining:
        logger.info("%s  already complete (%d questions) — writing CSV", tag, len(accumulated))
    else:
        logger.info(
            "%s  starting: %d remaining / %d total",
            tag, len(remaining), len(questions),
        )

        n = run_cfg.get("checkpoint_every_n", 50)
        for batch_start in range(0, len(remaining), n):
            batch = remaining[batch_start : batch_start + n]
            try:
                batch_results = runner.run_many(batch)
            except Exception as exc:
                logger.error("%s  batch failed: %s — continuing", tag, exc)
                continue

            for row in batch_results:
                row["benchmark"] = benchmark
            accumulated.extend(batch_results)
            completed_ids.extend(r["question_id"] for r in batch_results)
            checkpoint_mgr.save(completed_ids, accumulated, started_at)
            logger.info("%s  %d/%d questions complete", tag, len(accumulated), len(questions))

    output_path = write_run_results(
        results=accumulated,
        output_dir=paths["runs_dir"] / run_id,
        run_id=run_id,
        method_name=method,
        model_name=model_name,
        benchmark=benchmark,
    )
    checkpoint_mgr.delete()
    logger.info("%s  done → %s", tag, output_path)

    n_success = sum(1 for r in accumulated if r.get("model_status") == "success")
    return {
        "model": model_name,
        "method": method,
        "benchmark": benchmark,
        "total": len(accumulated),
        "success": n_success,
        "output": str(output_path),
    }


def run_model_jobs(
    model_name: str,
    model_cfg: dict,
    jobs_for_model: list[dict],
    questions_cache: dict[tuple, list[dict]],
    run_id: str,
    run_cfg: dict,
    paths: dict[str, Path],
) -> list[dict]:
    """Load one backend for a model and run all its jobs sequentially."""
    try:
        backend = build_backend(model_name, model_cfg)
        backend.load()
    except Exception as exc:
        logger.error("[%s] Failed to load backend: %s — skipping all jobs", model_name, exc)
        return []

    summaries = []
    for job in jobs_for_model:
        for method in job["methods"]:
            key = (job["benchmark"], job["split"])
            questions = questions_cache[key]
            try:
                summary = run_single_method(
                    backend=backend,
                    model_name=model_name,
                    method=method,
                    benchmark=job["benchmark"],
                    split=job["split"],
                    questions=questions,
                    run_id=run_id,
                    run_cfg=run_cfg,
                    paths=paths,
                )
                summaries.append(summary)
            except Exception as exc:
                logger.error(
                    "[%s] %s failed: %s — continuing with remaining methods",
                    model_name, method, exc,
                )
                summaries.append(
                    {"model": model_name, "method": method, "error": str(exc)}
                )

    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local model experiment runner for two-prompt research."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID string; auto-generates a timestamp if omitted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print preflight estimate and exit without making any backend calls",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the preflight confirmation prompt (for automated/SLURM runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config["paths"])
    run_cfg = config["run"]
    models_cfg = config["models"]
    jobs = run_cfg["jobs"]

    preflight_estimate(config, paths)

    if args.dry_run:
        print("--dry-run specified. Exiting.")
        return

    if not args.yes:
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("Run ID: %s", run_id)
    logger.info("Config: %s", args.config)

    run_dir = paths["runs_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    prompt_version = run_cfg.get("prompt_version", "v1")
    prompt_src = paths["prompts_dir"] / prompt_version
    prompt_dst = run_dir / "prompts" / prompt_version
    if prompt_src.exists() and not prompt_dst.exists():
        shutil.copytree(prompt_src, prompt_dst, dirs_exist_ok=True)
    logger.info("Snapshotted config and prompts/%s → %s", prompt_version, run_dir)

    max_questions = run_cfg.get("max_questions") or None

    questions_cache: dict[tuple, list[dict]] = {}
    for job in jobs:
        key = (job["benchmark"], job["split"])
        if key not in questions_cache:
            logger.info("Loading questions: benchmark=%s split=%s", *key)
            questions_cache[key] = load_questions(
                job["benchmark"], job["split"], paths, max_questions=max_questions
            )
            logger.info("  %d questions loaded", len(questions_cache[key]))

    jobs_by_model: dict[str, list[dict]] = {}
    for job in jobs:
        jobs_by_model.setdefault(job["model"], []).append(job)

    # Models run sequentially — only one set of weights in GPU memory at a time.
    all_summaries: list[list[dict]] = []
    for model_name, model_jobs in jobs_by_model.items():
        summaries = run_model_jobs(
            model_name=model_name,
            model_cfg=models_cfg[model_name],
            jobs_for_model=model_jobs,
            questions_cache=questions_cache,
            run_id=run_id,
            run_cfg=run_cfg,
            paths=paths,
        )
        all_summaries.append(summaries)

    flat = [s for per_model in all_summaries for s in per_model]
    print("\n── Run complete ───────────────────────────────────────────────")
    print(f"  Run ID: {run_id}")
    print(f"  Results in: {paths['runs_dir'] / run_id}/")
    print()
    for s in flat:
        if "error" in s:
            print(f"  FAILED  {s['model']:<40}  {s['method']:<22}  {s['error']}")
        else:
            pct = f"{100 * s['success'] / s['total']:.0f}%" if s["total"] else "—"
            print(
                f"  OK      {s['model']:<40}  {s['method']:<22}"
                f"  {s['success']}/{s['total']} success ({pct})"
            )
    print("───────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
