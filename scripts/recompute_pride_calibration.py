"""One-off fix: recompute the PriDe ARC-Challenge calibration prior that was
contaminated by the hardcoded-OPTION_LETTERS bug.

Background: one of the 50 seeded calibration questions (question_id
d7b41517201d67a6) is a 3-option ARC-Challenge item with no real D choice.
The pre-fix PriDeRunner._cyclic_rollout_prob_matrix scored a literal "D"
against that question anyway, feeding a phantom (real-but-meaningless)
logprob into the Eq.(7) prior average used for every eval question across
the whole run. src/modelgen/runners/pride.py and pride_debias.py are now
fixed to restrict scoring to each question's real options.

This script reuses the EXACT same 50-question calibration set already
recorded in the existing sidecar (same seed=42 IDs) -- no resampling, only
the scoring logic for those questions changes -- recomputes the prior with
the corrected code, overwrites the sidecar, then re-derives
pride_adjusted_choice (and dependent is_correct/parsed_choice/score_status
columns) for every row in the model's PriDe ARC-Challenge run CSV using the
corrected prior. Requires the model loaded for ~200 cheap score_options()
calls (single forward pass each, no generation) -- meant to be bundled into
the same SLURM job that's already loading the model for other work.

Usage:
    python scripts/recompute_pride_calibration.py \\
        --config config/llama70b_arc_expensive.yaml \\
        --model-name meta-llama/Llama-3.1-70B-Instruct \\
        --apply

Without --apply, runs in dry-run mode: prints the diff but writes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from modelgen.backends import HFCausalLMBackend
from modelgen.backends.dummy import DummyBackend
from modelgen.io.readers import read_normalized_questions
from modelgen.parsing.types import PARSE_OK, ParseResult
from modelgen.runners.pride import PriDeRunner
from modelgen.runners.pride_debias import apply_debiased_choice_from_defaults
from modelgen.scoring.scorer import score_prediction

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sidecar_slug(model_path: str) -> str:
    return model_path.replace("/", "_").replace(" ", "_").replace(":", "_")


def find_sidecar(runs_dir: Path, model_path: str) -> Path:
    slug = _sidecar_slug(model_path)
    matches = sorted(runs_dir.glob(f"*/pride_calibration__{slug}__arc_challenge.json"))
    if not matches:
        raise FileNotFoundError(
            f"No PriDe ARC-Challenge calibration sidecar found for {model_path!r} under {runs_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple sidecars found for {model_path!r}, expected exactly one: {matches}"
        )
    return matches[0]


def find_eval_csv(runs_dir: Path, model_path: str) -> Path:
    slug = _sidecar_slug(model_path)
    matches = sorted(runs_dir.glob(f"*/*_pride_{slug}_arc_challenge.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No PriDe ARC-Challenge eval CSV found for {model_path!r} under {runs_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple eval CSVs found for {model_path!r}, expected exactly one: {matches}"
        )
    return matches[0]


def real_letters_for_row(row: dict) -> list[str]:
    letters = []
    for letter, col in (("A", "choice_a"), ("B", "choice_b"), ("C", "choice_c"), ("D", "choice_d")):
        v = row.get(col)
        if v is not None and str(v).strip().lower() not in ("", "nan"):
            letters.append(letter)
    return letters


def recompute_prior(
        backend,
        sidecar_path: Path,
        data_processed_dir: Path,
        prompts_dir: Path,
) -> tuple[dict, list[str]]:
    """Recompute the Eq.(7) prior over the same calibration question set
    recorded in the existing sidecar, using the corrected scoring logic.

    Returns (corrected_peprior_probs, calibration_question_ids).
    """
    old_sidecar = json.loads(sidecar_path.read_text())
    cal_ids = old_sidecar["calibration_question_ids"]
    cal_seed = int(old_sidecar["calibration_seed"])

    df = read_normalized_questions("arc_challenge_normalized.csv", data_processed_dir)
    by_id = {row["question_id"]: row for row in df.to_dict(orient="records")}
    missing = [qid for qid in cal_ids if qid not in by_id]
    if missing:
        raise RuntimeError(f"Calibration question ids not found in dataset: {missing}")
    cal_rows = [by_id[qid] for qid in cal_ids]

    runner = PriDeRunner(
        backend=backend,
        method_name="pride",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=prompts_dir,
        run_id="pride_calibration_recompute",
        calibration_n=len(cal_rows),
        calibration_seed=cal_seed,
        calibration_benchmark="arc_challenge",
        calibration_questions=cal_rows,
    )
    # Bypass the sidecar-reuse check in _ensure_calibration — we explicitly
    # want a fresh fit from the corrected code, not a reload of the old one.
    corrected_state = runner._fit_calibration_prior(cal_rows)

    new_sidecar = dict(old_sidecar)
    new_sidecar["peprior_probs"] = {
        L: float(corrected_state.peprior_probs.get(L, 0.0)) for L in ("A", "B", "C", "D")
    }
    return new_sidecar, cal_ids


def rescore_eval_csv(eval_csv_path: Path, corrected_peprior_probs: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Re-derive pride_adjusted_choice for every row using the corrected prior.

    Returns (all_rows, changed_rows_before_after, fieldnames).
    """
    import csv

    with eval_csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    from modelgen.runners.pride_debias import CalibrationState
    state = CalibrationState(peprior_probs=corrected_peprior_probs)

    changed = []
    for row in rows:
        real_letters = tuple(real_letters_for_row(row))
        raw_scores_all = json.loads(row["option_logprob_json"])
        # Drop any phantom letter the old buggy code scored that isn't a
        # real option for this question (only affects the 3 known 3-option
        # eval rows; for every other row this is a no-op).
        raw_scores = {L: raw_scores_all[L] for L in real_letters if L in raw_scores_all}

        old_choice = row["pride_adjusted_choice"]
        new_choice = apply_debiased_choice_from_defaults(state, raw_scores, letters=real_letters)

        if new_choice != old_choice:
            parse_result = ParseResult(
                final_choice=new_choice,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=new_choice,
                reason="pride_eq8_recomputed",
            )
            score_result = score_prediction(parse_result, row["correct_option"])
            changed.append({
                "question_id": row["question_id"],
                "old_choice": old_choice,
                "new_choice": new_choice,
                "correct_option": row["correct_option"],
            })
            row["pride_adjusted_choice"] = new_choice
            row["parsed_choice"] = score_result.predicted_choice
            row["is_correct"] = score_result.is_correct
            row["score_status"] = score_result.status
            row["peprior_json"] = json.dumps(corrected_peprior_probs)
            row["option_logprob_json"] = json.dumps(raw_scores)

    return rows, changed, fieldnames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--apply", action="store_true", help="Write changes; default is dry-run.")
    args = parser.parse_args()

    with args.config.open() as f:
        config = yaml.safe_load(f)

    model_cfg = config["models"][args.model_name]
    paths_cfg = config["paths"]
    paths = {k: ROOT / v for k, v in paths_cfg.items()}

    if model_cfg.get("family") == "dummy":
        backend = DummyBackend()
    else:
        backend = HFCausalLMBackend(
            model_path=model_cfg.get("model_path", args.model_name),
            family=model_cfg.get("family", "unknown"),
            size_label=model_cfg.get("size_label"),
            device=model_cfg.get("device"),
        )
    logger.info("Loading model %s ...", args.model_name)
    backend.load()
    logger.info("Model loaded.")

    sidecar_path = find_sidecar(paths["runs_dir"], args.model_name)
    eval_csv_path = find_eval_csv(paths["runs_dir"], args.model_name)
    logger.info("Sidecar: %s", sidecar_path)
    logger.info("Eval CSV: %s", eval_csv_path)

    new_sidecar, cal_ids = recompute_prior(
        backend=backend,
        sidecar_path=sidecar_path,
        data_processed_dir=paths["data_processed_dir"],
        prompts_dir=paths["prompts_dir"],
    )

    old_sidecar = json.loads(sidecar_path.read_text())
    print(f"\n=== Prior comparison for {args.model_name} ===")
    print(f"  calibration questions (n={len(cal_ids)}, unchanged set, seed={old_sidecar['calibration_seed']})")
    for L in ("A", "B", "C", "D"):
        old_v = old_sidecar["peprior_probs"].get(L, 0.0)
        new_v = new_sidecar["peprior_probs"].get(L, 0.0)
        print(f"    {L}: old={old_v:.4f}  new={new_v:.4f}  delta={new_v - old_v:+.4f}")

    rows, changed, fieldnames = rescore_eval_csv(eval_csv_path, new_sidecar["peprior_probs"])
    print(f"\n=== Eval row re-score for {args.model_name} ===")
    print(f"  rows scanned: {len(rows)}  rows changed: {len(changed)}")
    for c in changed:
        print(f"    CHANGED qid={c['question_id']} old={c['old_choice']} new={c['new_choice']} correct={c['correct_option']}")

    if not args.apply:
        print("\nDry-run only (pass --apply to write sidecar + CSV changes).")
        return

    sidecar_path.write_text(json.dumps(new_sidecar, indent=2))
    logger.info("Wrote corrected sidecar: %s", sidecar_path)

    if changed:
        import csv
        with eval_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote corrected eval CSV: %s (%d rows changed)", eval_csv_path, len(changed))
    else:
        logger.info("No eval rows changed — CSV left untouched.")


if __name__ == "__main__":
    main()
