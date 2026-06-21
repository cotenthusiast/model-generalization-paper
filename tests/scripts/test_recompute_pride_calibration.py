"""Integration test for scripts/recompute_pride_calibration.py — exercises
the full sidecar-recompute + eval-CSV-rescore plumbing against a DummyBackend
fixture (no real model needed). Real-model runs on Kelvin2 only swap the
backend; this test covers the script's logic end to end."""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import recompute_pride_calibration as script  # noqa: E402


@pytest.fixture
def fixture_paths(tmp_path):
    """Build a minimal runs_dir/data_processed_dir tree mirroring the real
    layout: one 4-option and one 3-option calibration question, one
    3-option eval question whose saved row carries a phantom "D" score
    (simulating the pre-fix bug's output) and currently has D as the chosen
    answer — the recompute should correct it."""
    data_processed_dir = tmp_path / "data" / "processed"
    data_processed_dir.mkdir(parents=True)
    runs_dir = tmp_path / "runs"
    run_subdir = runs_dir / "testrun"
    run_subdir.mkdir(parents=True)

    with (data_processed_dir / "arc_challenge_normalized.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id", "subject", "question_text",
                "choice_a", "choice_b", "choice_c", "choice_d",
                "correct_option", "correct_answer_text",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "question_id": "cal4opt", "subject": "x", "question_text": "Q1?",
            "choice_a": "a1", "choice_b": "b1", "choice_c": "c1", "choice_d": "d1",
            "correct_option": "A", "correct_answer_text": "a1",
        })
        writer.writerow({
            "question_id": "cal3opt", "subject": "x", "question_text": "Q2?",
            "choice_a": "a2", "choice_b": "b2", "choice_c": "c2", "choice_d": "",
            "correct_option": "B", "correct_answer_text": "b2",
        })

    model_path = "dummy-test-model"
    slug = script._sidecar_slug(model_path)

    old_sidecar = {
        "schema_version": 3,
        "version": "v2-pride-iclr2024",
        "calibration_seed": 42,
        "n_options": 4,
        "calibration_question_ids": ["cal4opt", "cal3opt"],
        "peprior_probs": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
        "epsilon": 1e-12,
    }
    sidecar_path = run_subdir / f"pride_calibration__{slug}__arc_challenge.json"
    sidecar_path.write_text(json.dumps(old_sidecar))

    eval_csv_path = run_subdir / f"testrun_pride_{slug}_arc_challenge.csv"
    fieldnames = [
        "question_id", "choice_a", "choice_b", "choice_c", "choice_d",
        "correct_option", "pride_adjusted_choice", "parsed_choice",
        "is_correct", "score_status", "peprior_json", "option_logprob_json",
    ]
    with eval_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # D would win the old (buggy) unrestricted argmax with this prior +
        # scores; the question has no real D option (no choice_d).
        writer.writerow({
            "question_id": "eval3opt",
            "choice_a": "ea", "choice_b": "eb", "choice_c": "ec", "choice_d": "",
            "correct_option": "A",
            "pride_adjusted_choice": "D",
            "parsed_choice": "D",
            "is_correct": "False",
            "score_status": "scored",
            "peprior_json": json.dumps(old_sidecar["peprior_probs"]),
            "option_logprob_json": json.dumps({"A": -2.0, "B": -2.0, "C": -2.0, "D": 5.0}),
        })

    return {
        "data_processed_dir": data_processed_dir,
        "runs_dir": runs_dir,
        "prompts_dir": REPO_ROOT / "prompts",
        "model_path": model_path,
        "sidecar_path": sidecar_path,
        "eval_csv_path": eval_csv_path,
    }


class DummyBackend:
    """Minimal backend matching the real script's expectations: scores
    whatever options it's asked about with a fixed, deterministic value."""

    def load(self):
        pass

    def score_options(self, prompt, options):
        from modelgen.backends.types import ModelOptionScoreResult
        scores = {opt: -1.0 for opt in options}
        return ModelOptionScoreResult(scores=scores, raw_logprobs=dict(scores), metadata={})


def test_find_sidecar_and_eval_csv(fixture_paths):
    sidecar = script.find_sidecar(fixture_paths["runs_dir"], fixture_paths["model_path"])
    eval_csv = script.find_eval_csv(fixture_paths["runs_dir"], fixture_paths["model_path"])

    assert sidecar == fixture_paths["sidecar_path"]
    assert eval_csv == fixture_paths["eval_csv_path"]


def test_recompute_prior_reuses_same_calibration_ids(fixture_paths):
    backend = DummyBackend()
    backend.load()

    new_sidecar, cal_ids = script.recompute_prior(
        backend=backend,
        sidecar_path=fixture_paths["sidecar_path"],
        data_processed_dir=fixture_paths["data_processed_dir"],
        prompts_dir=fixture_paths["prompts_dir"],
    )

    assert set(cal_ids) == {"cal4opt", "cal3opt"}
    assert set(new_sidecar["peprior_probs"].keys()) == {"A", "B", "C", "D"}
    assert new_sidecar["calibration_question_ids"] == ["cal4opt", "cal3opt"]


def test_rescore_eval_csv_corrects_phantom_d_choice(fixture_paths):
    # A prior where D dominates (mirrors a contaminated prior) — Eq.(8)
    # divides by the prior, so a large D prior suppresses D's debiased
    # probability. Combined with restricting the argmax to real options,
    # the eval question (no real D) must never end up choosing "D".
    corrected_peprior = {"A": 0.2, "B": 0.3, "C": 0.2, "D": 0.3}

    rows, changed, fieldnames = script.rescore_eval_csv(
        fixture_paths["eval_csv_path"], corrected_peprior,
    )

    assert len(rows) == 1
    assert rows[0]["pride_adjusted_choice"] != "D"
    assert len(changed) == 1
    assert changed[0]["old_choice"] == "D"
    assert changed[0]["new_choice"] != "D"
    # The phantom D score must be dropped from the patched row's stored scores.
    assert "D" not in json.loads(rows[0]["option_logprob_json"])


def test_rescore_is_idempotent_dry_run_does_not_mutate_file(fixture_paths):
    before = fixture_paths["eval_csv_path"].read_text()
    script.rescore_eval_csv(fixture_paths["eval_csv_path"], {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    after = fixture_paths["eval_csv_path"].read_text()

    assert before == after
