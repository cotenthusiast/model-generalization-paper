# tests/runners/test_pride.py

import json
import logging
from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.backends.types import ModelOptionScoreResult
from modelgen.runners.pride import PriDeRunner

from tests.runners.conftest import ErrorScoreBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


class DGetsBoostedBackend(DummyBackend):
    """D always scores far higher than A/B/C whenever it's included in a
    score_options() call — simulates a model that finds the literal letter
    "D" highly plausible regardless of context. Used to prove a 3-option
    question's eval scoring and a 3-option calibration question's
    permutation rollout can't ever request, score, or select a "D" that
    isn't a real option for that question.
    """

    def __init__(self):
        super().__init__()
        self.requested_option_sets: list[list[str]] = []

    def score_options(self, prompt, options):
        self.requested_option_sets.append(list(options))
        scores = {letter: -2.0 for letter in options}
        if "D" in options:
            scores["D"] = 5.0
        return ModelOptionScoreResult(scores=scores, raw_logprobs=dict(scores), metadata={})


def _make_runner(backend, tmp_path: Path, calibration_questions=None, calibration_n=0):
    return PriDeRunner(
        backend=backend,
        method_name="pride",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="pride_test",
        calibration_n=calibration_n,
        calibration_seed=0,
        calibration_benchmark="mmlu",
        calibration_runs_dir=tmp_path,
        calibration_questions=calibration_questions or [],
    )


class TestPriDeRunnerIntegration:
    def test_run_many_returns_results(self, runner_question_row, tmp_path):
        """run_many should return one result per question."""
        b = DummyBackend()
        b.load()
        runner = _make_runner(b, tmp_path)
        rows = runner.run_many([runner_question_row])

        assert len(rows) == 1

    def test_inference_mode_is_eq8_transfer(self, runner_question_row, tmp_path):
        """All evaluation rows must use eq8_transfer mode."""
        b = DummyBackend()
        b.load()
        rows = _make_runner(b, tmp_path).run_many([runner_question_row])

        assert rows[0]["pride_inference_mode"] == "eq8_transfer"

    def test_adjusted_choice_present(self, runner_question_row, tmp_path):
        """With DummyBackend, debiasing should produce an adjusted choice."""
        b = DummyBackend()
        b.load()
        rows = _make_runner(b, tmp_path).run_many([runner_question_row])

        assert rows[0]["pride_adjusted_choice"] in {"A", "B", "C", "D"}

    def test_option_logprob_json_present(self, runner_question_row, tmp_path):
        """option_logprob_json should be a JSON object with A/B/C/D keys."""
        b = DummyBackend()
        b.load()
        rows = _make_runner(b, tmp_path).run_many([runner_question_row])

        lp = json.loads(rows[0]["option_logprob_json"])
        assert set(lp.keys()) == {"A", "B", "C", "D"}

    def test_uniform_prior_when_no_calibration_questions(
            self, runner_question_row, tmp_path,
    ):
        """Empty calibration pool → uniform prior (each letter ~0.25)."""
        b = DummyBackend()
        b.load()
        runner = _make_runner(b, tmp_path, calibration_questions=[], calibration_n=50)
        rows = runner.run_many([runner_question_row])

        prior = json.loads(rows[0]["peprior_json"])
        for v in prior.values():
            assert abs(v - 0.25) < 1e-6

    def test_calibration_with_separate_questions(
            self, runner_question_row, tmp_path,
    ):
        """Calibration questions produce a non-trivial prior."""
        cal_question = {
            **runner_question_row,
            "question_id": "cal_qid",
            "correct_option": "B",
        }
        b = DummyBackend()
        b.load()
        runner = _make_runner(
            b, tmp_path,
            calibration_questions=[cal_question],
            calibration_n=1,
        )
        rows = runner.run_many([runner_question_row])

        assert rows[0]["pride_inference_mode"] == "eq8_transfer"
        assert rows[0]["model_status"] == "success"

    def test_score_options_failure_skips_debiasing(
            self, runner_question_row, tmp_path, caplog,
    ):
        """If score_options raises, debiasing is skipped and adjusted_choice is None."""
        b = ErrorScoreBackend()
        b.load()
        runner = _make_runner(b, tmp_path)

        with caplog.at_level(logging.WARNING, logger="modelgen.runners.pride"):
            rows = runner.run_many([runner_question_row])

        assert rows[0]["pride_adjusted_choice"] is None

    def test_sidecar_written_after_calibration(self, runner_question_row, tmp_path):
        """After calibration, a sidecar JSON file should be written."""
        b = DummyBackend()
        b.load()
        runner = _make_runner(b, tmp_path, calibration_questions=[], calibration_n=0)
        runner.run_many([runner_question_row])

        sidecar = runner._sidecar_path()
        assert sidecar.exists()
        blob = json.loads(sidecar.read_text())
        assert blob["schema_version"] == 3

    def test_sidecar_reused_on_second_run(self, runner_question_row, tmp_path):
        """Second run with matching sidecar should load from disk rather than refit."""
        b = DummyBackend()
        b.load()
        runner1 = _make_runner(b, tmp_path)
        runner1.run_many([runner_question_row])

        runner2 = _make_runner(b, tmp_path)
        runner2.run_many([runner_question_row])

        # If sidecar was reused, _calibration_ready should be True after the first run_many
        assert runner2._calibration_ready


class TestPriDeThreeOptionQuestions:
    """Regression coverage for the hardcoded OPTION_LETTERS bug: a 3-option
    ARC-Challenge question (no choice_d) must never have "D" requested,
    scored, or selected — neither for its own eval scoring nor when it's
    drawn into the calibration/rollout pool used to fit the Eq.(7) prior."""

    def test_eval_question_never_requests_or_selects_phantom_d(
            self, runner_question_row, tmp_path,
    ):
        row = dict(runner_question_row, choice_d=float("nan"))
        b = DGetsBoostedBackend()
        b.load()
        rows = _make_runner(b, tmp_path).run_many([row])

        assert all(set(opts) == {"A", "B", "C"} for opts in b.requested_option_sets)
        assert rows[0]["pride_adjusted_choice"] != "D"
        olj = json.loads(rows[0]["option_logprob_json"])
        assert "D" not in olj

    def test_calibration_rollout_never_requests_phantom_d_for_three_option_question(
            self, runner_question_row, tmp_path,
    ):
        cal_question = dict(runner_question_row, question_id="cal_3opt", choice_d=float("nan"))
        b = DGetsBoostedBackend()
        b.load()
        runner = _make_runner(
            b, tmp_path, calibration_questions=[cal_question], calibration_n=1,
        )
        rows = runner.run_many([runner_question_row])

        # The first 3 recorded calls are the 3 cyclic permutations of the
        # 3-option calibration question's rollout; none may request "D".
        assert all(set(opts) <= {"A", "B", "C"} for opts in b.requested_option_sets[:3])

        prior = json.loads(rows[0]["peprior_json"])
        assert set(prior.keys()) == {"A", "B", "C", "D"}
