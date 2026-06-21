# tests/runners/test_calibration.py

import json
import logging
from pathlib import Path

import pytest

from modelgen.backends.dummy import DummyBackend
from modelgen.backends.types import ModelOptionScoreResult
from modelgen.runners.calibration import AnswerCalibrationRunner

from tests.runners.conftest import ErrorScoreBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


class HighDScoreBackend(DummyBackend):
    """D scores far higher than the real options whenever it's included in a
    score_options() call against a real (non-neutral) prompt, but is not
    boosted in the neutral calibration-prior prompt (_build_neutral_prompt
    always uses question="N/A"). Simulates a model whose per-question logprob
    for a never-shown "D" happens to be high even though the prior phase
    never measured that elevated score — proof that a 3-option question
    can't let D win just because the old hardcoded-ABCD code would have
    asked for it and let it through.
    """

    def __init__(self):
        super().__init__()
        self.requested_option_sets: list[list[str]] = []

    def score_options(self, prompt, options):
        self.requested_option_sets.append(list(options))
        is_neutral = "N/A" in prompt
        scores = {letter: -2.0 for letter in options}
        if "D" in options and not is_neutral:
            scores["D"] = 5.0
        return ModelOptionScoreResult(scores=scores, raw_logprobs=dict(scores), metadata={})


def _make_runner(backend):
    return AnswerCalibrationRunner(
        backend=backend,
        method_name="calibration",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="calibration_test",
    )


class TestAnswerCalibrationRunnerIntegration:
    def test_run_many_returns_one_result_per_question(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        assert len(rows) == 1

    def test_model_status_is_success(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        assert rows[0]["model_status"] == "success"

    def test_calibration_adjusted_choice_is_valid_letter(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        assert rows[0]["calibration_adjusted_choice"] in {"A", "B", "C", "D"}

    def test_option_logprob_json_has_abcd_keys(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        lp = json.loads(rows[0]["option_logprob_json"])
        assert set(lp.keys()) == {"A", "B", "C", "D"}

    def test_prior_logprob_json_has_abcd_keys(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        prior = json.loads(rows[0]["prior_logprob_json"])
        assert set(prior.keys()) == {"A", "B", "C", "D"}

    def test_calibration_ready_after_run(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        runner.run_many([runner_question_row])

        assert runner._calibration_ready is True

    def test_prior_stored_as_attribute_after_setup(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        runner.setup([runner_question_row])

        assert hasattr(runner, "_prior")
        assert set(runner._prior.keys()) == {"A", "B", "C", "D"}

    def test_score_options_failure_returns_error_row(self, runner_question_row, caplog):
        b = ErrorScoreBackend()
        b.load()
        runner = _make_runner(b)

        with caplog.at_level(logging.WARNING, logger="modelgen.runners.calibration"):
            rows = runner.run_many([runner_question_row])

        assert rows[0]["calibration_adjusted_choice"] is None
        assert rows[0]["model_status"] == "error"

    def test_multiple_questions_all_get_results(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row, runner_question_row])

        assert len(rows) == 2
        assert all(r["model_status"] == "success" for r in rows)

    def test_missing_fourth_option_has_no_phantom_choice_in_prompt(self, runner_question_row):
        """A 3-option question must not show a phantom D in the rendered prompt."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([row])

        assert "nan" not in rows[0]["prompt"].lower()
        assert "D." not in rows[0]["prompt"]
        assert rows[0]["model_status"] == "success"

    def test_three_option_question_never_scores_or_selects_phantom_d(
            self, runner_question_row,
    ):
        """A 3-option question (no choice_d) must never request, score, or
        select "D" — even when the backend would make D win if it were ever
        considered. Regression test for the hardcoded _OPTION_LETTERS bug at
        calibration.py run_one (score_options was always called with all 4
        letters regardless of how many real options the question had)."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = HighDScoreBackend()
        b.load()
        rows = _make_runner(b).run_many([row])

        # b.requested_option_sets[0] is the one-time neutral calibration call
        # (still all 4 letters, correctly); [1:] are the per-question calls,
        # which must request only this question's 3 real letters.
        assert all(set(opts) == {"A", "B", "C"} for opts in b.requested_option_sets[1:])

        assert rows[0]["calibration_adjusted_choice"] != "D"
        olj = json.loads(rows[0]["option_logprob_json"])
        assert "D" not in olj


class TestApplyCorrection:
    def test_subtracts_prior_from_raw_scores(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)

        raw = {"A": -1.0, "B": -2.0, "C": -0.5, "D": -3.0}
        prior = {"A": -0.5, "B": -0.5, "C": -0.5, "D": -0.5}

        result = runner._apply_correction(raw, prior)

        # After subtracting uniform prior, C still has the highest score
        assert result == "C"

    def test_prior_shifts_winner(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)

        # A has the highest raw score
        raw = {"A": -0.5, "B": -2.0, "C": -1.5, "D": -3.0}
        # But A also has a very high prior (model strongly favors A by default)
        prior = {"A": -0.1, "B": -2.0, "C": -1.0, "D": -3.0}

        # After correction: A=-0.4, B=0.0, C=-0.5, D=0.0 → B or D wins, not A
        result = runner._apply_correction(raw, prior)

        assert result != "A"
