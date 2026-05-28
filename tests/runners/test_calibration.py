# tests/runners/test_calibration.py

import json
import logging
from pathlib import Path

import pytest

from twoprompt.backends.dummy import DummyBackend
from twoprompt.runners.calibration import AnswerCalibrationRunner

from tests.runners.conftest import ErrorScoreBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


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

        with caplog.at_level(logging.WARNING, logger="twoprompt.runners.calibration"):
            rows = runner.run_many([runner_question_row])

        assert rows[0]["calibration_adjusted_choice"] is None
        assert rows[0]["model_status"] == "error"

    def test_multiple_questions_all_get_results(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row, runner_question_row])

        assert len(rows) == 2
        assert all(r["model_status"] == "success" for r in rows)


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
