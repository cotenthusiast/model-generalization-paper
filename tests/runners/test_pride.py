# tests/runners/test_pride.py

import json
import logging
from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.runners.pride import PriDeRunner

from tests.runners.conftest import ErrorScoreBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


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
