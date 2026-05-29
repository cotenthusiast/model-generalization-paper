# tests/runners/test_direct_mcq.py

from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.runners.direct_mcq import DirectMCQRunner
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from modelgen.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import ErrorBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, method_name="baseline"):
    return DirectMCQRunner(
        backend=backend,
        method_name=method_name,
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestDirectMCQRunnerRunOne:
    def test_correct_answer(self, runner_question_row):
        """Backend returns 'C' — should parse and score correct."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_status"] == PARSE_OK

    def test_incorrect_answer(self, runner_question_row):
        """Backend returns 'A' — should parse and score incorrect."""
        b = DummyBackend(fixed_text="A")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_failed_backend_call(self, runner_question_row, error_backend):
        """Backend raises — parsed and score fields should be None."""
        result = _make_runner(error_backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert result["model_status"] == "error"
        assert result["error_type"] == "LocalInferenceError"

    def test_unparseable_response(self, runner_question_row):
        """Backend returns gibberish — parse missing, score unscorable."""
        b = DummyBackend(fixed_text="I'm not sure about this question")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_result_row_metadata(self, runner_question_row):
        """Result row should carry all trace metadata correctly."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b, method_name="baseline").run_one(
            runner_question_row, sample_index=3
        )

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "baseline"
        assert result["split_name"] == "robustness"
        assert result["subject"] == "computer_security"
        assert result["sample_index"] == 3
        assert result["provider"] == "dummy"
        assert result["model_name"] == "dummy://"

    def test_prompt_contains_question_and_options(self, runner_question_row):
        """The prompt should include the question and all options."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert "securely browse websites" in result["prompt"]
        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    def test_lowercase_answer_parsed(self, runner_question_row):
        """Backend returns lowercase letter — should still parse correctly."""
        b = DummyBackend(fixed_text="c")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_model_status_success(self, runner_question_row):
        """Successful backend call sets model_status to 'success'."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["model_status"] == "success"

    def test_latency_seconds_present(self, runner_question_row):
        """latency_seconds should be a non-negative float."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert isinstance(result["latency_seconds"], float)
        assert result["latency_seconds"] >= 0.0


class TestDirectMCQRunnerBuildPrompt:
    def test_prompt_format(self, runner_question_row):
        """Prompt should be a non-empty string containing the question."""
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        prompt = runner._build_prompt(runner_question_row)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert runner_question_row["question_text"] in prompt

    def test_prompt_contains_all_options(self, runner_question_row):
        """Prompt should include all four option texts."""
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        prompt = runner._build_prompt(runner_question_row)

        assert "FTP" in prompt
        assert "HTTP" in prompt
        assert "HTTPS" in prompt
        assert "SMTP" in prompt
