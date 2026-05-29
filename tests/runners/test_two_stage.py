# tests/runners/test_two_stage.py

from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.runners.two_stage import TwoStageRunner
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from modelgen.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import ErrorBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return TwoStageRunner(
        backend=backend,
        method_name="two_prompt",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestTwoStageRunnerRunOne:
    def test_correct_two_stage(self, runner_question_row):
        """Stage 1 returns 'HTTPS', stage 2 matches to 'C' — should score correct."""
        # DummyBackend always returns the same text, so both calls return "C".
        # Stage 1: "C" is the free-text answer.
        # Stage 2: "C" is parsed directly → correct.
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_two_stage(self, runner_question_row):
        """Stage 2 returns 'A' — should score incorrect."""
        b = DummyBackend(fixed_text="A")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_stage_one_failure_returns_early(self, runner_question_row, error_backend):
        """If stage 1 fails, should return immediately with no parse or score."""
        result = _make_runner(error_backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert result["model_status"] == "error"

    def test_free_text_response_preserved(self, runner_question_row):
        """The intermediate free-text response should be saved in the result row."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["free_text_response"] == "HTTPS"
        assert result["free_text_prompt"] is not None
        assert result["free_text_latency"] is not None

    def test_matching_prompt_contains_free_text(self, runner_question_row):
        """The option-matching prompt should include the free-text answer."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert "HTTPS" in result["prompt"]

    def test_result_row_metadata(self, runner_question_row):
        """Result row should carry trace metadata."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "two_prompt"
        assert result["split_name"] == "robustness"

    def test_unparseable_matching_response(self, runner_question_row):
        """Stage 2 returns gibberish — should be unscorable."""
        b = DummyBackend(fixed_text="I think it might be one of those")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE
        assert result["free_text_response"] == "I think it might be one of those"

    def test_fallback_not_used_by_default(self, runner_question_row):
        """Without fallback_on_parse_failure=True, fallback_used is False."""
        b = DummyBackend(fixed_text="I think it might be one of those")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["fallback_used"] is False

    def test_latency_seconds_present(self, runner_question_row):
        """latency_seconds should be a non-negative float from the matching stage."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert isinstance(result["latency_seconds"], float)
        assert result["latency_seconds"] >= 0.0
