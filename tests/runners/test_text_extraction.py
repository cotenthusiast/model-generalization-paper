# tests/runners/test_text_extraction.py

from pathlib import Path

import pytest

from modelgen.backends.dummy import DummyBackend
from modelgen.parsing.types import PARSE_MISSING, PARSE_OK
from modelgen.runners.text_extraction import TextExtractionRunner, match_free_text_to_options
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import ErrorBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, similarity_threshold=0.1):
    return TextExtractionRunner(
        backend=backend,
        method_name="text_extraction",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        similarity_threshold=similarity_threshold,
    )


class TestMatchFreeTextToOptions:
    """Unit tests for the deterministic matching helper."""

    def test_exact_match_correct(self, canonical_options, sentence_model):
        result, score = match_free_text_to_options("HTTPS", canonical_options, 0.1, sentence_model)
        assert result.final_choice == "C"
        assert result.status == PARSE_OK
        assert score == pytest.approx(1.0)

    def test_exact_match_incorrect(self, canonical_options, sentence_model):
        result, score = match_free_text_to_options("FTP", canonical_options, 0.1, sentence_model)
        assert result.final_choice == "A"
        assert result.status == PARSE_OK

    def test_below_threshold_is_missing(self, canonical_options, sentence_model):
        result, score = match_free_text_to_options(
            "completely unrelated response xyz", canonical_options, 0.9, sentence_model
        )
        assert result.final_choice is None
        assert result.status == PARSE_MISSING

    def test_empty_free_text_is_missing(self, canonical_options, sentence_model):
        result, score = match_free_text_to_options("", canonical_options, 0.1, sentence_model)
        assert result.final_choice is None
        assert result.status == PARSE_MISSING
        assert score is None

    def test_zero_threshold_always_selects(self, canonical_options, sentence_model):
        result, score = match_free_text_to_options(
            "totally irrelevant text zzz", canonical_options, 0.0, sentence_model
        )
        assert result.final_choice is not None
        assert result.status == PARSE_OK


class TestTextExtractionRunnerRunOne:
    def test_correct_answer(self, runner_question_row):
        """Free-text matches option C exactly — should score correct."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_answer(self, runner_question_row):
        """Free-text matches option A — should score incorrect."""
        b = DummyBackend(fixed_text="FTP")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_backend_failure_returns_no_parse_or_score(self, runner_question_row, error_backend):
        result = _make_runner(error_backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["model_status"] == "error"

    def test_low_similarity_is_unscorable(self, runner_question_row):
        """Response below threshold produces PARSE_MISSING → SCORE_UNSCORABLE."""
        b = DummyBackend(fixed_text="completely unrelated text xyz")
        b.load()
        result = _make_runner(b, similarity_threshold=0.9).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_free_text_response_preserved(self, runner_question_row):
        """Stage 1 output should be stored in the result row."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["free_text_response"] == "HTTPS"

    def test_best_similarity_score_present(self, runner_question_row):
        """best_similarity_score should be a float when generation succeeds."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert isinstance(result["best_similarity_score"], float)
        assert result["best_similarity_score"] > 0.0

    def test_result_row_metadata(self, runner_question_row):
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "text_extraction"
        assert result["split_name"] == "robustness"

    def test_prompt_contains_question_and_all_options(self, runner_question_row):
        """Stage 1 prompt must include the question and all four option texts."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert runner_question_row["question_text"] in result["prompt"]
        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    def test_no_second_llm_call(self, runner_question_row):
        """Stage 2 is deterministic — backend.generate should be called exactly once."""
        call_count = 0
        original_generate = DummyBackend.generate

        class CountingBackend(DummyBackend):
            def generate(self, prompt, config=None):
                nonlocal call_count
                call_count += 1
                return original_generate(self, prompt, config)

        b = CountingBackend(fixed_text="HTTPS")
        b.load()
        _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert call_count == 1
