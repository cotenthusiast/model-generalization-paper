# tests/runners/test_text_extraction.py

from pathlib import Path

import pytest

from modelgen.backends.dummy import DummyBackend
from modelgen.parsing.types import PARSE_MISSING, PARSE_OK
from modelgen.runners.text_extraction import (
    TextExtractionRunner,
    extract_final_answer_span,
    match_free_text_to_options,
    resolve_stage2_answer,
)
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


class TestExtractFinalAnswerSpan:
    """Unit tests for the abcd answer-isolation helper."""

    def test_returns_empty_string_for_none(self):
        assert extract_final_answer_span(None) == ""

    def test_single_sentence_returned_unchanged(self):
        assert extract_final_answer_span("HTTPS") == "HTTPS"

    def test_falls_back_to_first_sentence_when_no_cue_present(self):
        """Regression test: real abcd responses state the answer in sentence
        1, then add caveats/restatements in later sentences. The fallback
        (no explicit cue anywhere) must prefer the earliest sentence, not
        the last -- the literal opposite of the previous (broken) behavior."""
        text = "Some context here. More discussion follows. The relevant fact is X."
        assert extract_final_answer_span(text) == "Some context here."

    def test_isolates_explicit_conclusion_cue_over_earlier_discussion(self):
        text = (
            "Some might think this relates to symmetric groups, but actually "
            "permutations form a broader category. A permutation can be a "
            "product of disjoint cycles. Therefore the answer is that a "
            "cycle is a type of permutation."
        )
        result = extract_final_answer_span(text)
        assert "Therefore" in result
        assert "symmetric groups" not in result

    def test_first_cue_sentence_wins_when_multiple_present(self):
        """Regression test: a model second-guessing itself ("On reflection,
        the answer is C") must not override its first stated answer -- real
        Qwen-32B/ARC-Challenge data showed exactly this shape, with 80% of a
        125-question regression set flipping to correct once the *first*
        stated answer was used instead of the last."""
        text = "The answer is A. On reflection, the answer is C."
        result = extract_final_answer_span(text)
        assert "The answer is A" in result
        assert "the answer is C" not in result

    def test_strips_trailing_conversational_filler_before_cue_search(self):
        text = "The answer is B. Let me know if you have any other questions!"
        result = extract_final_answer_span(text)
        assert "The answer is B" in result
        assert "Let me know" not in result

    def test_strips_trailing_filler_before_first_sentence_fallback(self):
        text = "FTP is the correct protocol here. I hope this helps!"
        result = extract_final_answer_span(text)
        assert result == "FTP is the correct protocol here."

    def test_all_filler_does_not_crash_and_returns_something(self):
        """Edge case: if every sentence looks like filler, stripping must not
        leave an empty candidate list -- fall back to the original text
        rather than crashing or returning an empty span."""
        text = "Let me know if you have any other questions!"
        result = extract_final_answer_span(text)
        assert result != ""

    def test_real_world_hedge_and_pivot_pattern(self):
        """Regression test mirroring real Qwen-32B/ARC-Challenge abcd output:
        a clearly stated first answer, followed by a hedge that pivots to a
        different (sometimes invented) restatement. The first statement must
        win."""
        text = (
            "An infectious, cell-cycle disease. "
            "(Note: while DFTD is indeed infectious, it is not typically "
            "described as a cell-cycle disease in scientific literature.) "
            "However, given the options, the best answer would be: an "
            "infectious, chronic disease."
        )
        result = extract_final_answer_span(text)
        assert result == "An infectious, cell-cycle disease."


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


class TestResolveStage2Answer:
    """Unit tests for the shared abcd/text_extraction stage-2 resolver."""

    def test_leading_letter_resolved_without_embedding_model(self, canonical_options):
        """Passing model=None proves this path never touches the embedding
        model at all -- it must be resolved before that call."""
        result, score = resolve_stage2_answer("C. HTTPS is correct.", canonical_options, 0.1, None)
        assert result.final_choice == "C"
        assert score == 1.0

    def test_cue_stated_letter_resolved_without_embedding_model(self, canonical_options):
        """Regression test: real Llama-3.1-8B-Instruct text_extraction output
        showed ~25% of rows isolating a span like "The best answer is C."
        instead of restated option text. Embedding that against option text
        scores too low on every option and used to come back unscorable --
        the cue-stated letter must be resolved directly instead."""
        text = (
            "HTTPS is used for secure browsing. "
            "Therefore, the best answer is C."
        )
        result, score = resolve_stage2_answer(text, canonical_options, 0.1, None)
        assert result.final_choice == "C"
        assert score == 1.0

    def test_indefinite_article_after_therefore_not_treated_as_letter(
        self, canonical_options, sentence_model
    ):
        """"Therefore, a combination..." must not be misread as a stated
        letter just because "a" upper-cases to a valid choice -- the
        lowercase, unpunctuated token must fall through to embedding match."""
        text = "Therefore, a combination of factors explains this, namely HTTPS."
        result, score = resolve_stage2_answer(text, canonical_options, 0.1, sentence_model)
        assert result.final_choice == "C"

    def test_indefinite_article_after_answer_is_not_treated_as_letter(
        self, canonical_options, sentence_model
    ):
        text = "The answer is a well-known secure web protocol, namely HTTPS."
        result, score = resolve_stage2_answer(text, canonical_options, 0.1, sentence_model)
        assert result.final_choice == "C"

    def test_falls_back_to_embedding_match_when_no_letter_present(
        self, canonical_options, sentence_model
    ):
        result, score = resolve_stage2_answer("HTTPS", canonical_options, 0.1, sentence_model)
        assert result.final_choice == "C"
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

    def test_missing_fourth_option_has_no_phantom_choice(self, runner_question_row):
        """A 3-option question (choice_d is None/NaN) must not crash or show a phantom option."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(row, sample_index=0)

        assert "nan" not in result["prompt"].lower()
        assert "D." not in result["prompt"]
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_leading_letter_resolved_directly_without_embedding(self, runner_question_row):
        """Regression test: despite stage 1's instruction not to state a
        letter, real data shows Qwen-7B does so anyway in ~90% of responses.
        A declared leading letter must be resolved directly rather than
        embedding the full response."""
        b = DummyBackend(fixed_text="C. HTTPS is the secure protocol used for browsing.")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["best_similarity_score"] == 1.0
        assert result["is_correct"] is True

    def test_long_hedging_response_uses_earliest_statement(self, runner_question_row):
        """Regression test mirroring real Llama-3.1-8B-Instruct output: long,
        repetitive, or hedging responses where the real answer is stated
        early and would otherwise be diluted by embedding the full text."""
        b = DummyBackend(
            fixed_text=(
                "HTTPS is the correct protocol. "
                "(Note: while FTP is also a protocol, it is not used for "
                "secure browsing.) However, on reflection, some sources "
                "suggest the answer might instead be considered SMTP."
            )
        )
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
