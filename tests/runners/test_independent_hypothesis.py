# tests/runners/test_independent_hypothesis.py

from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.backends.types import ModelGenerationResult
from modelgen.runners.independent_hypothesis import IndependentHypothesisRunner
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT

from tests.runners.conftest import ErrorBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _score_text(score) -> str:
    return f"Brief analysis. <score>{score}</score>"


class ScoreByHypothesisBackend(DummyBackend):
    """Test backend that returns a canned response keyed by which option
    text appears in the prompt's hypothesis line, so each of the N
    independent generate() calls can be given a distinct score regardless
    of call order. Mirrors test_permutation.py's ContentAwareScoreBackend,
    the established pattern in this repo for content-keyed test backends.
    """

    def __init__(self, responses_by_option_text: dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self._responses_by_option_text = responses_by_option_text

    def generate(self, prompt, config=None):
        if not self._loaded:
            raise RuntimeError("Call load() before generate().")
        for option_text, response_text in self._responses_by_option_text.items():
            if f"The correct answer is {option_text}." in prompt:
                return ModelGenerationResult(
                    raw_text=response_text,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(response_text.split()),
                    finish_reason="eos",
                    metadata={"backend": "dummy"},
                )
        raise AssertionError(f"No matching hypothesis option found in prompt: {prompt}")


class SingleOptionFailureBackend(ScoreByHypothesisBackend):
    """Like ScoreByHypothesisBackend, but raises for one specific option text."""

    def __init__(self, responses_by_option_text: dict[str, str], failing_option_text: str, **kwargs):
        super().__init__(responses_by_option_text, **kwargs)
        self._failing_option_text = failing_option_text

    def generate(self, prompt, config=None):
        if not self._loaded:
            raise RuntimeError("Call load() before generate().")
        if f"The correct answer is {self._failing_option_text}." in prompt:
            raise RuntimeError("Simulated inference failure.")
        return super().generate(prompt, config)


def _make_runner(backend):
    return IndependentHypothesisRunner(
        backend=backend,
        method_name="independent_hypothesis",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestParseConfidenceScore:
    """Tests for IndependentHypothesisRunner._parse_confidence_score."""

    def test_parses_integer_score(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("text <score>72</score>")
        assert score == 72.0
        assert ok is True

    def test_parses_float_score(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("<score>12.5</score>")
        assert score == 12.5
        assert ok is True

    def test_missing_tag_falls_back_to_zero(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("no score tag here")
        assert score == 0.0
        assert ok is False

    def test_none_text_falls_back_to_zero(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score(None)
        assert score == 0.0
        assert ok is False

    def test_malformed_number_falls_back_to_zero(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("<score>abc</score>")
        assert score == 0.0
        assert ok is False

    def test_last_occurrence_wins(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score(
            "<score>10</score> reconsidering... <score>90</score>"
        )
        assert score == 90.0
        assert ok is True


class TestArgmaxWithRandomTiebreak:
    """Tests for IndependentHypothesisRunner._argmax_with_random_tiebreak."""

    def test_clear_winner(self):
        scores = {"A": 10.0, "B": 90.0, "C": 5.0, "D": 0.0}
        assert IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, "q1") == "B"

    def test_tie_is_deterministic_for_same_seed_and_question(self):
        scores = {"A": 50.0, "B": 50.0, "C": 0.0, "D": 0.0}
        first = IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, "q1")
        second = IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, "q1")
        assert first == second
        assert first in {"A", "B"}

    def test_tie_can_differ_across_questions(self):
        scores = {"A": 50.0, "B": 50.0, "C": 0.0, "D": 0.0}
        results = {
            IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, f"q{i}")
            for i in range(20)
        }
        assert results == {"A", "B"}


class TestIndependentHypothesisRunnerRunOne:
    """Tests for IndependentHypothesisRunner.run_one execution flow."""

    def test_makes_one_call_per_option(self, runner_question_row):
        responses = {
            "FTP": _score_text(50), "HTTP": _score_text(50),
            "HTTPS": _score_text(50), "SMTP": _score_text(50),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        # No assertion needed beyond "didn't raise" — the AssertionError inside
        # ScoreByHypothesisBackend.generate would fire if a prompt ever showed
        # more than one option's text, which is covered explicitly below.

    def test_each_prompt_contains_exactly_one_option(self, runner_question_row):
        responses = {
            "FTP": _score_text(50), "HTTP": _score_text(50),
            "HTTPS": _score_text(50), "SMTP": _score_text(50),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        # option_a_raw_text etc. confirm all four independent calls succeeded —
        # if any prompt had leaked a second option, ScoreByHypothesisBackend
        # would have raised AssertionError before getting here.
        for letter in ["a", "b", "c", "d"]:
            assert result[f"option_{letter}_raw_text"] is not None

    def test_highest_score_wins(self, runner_question_row):
        """Option C (HTTPS, the correct answer) gets the highest score."""
        responses = {
            "FTP": _score_text(10), "HTTP": _score_text(20),
            "HTTPS": _score_text(90), "SMTP": _score_text(5),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["final_prediction"] == "C"
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_prediction_scores_incorrect(self, runner_question_row):
        responses = {
            "FTP": _score_text(95), "HTTP": _score_text(20),
            "HTTPS": _score_text(10), "SMTP": _score_text(5),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["final_prediction"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_regex_parse_failure_falls_back_to_zero(self, runner_question_row):
        """An option whose response has no <score> tag falls back to score 0."""
        responses = {
            "FTP": "no tag here", "HTTP": _score_text(10),
            "HTTPS": _score_text(20), "SMTP": _score_text(5),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["option_a_score"] == 0.0
        assert result["option_a_score_parse_ok"] is False
        assert result["final_prediction"] == "C"  # highest of 0, 10, 20, 5

    def test_single_call_failure_falls_back_to_zero_but_still_scores(self, runner_question_row):
        responses = {
            "HTTP": _score_text(10), "HTTPS": _score_text(90), "SMTP": _score_text(5),
        }
        backend = SingleOptionFailureBackend(responses, failing_option_text="FTP")
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["option_a_model_status"] == "error"
        assert result["option_a_score"] == 0.0
        assert result["n_model_failures"] == 1
        assert result["final_prediction"] == "C"
        assert result["is_correct"] is True
        assert result["model_status"] == "success"

    def test_all_calls_fail_is_unscorable(self, runner_question_row, error_backend):
        result = _make_runner(error_backend).run_one(runner_question_row, sample_index=0)

        assert result["n_model_failures"] == 4
        assert result["final_prediction"] is None
        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert result["model_status"] == "error"

    def test_per_option_columns_present(self, runner_question_row):
        responses = {
            "FTP": _score_text(1), "HTTP": _score_text(2),
            "HTTPS": _score_text(3), "SMTP": _score_text(4),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        for letter in ["a", "b", "c", "d"]:
            assert f"option_{letter}_raw_text" in result
            assert f"option_{letter}_model_status" in result
            assert f"option_{letter}_score_parse_ok" in result
            assert f"option_{letter}_score" in result
        assert result["option_a_score"] == 1.0
        assert result["option_b_score"] == 2.0
        assert result["option_c_score"] == 3.0
        assert result["option_d_score"] == 4.0

    def test_method_name_is_independent_hypothesis(self, runner_question_row):
        responses = {
            "FTP": _score_text(50), "HTTP": _score_text(50),
            "HTTPS": _score_text(50), "SMTP": _score_text(50),
        }
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["method_name"] == "independent_hypothesis"

    def test_missing_fourth_option_makes_only_three_calls(self, runner_question_row):
        """ARC-Challenge questions with no choice_d must never get a phantom call."""
        row = dict(runner_question_row, choice_d=float("nan"))
        responses = {"FTP": _score_text(10), "HTTP": _score_text(20), "HTTPS": _score_text(90)}
        backend = ScoreByHypothesisBackend(responses)
        backend.load()
        result = _make_runner(backend).run_one(row, sample_index=0)

        assert "option_d_score" not in result
        assert result["final_prediction"] == "C"
