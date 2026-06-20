# tests/runners/test_additional_option.py

import json
from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.parsing.parser import parse_model_answer
from modelgen.parsing.types import PARSE_MISSING, PARSE_OK
from modelgen.runners.additional_option import (
    AdditionalOptionRunner,
    jaccard_similarity,
    match_options_via_scoring,
    match_text_to_options_jaccard,
)
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return AdditionalOptionRunner(
        backend=backend,
        method_name="additional_option",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="additional_option_test",
    )


class TestAdditionalOptionPrompt:
    def test_prompt_contains_option_e(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        prompt = runner._build_prompt(runner_question_row)

        assert "E. I don't know" in prompt

    def test_prompt_does_not_contain_option_f(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        prompt = runner._build_prompt(runner_question_row)

        assert "F. I don't know" not in prompt
        assert "F." not in prompt

    def test_options_dict_contains_e(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(runner_question_row)

        assert "E" in options
        assert options["E"] == "I don't know"

    def test_options_dict_has_exactly_five_keys(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(runner_question_row)

        assert set(options.keys()) == {"A", "B", "C", "D", "E"}


class TestAdditionalOptionEParsing:
    def test_parse_model_answer_e_with_e_in_options(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(runner_question_row)

        result = parse_model_answer("E", options)

        assert result.final_choice == "E"
        assert result.status == PARSE_OK

    def test_run_many_produces_result_rows(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        assert len(rows) == 1
        assert rows[0]["model_status"] == "success"

    def test_missing_fourth_option_has_no_phantom_choice(self, runner_question_row):
        """A 3-option question keeps E but must not show a phantom D."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(row)

        assert set(options.keys()) == {"A", "B", "C", "E"}
        prompt = runner._build_prompt(row)
        assert "nan" not in prompt.lower()
        assert "D." not in prompt
        assert "E. I don't know" in prompt


class TestJaccardSimilarity:
    def test_identical_strings_score_one(self):
        assert jaccard_similarity("HTTPS", "HTTPS") == 1.0

    def test_disjoint_strings_score_zero(self):
        assert jaccard_similarity("HTTPS", "FTP") == 0.0

    def test_partial_overlap(self):
        # {i, do, not, know} vs {i, dont, know} -> tokenizer splits "don't" into "don","t"
        score = jaccard_similarity("I do not know", "I don't know")
        assert 0.0 < score < 1.0

    def test_empty_strings_score_zero(self):
        assert jaccard_similarity("", "HTTPS") == 0.0
        assert jaccard_similarity("HTTPS", "") == 0.0


class TestMatchTextToOptionsJaccard:
    def test_exact_match_correct_option(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP", "E": "I don't know"}
        result, score = match_text_to_options_jaccard("HTTPS", options)
        assert result.final_choice == "C"
        assert result.status == PARSE_OK

    def test_bare_letter_resolved_directly_not_via_jaccard(self):
        """Regression test: a lone letter token has zero word overlap with
        any option's text under Jaccard, so every option used to tie at 0.0
        and the tie-break silently always picked 'A' regardless of which
        letter was actually stated."""
        options = {
            "A": "symmetric only",
            "B": "anti-symmetric only",
            "C": "both symmetric and anti-symmetric",
            "D": "an equivalence relation",
            "E": "I don't know",
        }
        result, score = match_text_to_options_jaccard(" D", options)
        assert result.final_choice == "D"
        assert result.status == PARSE_OK

    def test_bare_letter_with_surrounding_punctuation(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP", "E": "I don't know"}
        result, score = match_text_to_options_jaccard("D.", options)
        assert result.final_choice == "D"

    def test_bare_lowercase_letter_resolved_directly(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP", "E": "I don't know"}
        result, score = match_text_to_options_jaccard("d", options)
        assert result.final_choice == "D"
        assert score == 1.0

    def test_idk_phrasing_matches_e(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP", "E": "I don't know"}
        result, score = match_text_to_options_jaccard("I do not know the answer", options)
        assert result.final_choice == "E"
        assert result.status == PARSE_OK

    def test_empty_text_is_missing(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP", "E": "I don't know"}
        result, score = match_text_to_options_jaccard("", options)
        assert result.final_choice is None
        assert result.status == PARSE_MISSING
        assert score is None

    def test_ties_break_to_canonical_order(self):
        """Equal (zero) similarity to every option must deterministically pick A."""
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP", "E": "I don't know"}
        result, score = match_text_to_options_jaccard("completely unrelated zzz", options)
        assert result.final_choice == "A"
        assert score == 0.0

    def test_leading_letter_with_restated_option_resolved_directly(self):
        """Regression test: real Qwen-32B/ARC-Challenge data shows 999/1000
        responses in the shape "<letter>. <restated option text>\\n\\n<explanation>".
        Without a leading-letter precondition, the explanation's vocabulary
        Jaccard-matches poorly against the short option texts, and ties at
        0.0 silently pick 'A' regardless of the letter actually stated --
        which is exactly what skewed the overall distribution to ~55% 'A'."""
        options = {
            "A": "the dependent variable",
            "B": "the control",
            "C": "the hypothesis",
            "D": "the independent (manipulated) variable",
            "E": "I don't know",
        }
        raw_text = (
            " D. the independent (manipulated) variable\n\nThe speed of the car is "
            "being changed intentionally in each trial, making it the independent "
            "variable. The distance the car jumps would be the dependent variable, "
            "as it depends on the speed."
        )
        result, score = match_text_to_options_jaccard(raw_text, options)
        assert result.final_choice == "D"
        assert result.status == PARSE_OK
        assert score == 1.0

    def test_leading_letter_not_confused_with_indefinite_article(self):
        """A response that opens with the word "A" followed by a space (not
        punctuation) must not be misread as a declared leading letter -- it
        should fall through to genuine Jaccard matching."""
        options = {
            "A": "symmetric only",
            "B": "anti-symmetric only",
            "C": "both symmetric and anti-symmetric",
            "D": "an equivalence relation",
            "E": "I don't know",
        }
        raw_text = "A relation can be both symmetric and anti-symmetric at once."
        result, score = match_text_to_options_jaccard(raw_text, options)
        assert result.final_choice == "C"

    def test_leading_letter_with_three_option_question(self):
        """ARC-Challenge questions with only 3 real options (no D) must still
        resolve a leading letter correctly."""
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "E": "I don't know"}
        raw_text = "C. HTTPS\n\nThis is the standard secure web protocol."
        result, score = match_text_to_options_jaccard(raw_text, options)
        assert result.final_choice == "C"
        assert score == 1.0


class TestMatchOptionsViaScoring:
    """Unit tests for the Eq.(6)-faithful scoring-based matcher."""

    def test_real_option_with_highest_probability_wins(self):
        raw_scores = {"A": -3.0, "B": -3.0, "C": -0.1, "D": -3.0, "E": -3.0}
        result, prob_map = match_options_via_scoring(
            raw_scores, ["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]
        )
        assert result.final_choice == "C"
        assert result.status == PARSE_OK
        assert set(prob_map.keys()) == {"A", "B", "C", "D", "E"}

    def test_idk_never_selectable_even_with_highest_probability(self):
        """Regression test for Eq.(6): â = argmax_{a in A\\o_aux} P(y=a|x_A).
        IDK is structurally excluded from the argmax regardless of how much
        probability mass it has -- this is the core fix over the old Jaccard
        approach, which let IDK be the final selected (and scored) answer."""
        raw_scores = {"A": -0.01, "B": -5.0, "C": -5.0, "D": -5.0, "E": 0.0}
        result, prob_map = match_options_via_scoring(
            raw_scores, ["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]
        )
        assert result.final_choice != "E"
        assert result.final_choice == "A"
        # IDK's mass is still real in the recorded distribution -- it's only
        # excluded from the argmax, not from the computation.
        assert prob_map["E"] > prob_map["A"]

    def test_three_option_question_restricts_argmax_to_real_options(self):
        """A 3-option ARC-Challenge question (no D) must restrict the argmax
        to {A, B, C}, excluding both the missing D and IDK."""
        raw_scores = {"A": -3.0, "B": -0.1, "C": -3.0, "E": -3.0}
        result, prob_map = match_options_via_scoring(
            raw_scores, ["A", "B", "C"], ["A", "B", "C", "E"]
        )
        assert result.final_choice == "B"
        assert "D" not in prob_map

    def test_parse_reason_tags_eq6_mechanism(self):
        raw_scores = {"A": -0.1, "B": -3.0, "C": -3.0, "D": -3.0, "E": -3.0}
        result, _ = match_options_via_scoring(
            raw_scores, ["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]
        )
        assert result.reason == "aoi_eq6_argmax_excluding_idk"


class TestAdditionalOptionRunOne:
    def test_correct_answer_scored_correct(self, runner_question_row):
        b = DummyBackend(fixed_scores={"A": -3.0, "B": -3.0, "C": -0.1, "D": -3.0, "E": -3.0})
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_reason"] == "aoi_eq6_argmax_excluding_idk"

    def test_idk_dominant_probability_never_selected(self, runner_question_row):
        """Regression test for the redesign: even when IDK has the highest
        probability mass of all shown options, the final answer must be the
        best REAL option -- IDK is structurally never the final choice. This
        replaces the old behavior (test_idk_response_scored_incorrect_not_
        unscorable), which let IDK be picked and scored as a real wrong
        answer -- that was not faithful to Choi et al.'s Eq.(6)."""
        b = DummyBackend(fixed_scores={"A": -0.01, "B": -5.0, "C": -5.0, "D": -5.0, "E": 0.0})
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] != "E"
        assert result["parsed_choice"] in {"A", "B", "C", "D"}

    def test_option_logprob_and_aoi_probs_present(self, runner_question_row):
        b = DummyBackend(fixed_scores={"A": -3.0, "B": -3.0, "C": -0.1, "D": -3.0, "E": -3.0})
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        raw = json.loads(result["option_logprob_json"])
        probs = json.loads(result["aoi_probs_json"])
        assert set(raw.keys()) == {"A", "B", "C", "D", "E"}
        assert set(probs.keys()) == {"A", "B", "C", "D", "E"}
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_missing_fourth_option_restricts_to_real_options(self, runner_question_row):
        """A 3-option question (choice_d NaN) must score over {A,B,C,E} and
        select among {A,B,C} only, never D (not shown) or E (excluded)."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = DummyBackend(fixed_scores={"A": -3.0, "B": -3.0, "C": -0.1, "E": -3.0})
        b.load()
        result = _make_runner(b).run_one(row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_backend_failure_returns_no_parse_or_score(self, runner_question_row):
        from tests.runners.conftest import ErrorScoreBackend

        b = ErrorScoreBackend()
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["model_status"] == "error"
        assert result["option_logprob_json"] is None
