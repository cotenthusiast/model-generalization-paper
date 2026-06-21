# tests/evaluation/test_evaluate_run.py
#
# Tests for the post-hoc baseline fallback logic in scripts/evaluate_run.py.
# The script lives outside the modelgen package, so we load it via importlib.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "evaluate_run.py"
_spec = importlib.util.spec_from_file_location("evaluate_run_script", _SCRIPT_PATH)
_evaluate_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate_run)

apply_baseline_fallback = _evaluate_run.apply_baseline_fallback
compute_accuracy = _evaluate_run.compute_accuracy
reparse_run = _evaluate_run.reparse_run
rematch_abcd_rows = _evaluate_run.rematch_abcd_rows
rematch_text_extraction_rows = _evaluate_run.rematch_text_extraction_rows
rematch_additional_option_rows = _evaluate_run.rematch_additional_option_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    question_id: str,
    method_name: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    model_status: str = "success",
    is_correct=None,
    parsed_choice=None,
    parse_status: str | None = None,
    correct_option: str = "C",
    subject: str = "anatomy",
) -> dict:
    return {
        "question_id": question_id,
        "method_name": method_name,
        "model_name": model_name,
        "model_status": model_status,
        "is_correct": is_correct,
        "parsed_choice": parsed_choice,
        "parse_status": parse_status,
        "correct_option": correct_option,
        "subject": subject,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_mixed():
    """A DataFrame with baseline rows and unscorable two-stage rows."""
    return pd.DataFrame(
        [
            # Baseline — one correct, one incorrect
            _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C", parse_status="ok"),
            _row(question_id="q2", method_name="baseline", is_correct=False, parsed_choice="A", parse_status="ok"),
            # two_prompt — q1 unscorable (parse failure), q2 scored normally
            _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None, parse_status="missing"),
            _row(question_id="q2", method_name="two_prompt", is_correct=True, parsed_choice="C", parse_status="ok"),
            # two_prompt_cyclic — q1 unscorable
            _row(question_id="q1", method_name="two_prompt_cyclic", is_correct=None, parsed_choice=None, parse_status="missing"),
            # cyclic — q1 unscorable (should NOT be substituted)
            _row(question_id="q1", method_name="cyclic", is_correct=None, parsed_choice=None, parse_status="missing"),
        ]
    )


# ---------------------------------------------------------------------------
# apply_baseline_fallback — column injection
# ---------------------------------------------------------------------------


class TestApplyBaselineFallbackColumn:
    def test_fallback_applied_column_is_added(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        assert "fallback_applied" in result.columns

    def test_fallback_applied_has_no_null_values(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        assert result["fallback_applied"].notna().all()

    def test_original_df_not_mutated(self, df_mixed):
        original_nulls = df_mixed["is_correct"].isna().sum()
        apply_baseline_fallback(df_mixed)
        assert df_mixed["is_correct"].isna().sum() == original_nulls


# ---------------------------------------------------------------------------
# apply_baseline_fallback — substitution logic
# ---------------------------------------------------------------------------


class TestApplyBaselineFallbackSubstitution:
    def test_unscorable_two_prompt_row_gets_is_correct_from_baseline(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["is_correct"] is True

    def test_unscorable_two_prompt_row_gets_parsed_choice_from_baseline(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["parsed_choice"] == "C"

    def test_unscorable_two_prompt_cyclic_row_substituted(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt_cyclic")]
        assert bool(row.iloc[0]["is_correct"]) is True
        assert bool(row.iloc[0]["fallback_applied"]) is True

    def test_already_scored_row_not_changed(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q2") & (result["method_name"] == "two_prompt")]
        assert bool(row.iloc[0]["is_correct"]) is True
        assert bool(row.iloc[0]["fallback_applied"]) is False

    def test_baseline_rows_never_marked_as_fallback(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        bl = result[result["method_name"] == "baseline"]
        assert bl["fallback_applied"].eq(False).all()

    def test_cyclic_rows_not_substituted(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "cyclic")]
        assert row.iloc[0]["is_correct"] is None
        assert bool(row.iloc[0]["fallback_applied"]) is False


# ---------------------------------------------------------------------------
# apply_baseline_fallback — eligibility rules
# ---------------------------------------------------------------------------


class TestApplyBaselineFallbackEligibility:
    def test_api_failure_rows_are_not_substituted(self):
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="two_prompt", model_status="failure", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["is_correct"] is None
        assert bool(row.iloc[0]["fallback_applied"]) is False

    def test_no_baseline_means_no_substitution(self):
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        assert result.iloc[0]["is_correct"] is None

    def test_question_id_must_match_for_substitution(self):
        """Baseline for q2 must not bleed into an unscorable q1 row."""
        df = pd.DataFrame(
            [
                _row(question_id="q2", method_name="baseline", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["is_correct"] is None

    def test_different_models_do_not_cross_contaminate(self):
        """Baseline from gpt-4.1-mini must not substitute into a gemini row."""
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", model_name="gpt-4.1-mini", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="two_prompt", model_name="gemini-2.5-flash", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["model_name"] == "gemini-2.5-flash")]
        assert row.iloc[0]["is_correct"] is None

    def test_fallback_transfers_incorrect_baseline_result(self):
        """Baseline answer can be wrong; the fallback still uses whatever the baseline had."""
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", is_correct=False, parsed_choice="A", parse_status="ok"),
                _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert bool(row.iloc[0]["is_correct"]) is False
        assert row.iloc[0]["parsed_choice"] == "A"
        assert bool(row.iloc[0]["fallback_applied"]) is True

    def test_multiple_models_each_fallback_independently(self):
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", model_name="gpt-4.1-mini", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="baseline", model_name="gemini-2.5-flash", is_correct=False, parsed_choice="A"),
                _row(question_id="q1", method_name="two_prompt", model_name="gpt-4.1-mini", is_correct=None, parsed_choice=None),
                _row(question_id="q1", method_name="two_prompt", model_name="gemini-2.5-flash", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        gpt_row = result[(result["model_name"] == "gpt-4.1-mini") & (result["method_name"] == "two_prompt")]
        gemini_row = result[(result["model_name"] == "gemini-2.5-flash") & (result["method_name"] == "two_prompt")]
        assert bool(gpt_row.iloc[0]["is_correct"]) is True
        assert bool(gemini_row.iloc[0]["is_correct"]) is False


# ---------------------------------------------------------------------------
# compute_accuracy — fallback_count column
# ---------------------------------------------------------------------------


class TestComputeAccuracyFallbackCount:
    def test_fallback_count_zero_when_no_fallback_applied(self, df_mixed):
        """Without applying fallback, fallback_count should be 0 for all rows."""
        # df_mixed has no fallback_applied column — count should default to 0
        accuracy = compute_accuracy(df_mixed)
        assert "fallback_count" in accuracy.columns
        assert (accuracy["fallback_count"] == 0).all()

    def test_fallback_count_reflects_substitutions(self, df_mixed):
        """After fallback, the count should match actual substitutions."""
        df_after = apply_baseline_fallback(df_mixed)
        accuracy = compute_accuracy(df_after)
        # q1 was substituted in two_prompt — fallback_count for (two_prompt, gpt-4.1-mini) == 1
        row = accuracy[(accuracy["method"] == "two_prompt") & (accuracy["model"] == "Qwen/Qwen2.5-7B-Instruct")]
        assert row.iloc[0]["fallback_count"] == 1

    def test_fallback_count_zero_for_baseline_method(self, df_mixed):
        df_after = apply_baseline_fallback(df_mixed)
        accuracy = compute_accuracy(df_after)
        bl = accuracy[accuracy["method"] == "baseline"]
        assert (bl["fallback_count"] == 0).all()


# ---------------------------------------------------------------------------
# reparse_run — score_options() methods (pride, calibration) have no raw_text
# ---------------------------------------------------------------------------


def _reparsable_row(
    *,
    question_id: str,
    method_name: str,
    raw_text,
    parse_reason: str,
    parsed_choice,
    is_correct,
    correct_option: str = "C",
) -> dict:
    return {
        "question_id": question_id,
        "method_name": method_name,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_status": "success",
        "raw_text": raw_text,
        "choice_a": "wrong A",
        "choice_b": "wrong B",
        "choice_c": "right answer",
        "choice_d": "wrong D",
        "correct_option": correct_option,
        "parsed_choice": parsed_choice,
        "parse_status": "ok",
        "parse_reason": parse_reason,
        "is_correct": is_correct,
    }


class TestReparseRunSkipsScoreOptionsMethods:
    """pride and calibration pick answers via score_options(), so raw_text is
    always null for them — reparse_run must not try to re-parse those rows."""

    @pytest.fixture
    def df_with_score_options_methods(self):
        return pd.DataFrame(
            [
                _reparsable_row(
                    question_id="q1",
                    method_name="baseline",
                    raw_text="The answer is C.",
                    parse_reason="direct",
                    parsed_choice=None,
                    is_correct=None,
                ),
                _reparsable_row(
                    question_id="q1",
                    method_name="pride",
                    raw_text=float("nan"),
                    parse_reason="pride_eq8",
                    parsed_choice="C",
                    is_correct=True,
                ),
                _reparsable_row(
                    question_id="q1",
                    method_name="calibration",
                    raw_text=float("nan"),
                    parse_reason="answer_calibration",
                    parsed_choice="A",
                    is_correct=False,
                ),
            ]
        )

    def test_does_not_raise_on_null_raw_text(self, df_with_score_options_methods):
        reparse_run(df_with_score_options_methods)

    def test_calibration_row_left_unchanged(self, df_with_score_options_methods):
        result = reparse_run(df_with_score_options_methods)
        row = result[result["method_name"] == "calibration"].iloc[0]
        assert row["parsed_choice"] == "A"
        assert row["is_correct"] == False  # noqa: E712

    def test_pride_row_left_unchanged(self, df_with_score_options_methods):
        result = reparse_run(df_with_score_options_methods)
        row = result[result["method_name"] == "pride"].iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_baseline_row_is_reparsed(self, df_with_score_options_methods):
        result = reparse_run(df_with_score_options_methods)
        row = result[result["method_name"] == "baseline"].iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712


class TestReparseRunSkipsEmbeddingMatchMethods:
    """abcd and text_extraction pick their answer via sentence-embedding
    cosine similarity against the free-text response, not the plain
    letter/text-match parser. Their raw_text is real prose (dash-labeled or
    letter-suppressed), so re-parsing it with parse_model_answer can find a
    spurious bare letter (e.g. the indefinite article "A") and silently
    overwrite a correct embedding-based match — reparse_run must skip them."""

    @pytest.fixture
    def df_with_embedding_match_methods(self):
        return pd.DataFrame(
            [
                _reparsable_row(
                    question_id="q1",
                    method_name="abcd",
                    raw_text=(
                        "A permutation can be a product of disjoint cycles, "
                        "which is the right answer."
                    ),
                    parse_reason="Embedding cosine match to option C (score=0.412)",
                    parsed_choice="C",
                    is_correct=True,
                ),
                _reparsable_row(
                    question_id="q1",
                    method_name="text_extraction",
                    raw_text=(
                        "A permutation can be a product of disjoint cycles, "
                        "which is the right answer."
                    ),
                    parse_reason="Embedding cosine match to option C (score=0.388)",
                    parsed_choice="C",
                    is_correct=True,
                ),
                _reparsable_row(
                    question_id="q1",
                    method_name="additional_option",
                    raw_text="A permutation can be a product of disjoint cycles.",
                    parse_reason="Jaccard text match to option C (score=0.250)",
                    parsed_choice="C",
                    is_correct=True,
                ),
            ]
        )

    def test_abcd_row_left_unchanged(self, df_with_embedding_match_methods):
        result = reparse_run(df_with_embedding_match_methods)
        row = result[result["method_name"] == "abcd"].iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_text_extraction_row_left_unchanged(self, df_with_embedding_match_methods):
        result = reparse_run(df_with_embedding_match_methods)
        row = result[result["method_name"] == "text_extraction"].iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_additional_option_row_left_unchanged(self, df_with_embedding_match_methods):
        result = reparse_run(df_with_embedding_match_methods)
        row = result[result["method_name"] == "additional_option"].iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# rematch_abcd_rows
# ---------------------------------------------------------------------------


def _abcd_row(*, free_text_response: str, correct_option: str = "C") -> dict:
    return {
        "method_name": "abcd",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": correct_option,
        "free_text_response": free_text_response,
        "parsed_choice": None,
        "is_correct": None,
    }


class TestRematchAbcdRows:
    """rematch_abcd_rows re-derives parsed_choice/is_correct for abcd rows
    from the saved free_text_response, using a small embedding model
    injected explicitly to avoid downloading the 600M-param production
    model in tests."""

    _SMALL_MODEL = "all-MiniLM-L6-v2"

    def test_no_op_when_no_abcd_rows_present(self, tmp_path):
        df = pd.DataFrame(
            [{"method_name": "baseline", "parsed_choice": "A", "is_correct": True}]
        )
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        pd.testing.assert_frame_equal(result, df)

    def test_matches_exact_option_text(self, tmp_path):
        df = pd.DataFrame([_abcd_row(free_text_response="HTTPS")])
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        row = result.iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_isolates_final_answer_from_longer_response(self, tmp_path):
        free_text = (
            "Some might think this relates to a different protocol entirely, "
            "but on reflection the answer is HTTPS."
        )
        df = pd.DataFrame([_abcd_row(free_text_response=free_text)])
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        row = result.iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_cache_hit_avoids_reloading_model(self, tmp_path, monkeypatch):
        """Second call with the same rows must not re-load SentenceTransformer."""
        df = pd.DataFrame([_abcd_row(free_text_response="HTTPS")])
        rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("SentenceTransformer should not be constructed on a cache hit")

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer", fail_if_called
        )
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        assert result.iloc[0]["parsed_choice"] == "C"

    def test_multiple_rows_batched_without_cross_contamination(self, tmp_path):
        """All rows' texts are encoded in one batched call — each row must
        still resolve to its own correct option, not another row's."""
        df = pd.DataFrame(
            [
                _abcd_row(free_text_response="HTTPS", correct_option="C"),
                _abcd_row(free_text_response="FTP", correct_option="A"),
                _abcd_row(free_text_response="SMTP", correct_option="D"),
            ]
        )
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        assert list(result["parsed_choice"]) == ["C", "A", "D"]
        assert list(result["is_correct"]) == [True, True, True]

    def test_other_methods_untouched(self, tmp_path):
        df = pd.DataFrame(
            [
                _abcd_row(free_text_response="HTTPS"),
                {
                    "method_name": "baseline",
                    "parsed_choice": "Z",
                    "is_correct": False,
                    "free_text_response": None,
                    "choice_a": "FTP",
                    "choice_b": "HTTP",
                    "choice_c": "HTTPS",
                    "choice_d": "SMTP",
                    "correct_option": "C",
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                },
            ]
        )
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        baseline_row = result[result["method_name"] == "baseline"].iloc[0]
        assert baseline_row["parsed_choice"] == "Z"
        assert baseline_row["is_correct"] == False  # noqa: E712

    def test_leading_letter_still_resolves_via_tier3_literal_match(self, tmp_path):
        """Deliberate behavior change: the paper's cascade has no
        declared-leading-letter shortcut (the Appendix F.3 prompt instructs
        the model never to write one), so this case now resolves through
        tier #3's literal option-text search (the restated "HTTPS" text),
        not through a model-free letter shortcut -- it happens to still
        land on the correct option here because the option text itself is
        present, not because the leading "C." was specially recognized."""
        df = pd.DataFrame([_abcd_row(free_text_response="C. HTTPS is correct.")])
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        assert result.iloc[0]["parsed_choice"] == "C"

    def test_bare_cue_stated_letter_is_no_longer_specially_rescued(self, tmp_path):
        """Deliberate behavior change: arXiv:2602.17445's cascade has no
        equivalent of text_extraction's cue-stated-letter shortcut. "the
        best answer is C." is captured whole by tier #1 ("answer is C."),
        then embedded as literal text -- with no option text to match
        against, the result is whatever the embedding model scores
        highest, not necessarily the bare letter "C" itself. This is an
        accepted, paper-faithful limitation: the Appendix F.3 prompt is
        designed to stop models from writing bare letters in the first
        place, so the cascade has no rescue path for when they do anyway."""
        df = pd.DataFrame(
            [
                _abcd_row(
                    free_text_response=(
                        "HTTPS is used for secure browsing. Therefore, the best answer is C."
                    )
                )
            ]
        )
        result = rematch_abcd_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        assert result.iloc[0]["parsed_choice"] in ("A", "B", "C", "D")
        assert result.iloc[0]["normalized_text"] == "answer is C."


# ---------------------------------------------------------------------------
# rematch_text_extraction_rows
# ---------------------------------------------------------------------------


def _text_extraction_row(*, free_text_response: str, correct_option: str = "C") -> dict:
    return {
        "method_name": "text_extraction",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": correct_option,
        "free_text_response": free_text_response,
        "parsed_choice": None,
        "is_correct": None,
    }


class TestRematchTextExtractionRows:
    """rematch_text_extraction_rows re-derives parsed_choice/is_correct for
    text_extraction rows from the saved free_text_response. Shares
    _rematch_with_embedding_fallback with rematch_abcd_rows, so this focuses
    on text_extraction-specific behavior (its default similarity_threshold
    is 0.1, not abcd's -inf) rather than re-testing shared machinery."""

    _SMALL_MODEL = "all-MiniLM-L6-v2"

    def test_no_op_when_no_text_extraction_rows_present(self, tmp_path):
        df = pd.DataFrame(
            [{"method_name": "baseline", "parsed_choice": "A", "is_correct": True}]
        )
        result = rematch_text_extraction_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        pd.testing.assert_frame_equal(result, df)

    def test_matches_exact_option_text(self, tmp_path):
        df = pd.DataFrame([_text_extraction_row(free_text_response="HTTPS")])
        result = rematch_text_extraction_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        row = result.iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_leading_letter_resolved_without_loading_model(self, tmp_path, monkeypatch):
        df = pd.DataFrame([_text_extraction_row(free_text_response="C. HTTPS is correct.")])

        def fail_if_called(*args, **kwargs):
            raise AssertionError("SentenceTransformer should not be constructed for a leading-letter row")

        monkeypatch.setattr("sentence_transformers.SentenceTransformer", fail_if_called)
        result = rematch_text_extraction_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        assert result.iloc[0]["parsed_choice"] == "C"

    def test_below_threshold_is_unscorable(self, tmp_path):
        """text_extraction's default threshold (0.1) must be applied to the
        embedding-fallback path, unlike abcd's -inf (always argmax)."""
        df = pd.DataFrame([_text_extraction_row(free_text_response="completely unrelated text xyz")])
        result = rematch_text_extraction_rows(
            df, similarity_threshold=0.9, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path
        )
        row = result.iloc[0]
        assert row["parsed_choice"] is None
        assert row["parse_status"] == "parse_missing"

    def test_cache_does_not_bake_in_threshold(self, tmp_path):
        """Same (span, options) pair rematched with two different thresholds
        must not reuse a stale threshold-derived status from the cache --
        the cache stores raw scores, and the threshold is applied on read."""
        df = pd.DataFrame([_text_extraction_row(free_text_response="completely unrelated text xyz")])
        lenient = rematch_text_extraction_rows(
            df, similarity_threshold=0.0, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path
        )
        strict = rematch_text_extraction_rows(
            df, similarity_threshold=0.99, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path
        )
        assert lenient.iloc[0]["parse_status"] == "parse_ok"
        assert strict.iloc[0]["parse_status"] == "parse_missing"

    def test_other_methods_untouched(self, tmp_path):
        df = pd.DataFrame(
            [
                _text_extraction_row(free_text_response="HTTPS"),
                {
                    "method_name": "baseline",
                    "parsed_choice": "Z",
                    "is_correct": False,
                    "free_text_response": None,
                    "choice_a": "FTP",
                    "choice_b": "HTTP",
                    "choice_c": "HTTPS",
                    "choice_d": "SMTP",
                    "correct_option": "C",
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                },
            ]
        )
        result = rematch_text_extraction_rows(df, embedding_model=self._SMALL_MODEL, cache_dir=tmp_path)
        baseline_row = result[result["method_name"] == "baseline"].iloc[0]
        assert baseline_row["parsed_choice"] == "Z"
        assert baseline_row["is_correct"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# rematch_additional_option_rows
# ---------------------------------------------------------------------------


def _additional_option_row(*, raw_text: str, correct_option: str = "C") -> dict:
    return {
        "method_name": "additional_option",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": correct_option,
        "raw_text": raw_text,
        "parsed_choice": None,
        "is_correct": None,
    }


class TestRematchAdditionalOptionRows:
    """rematch_additional_option_rows re-derives parsed_choice/is_correct for
    additional_option rows from the saved raw_text via Jaccard matching —
    pure string operations, no model to load."""

    def test_no_op_when_no_additional_option_rows_present(self):
        df = pd.DataFrame(
            [{"method_name": "baseline", "parsed_choice": "A", "is_correct": True}]
        )
        result = rematch_additional_option_rows(df)
        pd.testing.assert_frame_equal(result, df)

    def test_matches_exact_option_text(self):
        df = pd.DataFrame([_additional_option_row(raw_text="HTTPS")])
        result = rematch_additional_option_rows(df)
        row = result.iloc[0]
        assert row["parsed_choice"] == "C"
        assert row["is_correct"] == True  # noqa: E712

    def test_redesigned_eq6_rows_left_untouched(self):
        """Regression test: rows collected under the score_options()-based
        Eq.(6) redesign have raw_text=None and are already final at
        collection time. Without the raw_text-presence guard, this function
        would read raw_text=None as an empty response and overwrite an
        already-correct parsed_choice with PARSE_MISSING."""
        row = _additional_option_row(raw_text=None)
        row["raw_text"] = None
        row["parsed_choice"] = "D"
        row["is_correct"] = False
        row["parse_reason"] = "aoi_eq6_argmax_excluding_idk"
        df = pd.DataFrame([row])
        result = rematch_additional_option_rows(df)
        pd.testing.assert_frame_equal(result, df)

    def test_idk_response_matches_e_and_scores_incorrect(self):
        df = pd.DataFrame([_additional_option_row(raw_text="I really don't know")])
        result = rematch_additional_option_rows(df)
        row = result.iloc[0]
        assert row["parsed_choice"] == "E"
        assert row["is_correct"] == False  # noqa: E712

    def test_other_methods_untouched(self):
        df = pd.DataFrame(
            [
                _additional_option_row(raw_text="HTTPS"),
                {
                    "method_name": "baseline",
                    "parsed_choice": "Z",
                    "is_correct": False,
                    "raw_text": None,
                    "choice_a": "FTP",
                    "choice_b": "HTTP",
                    "choice_c": "HTTPS",
                    "choice_d": "SMTP",
                    "correct_option": "C",
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                },
            ]
        )
        result = rematch_additional_option_rows(df)
        baseline_row = result[result["method_name"] == "baseline"].iloc[0]
        assert baseline_row["parsed_choice"] == "Z"
        assert baseline_row["is_correct"] == False  # noqa: E712
