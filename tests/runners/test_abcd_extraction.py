# tests/runners/test_abcd_extraction.py

"""Tests for abcd_extraction.py's paper-faithful four-tier cascade.

Covers Appendix F.2 of Nowak, Cadet, and Chin, "ABCD: All Biases Come
Disguised" (arXiv:2602.17445): explicit "answer is" marker, "Answer:"
marker, literal option-text search, last-sentence fallback -- each tested
individually, plus the last-occurrence behavior shared by tiers 1/2/3, and
the random fallback on total extraction failure.
"""

import random

import pytest

from modelgen.parsing.types import PARSE_OK
from modelgen.runners.abcd_extraction import extract_candidate_span, resolve_abcd_answer


class TestExtractCandidateSpanTier1AnswerIs:
    def test_single_occurrence(self):
        text = "Some reasoning here. The answer is HTTPS."
        assert extract_candidate_span(text, {}) == "answer is HTTPS."

    def test_last_occurrence_wins_over_earlier_one(self):
        """Mirrors Appendix F.2: the negative lookahead in
        "answer is (?!.*answer is ).+" means the regex only matches at the
        last "answer is " in the text -- the opposite of this repo's old
        earliest-statement design for the pre-redesign abcd condition."""
        text = "First I thought the answer is HTTP. After reconsidering, the answer is HTTPS."
        result = extract_candidate_span(text, {})
        assert result == "answer is HTTPS."
        assert "HTTP." not in result or "HTTPS" in result


class TestExtractCandidateSpanTier2AnswerColon:
    def test_used_when_tier1_absent(self):
        text = "Some discussion. Answer: HTTPS"
        assert extract_candidate_span(text, {}) == "Some discussion. Answer: HTTPS"

    def test_case_insensitive_leading_letter_only(self):
        text = "Reasoning. answer: HTTPS"
        result = extract_candidate_span(text, {})
        assert "HTTPS" in result

    def test_last_occurrence_wins(self):
        text = "Answer: HTTP. On reflection, Answer: HTTPS"
        result = extract_candidate_span(text, {})
        assert result.endswith("Answer: HTTPS")


class TestExtractCandidateSpanTier3LiteralOptionSearch:
    """Appendix F.2 Expression #3 has no literal regex in the paper -- only
    a prose description ("verbosely searching each answer... as is"). This
    repo's interpretation: a literal substring search across all candidate
    option texts, taking the rightmost (last) occurrence, used only when
    tiers #1 and #2 found no cue marker at all."""

    def test_single_literal_option_match(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        text = "I believe HTTPS is correct here."
        assert extract_candidate_span(text, options) == "HTTPS"

    def test_last_literal_occurrence_wins_among_multiple_options(self):
        """Real-shaped hedge-and-pivot case: the model states an answer,
        then second-guesses into a different option later. Tier #3's
        last-occurrence rule resolves to the LAST-mentioned option text,
        not the first -- the deliberate fidelity tradeoff versus this
        repo's previous earliest-statement heuristic for this condition."""
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        text = (
            "HTTPS is the correct protocol. (Note: while FTP is also a "
            "protocol, it is not used for secure browsing.) However, on "
            "reflection, some sources suggest the answer might instead be "
            "considered SMTP."
        )
        assert extract_candidate_span(text, options) == "SMTP"


class TestExtractCandidateSpanTier4LastSentence:
    def test_used_when_no_cue_and_no_literal_option_text(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        text = "I am not sure. The protocol used for secure web traffic over TLS."
        result = extract_candidate_span(text, options)
        assert result.strip() == "The protocol used for secure web traffic over TLS."

    def test_single_sentence_text_returns_whole_text(self):
        assert extract_candidate_span("HTTPS", {}) == "HTTPS"


class TestResolveAbcdAnswerRandomFallback:
    """Appendix F.2: "If the model does not produce an answer, we choose a
    random answer." Reported as occurring once across the paper's entire
    MMLU-Pro evaluation -- a near-nonexistent edge case, triggered here only
    when the model produces no text at all."""

    def test_empty_text_triggers_random_fallback_without_embedding_model(self):
        """Passing model=None proves the random path never touches the
        embedding model -- it must short-circuit before that call."""
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        result, score = resolve_abcd_answer(
            "", options, float("-inf"), model=None, rng=random.Random(42)
        )
        assert result.final_choice in options
        assert result.status == PARSE_OK
        assert score is None

    def test_none_text_triggers_random_fallback(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        result, score = resolve_abcd_answer(
            None, options, float("-inf"), model=None, rng=random.Random(7)
        )
        assert result.final_choice in options
        assert result.status == PARSE_OK

    def test_random_fallback_is_seed_reproducible(self):
        options = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        r1, _ = resolve_abcd_answer("", options, float("-inf"), model=None, rng=random.Random(99))
        r2, _ = resolve_abcd_answer("", options, float("-inf"), model=None, rng=random.Random(99))
        assert r1.final_choice == r2.final_choice


class TestResolveAbcdAnswerEndToEnd:
    def test_tier1_span_resolves_to_correct_option_via_embedding(self, canonical_options, sentence_model):
        text = "Some reasoning. The answer is HTTPS."
        result, score = resolve_abcd_answer(text, canonical_options, float("-inf"), sentence_model)
        assert result.final_choice == "C"
        assert score > 0.5

    def test_tier3_literal_match_resolves_to_correct_option(self, canonical_options, sentence_model):
        text = "I believe HTTPS is correct here."
        result, score = resolve_abcd_answer(text, canonical_options, float("-inf"), sentence_model)
        assert result.final_choice == "C"
