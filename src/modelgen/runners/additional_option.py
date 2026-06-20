# src/modelgen/runners/additional_option.py

import json
import re
import time
from typing import Any

from modelgen.parsing.parser import detect_leading_letter, normalize_output_text
from modelgen.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from modelgen.pipeline.prompt_builder import build_direct_mcq_prompt
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.runners.direct_mcq import DirectMCQRunner
from modelgen.runners.pride_debias import logprob_map_to_label_distribution

_AUX_LETTER = "E"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-set Jaccard similarity: |intersection| / |union|, lowercased word tokens."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def match_text_to_options_jaccard(
    raw_text: str | None,
    options: dict[str, str],
) -> tuple[ParseResult, float | None]:
    """LEGACY — superseded by match_options_via_scoring / Eq.(6) below.

    Kept only so already-collected free-text-generation additional_option
    run data (raw_text-based rows in runs/) can still be re-evaluated; the
    active AdditionalOptionRunner no longer calls this. This function let
    "I don't know" be the final selected (and scored) answer, which Choi et
    al. 2025's actual Eq.(6) never permits — see match_options_via_scoring's
    docstring for the corrected behavior and citation.

    Select the option whose text has the highest Jaccard overlap with raw_text.
    Mirrors Choi et al.'s black-box auxiliary-option-injection (AOI)
    text-matching procedure for black-box models (§4.4): "generate text
    responses with the same input prompt; compute the Jaccard similarity
    between each choice option and the output text; select the choice with
    the highest similarity score." No abstention threshold — always returns
    the argmax, including possibly "I don't know" itself when it's the best
    match. Ties are broken by canonical option order (A before B before
    ... before E).

    A bare single-letter response (e.g. "C") is resolved directly rather
    than Jaccard-matched: a lone letter token has zero word overlap with any
    option's text, so every option ties at score 0.0 and the tie-break would
    silently always pick "A" regardless of which letter was actually stated.

    A response that opens with a declared letter followed by restated option
    text (e.g. "D. the independent variable\n\nThis is because...") is also
    resolved directly from that leading letter rather than Jaccard-matched:
    Jaccard-matching the trailing explanation against the short option texts
    routinely picks the wrong option (the explanation's vocabulary overlaps
    unevenly with the options), and on questions where it doesn't, ties at
    0.0 again hit the same always-"A" tie-break as the bare-letter case.
    Real-data check on a 1000-question Qwen-32B/ARC-Challenge run found 999/
    1000 responses in exactly this leading-letter-plus-explanation shape,
    with the Jaccard fallback skewing the overall answer distribution to
    ~55% "A" as a result.

    Only responses with neither a bare letter nor a declared leading letter
    (e.g. an explained "I don't know" with no stated letter at all) fall
    through to genuine Jaccard matching.
    """
    normalized = normalize_output_text(raw_text)
    if not normalized:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=raw_text,
            normalized_text=normalized,
            reason="Empty response",
        ), None

    letters = list(options.keys())

    stripped = normalized.strip("()[]{}<>\".,:;!?'")
    if stripped.upper() in options:
        return ParseResult(
            final_choice=stripped.upper(),
            status=PARSE_OK,
            raw_text=raw_text,
            normalized_text=normalized,
            reason="Bare letter response",
        ), 1.0

    leading = detect_leading_letter(normalized, options.keys())
    if leading is not None:
        return ParseResult(
            final_choice=leading,
            status=PARSE_OK,
            raw_text=raw_text,
            normalized_text=normalized,
            reason=f"Leading letter '{leading}' matched a valid option",
        ), 1.0

    scores = {letter: jaccard_similarity(normalized, options[letter]) for letter in letters}
    best_letter = max(letters, key=lambda l: scores[l])
    best_score = scores[best_letter]

    return ParseResult(
        final_choice=best_letter,
        status=PARSE_OK,
        raw_text=raw_text,
        normalized_text=normalized,
        reason=f"Jaccard text match to option {best_letter} (score={best_score:.3f})",
    ), best_score


def match_options_via_scoring(
    raw_scores: dict[str, float],
    real_letters: list[str],
    letters_shown: list[str],
) -> tuple[ParseResult, dict[str, float]]:
    """Eq.(6) of Choi et al. 2025 ("Mitigating Selection Bias with Node
    Pruning and Auxiliary Options", ACL 2025): the auxiliary "I don't know"
    option is added to the shown option set (Eq. 5: A := A ∪ {o_aux}), but
    the final answer is always the argmax restricted to the real options —
    IDK is structurally never selectable, no matter its probability mass:

        â = argmax_{a ∈ A\\o_aux} P(ŷ=a | x_A)

    P(ŷ=a|x_A) is the softmax over ALL shown options' first-token logprobs
    from a single forward pass (computed here via
    pride_debias.logprob_map_to_label_distribution over `letters_shown`,
    which includes IDK) — IDK's presence still perturbs the distribution
    over the real options through the shared softmax denominator, which is
    the entire point of the manipulation. Only the final argmax excludes it.

    `real_letters` is whichever real options exist for this specific
    question (e.g. ["A","B","C"] for a 3-option ARC-Challenge question with
    no D) — the paper doesn't address variable option counts explicitly, so
    this restricts the argmax to exactly the real options present for that
    question, same as every other method in this codebase already does via
    local_base._build_options' empty-option filtering. The auxiliary letter
    is always "E" regardless of how many real options exist (never slides
    into a gap left by a missing option), matching the existing prompt-
    building convention.

    Returns (ParseResult, prob_map) where prob_map is the full distribution
    over letters_shown (including IDK) for auditability.
    """
    probs = logprob_map_to_label_distribution(raw_scores, letters=letters_shown)
    prob_map = {letter: float(p) for letter, p in zip(letters_shown, probs)}

    real_probs = {letter: prob_map[letter] for letter in real_letters}
    best_letter = max(real_probs, key=real_probs.get)

    return ParseResult(
        final_choice=best_letter,
        status=PARSE_OK,
        raw_text=None,
        normalized_text=best_letter,
        reason="aoi_eq6_argmax_excluding_idk",
    ), prob_map


class AdditionalOptionRunner(DirectMCQRunner):
    """Runner for the additional-option condition (Choi et al. 2025, AOI).

    Presents the model with a standard multiple-choice question plus an
    additional "I don't know" option, then scores every shown option
    (including IDK) in a single score_options() forward pass and selects
    the final answer via Eq.(6): the argmax restricted to the real options,
    with IDK structurally excluded from ever being selected — see
    match_options_via_scoring. No free-text generation, no text matching.
    One score_options() call per question.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        options_with_idk = self._build_options(question_row)
        real_letters = [letter for letter in options_with_idk if letter != _AUX_LETTER]
        letters_shown = real_letters + [_AUX_LETTER]
        prompt = self._build_prompt(question_row)

        start = time.perf_counter()
        try:
            score_result_obj = self.backend.score_options(prompt, letters_shown)
            raw_scores = score_result_obj.scores
        except Exception as exc:
            latency = time.perf_counter() - start
            row = self._build_result_row(
                question_row=question_row,
                prompt=prompt,
                sample_index=sample_index,
                generation_result=None,
                latency_seconds=latency,
                parsed_result=None,
                score_result=None,
                error=str(exc),
            )
            row["option_logprob_json"] = None
            row["aoi_probs_json"] = None
            return row

        latency = time.perf_counter() - start
        parsed_result, prob_map = match_options_via_scoring(raw_scores, real_letters, letters_shown)
        score_result = self._score(parsed_result, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            generation_result=None,  # score_options-based: no generate() call
            latency_seconds=latency,
            parsed_result=parsed_result,
            score_result=score_result,
            error=None,
        )

        # _build_result_row sets model_status="error" when generation_result is
        # None; override to reflect the actual score_options outcome instead
        # (same pattern as PriDeRunner/AnswerCalibrationRunner).
        row["model_status"] = "success"
        row["error_type"] = None
        row["error_message"] = None
        row["error_stage"] = None
        row["error_retryable"] = None

        row["option_logprob_json"] = json.dumps(raw_scores)
        row["aoi_probs_json"] = json.dumps(prob_map)
        return row

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )

    def _build_options(self, question_row: Any) -> dict[str, str]:
        options = super()._build_options(question_row)
        options["E"] = "I don't know"
        return options