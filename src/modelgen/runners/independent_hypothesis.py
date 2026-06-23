# src/modelgen/runners/independent_hypothesis.py

import random
import re
import time
from typing import Any

from modelgen.parsing.types import PARSE_OK, ParseResult
from modelgen.pipeline.prompt_builder import build_independent_hypothesis_prompt
from modelgen.runners.local_base import LocalExperimentRunner

# Matches "<score>X</score>" where X is an int or float, case-insensitive.
# Last occurrence wins, consistent with parser.py's last-occurrence philosophy
# for reasoning models that restate candidate values before the final one.
_SCORE_PATTERN = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)


class IndependentHypothesisRunner(LocalExperimentRunner):
    """Runner for the independent-hypothesis condition.

    Every real option for a question (4, or 3 for the rare ARC-Challenge
    question missing a D choice — see LocalExperimentRunner._build_options)
    is evaluated independently: the backend receives the question and exactly
    one candidate option framed as a hypothesis, and returns a 0-100
    confidence score for that hypothesis alone. One backend.generate() call
    per option; no option ever appears alongside another in the same prompt.
    The final prediction is the option with the highest confidence score;
    ties are broken randomly using a seed derived from the run seed and
    question ID, so the outcome is reproducible.

    All raw responses and per-option scores are preserved in the result row
    so aggregation (e.g. a different tie-break or parse rule) can be redone
    post-hoc without rerunning inference.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        options = self._build_options(question_row)
        letters = list(options.keys())

        prompts = [
            build_independent_hypothesis_prompt(
                template=self._prompts["independent_hypothesis"],
                question=question_row["question_text"],
                option_text=options[letter],
            )
            for letter in letters
        ]

        scores: dict[str, float] = {}
        parse_ok: dict[str, bool] = {}
        raw_texts: dict[str, str | None] = {}
        model_statuses: dict[str, str] = {}
        n_failures = 0
        last_error: str | None = None

        start = time.perf_counter()
        for letter, prompt in zip(letters, prompts):
            generation_result, _, error = self._call_backend(prompt)
            if generation_result is not None:
                score, ok = self._parse_confidence_score(generation_result.raw_text)
                raw_texts[letter] = generation_result.raw_text
                model_statuses[letter] = "success"
            else:
                score, ok = 0.0, False
                raw_texts[letter] = None
                model_statuses[letter] = "error"
                n_failures += 1
                last_error = error
            scores[letter] = score
            parse_ok[letter] = ok
        latency = time.perf_counter() - start

        parsed_result = None
        score_result = None
        final_prediction = None

        if n_failures < len(letters):
            final_prediction = self._argmax_with_random_tiebreak(
                scores, self.generation_config.seed, question_row["question_id"]
            )
            parsed_result = ParseResult(
                final_choice=final_prediction,
                status=PARSE_OK,
                raw_text=None,
                normalized_text="",
                reason="argmax_of_independent_hypothesis_scores",
            )
            score_result = self._score(parsed_result, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            generation_result=None,
            latency_seconds=latency,
            parsed_result=parsed_result,
            score_result=score_result,
            error=last_error,
        )

        # _build_result_row marks model_status "error" whenever generation_result
        # is None, which is always true here since N separate calls were made
        # rather than one representative call. Patch the trace fields back to
        # "success" when at least one option call succeeded — same fix as
        # PermutationRunner, which has the identical multi-call shape.
        if n_failures < len(letters):
            row["model_status"] = "success"
            row["error_type"] = None
            row["error_message"] = None
            row["error_stage"] = None
            row["error_retryable"] = None

        for letter in letters:
            suffix = letter.lower()
            row[f"option_{suffix}_raw_text"] = raw_texts[letter]
            row[f"option_{suffix}_model_status"] = model_statuses[letter]
            row[f"option_{suffix}_score_parse_ok"] = parse_ok[letter]
            row[f"option_{suffix}_score"] = scores[letter]
        row["final_prediction"] = final_prediction
        row["n_model_failures"] = n_failures

        return row

    @staticmethod
    def _parse_confidence_score(raw_text: str | None) -> tuple[float, bool]:
        """Extract a confidence score from raw model output via regex.

        Args:
            raw_text: Raw model output text, expected to contain a
                ``<score>X</score>`` tag.

        Returns:
            Tuple of (score, parse_ok). On parse failure, score is 0.0
            and parse_ok is False.
        """
        if not raw_text:
            return 0.0, False
        matches = _SCORE_PATTERN.findall(raw_text)
        if not matches:
            return 0.0, False
        try:
            return float(matches[-1]), True
        except ValueError:
            return 0.0, False

    @staticmethod
    def _argmax_with_random_tiebreak(
        scores: dict[str, float],
        seed: int | None,
        question_id: str,
    ) -> str:
        """Pick the highest-scoring option letter, breaking ties randomly.

        The tie-break RNG is seeded from (seed, question_id) rather than a
        shared mutable generator, so the outcome is reproducible regardless
        of run-to-run iteration order.

        Args:
            scores: Mapping from option letter to confidence score.
            seed: Run seed for reproducible tie-breaking.
            question_id: Question identifier, mixed into the tie-break seed.

        Returns:
            The selected option letter.
        """
        best_score = max(scores.values())
        tied = sorted(letter for letter, s in scores.items() if s == best_score)
        if len(tied) == 1:
            return tied[0]
        rng = random.Random(f"{seed}:{question_id}")
        return rng.choice(tied)
