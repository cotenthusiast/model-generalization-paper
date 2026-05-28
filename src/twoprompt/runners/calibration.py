# src/twoprompt/runners/calibration.py

from __future__ import annotations

import json
import logging
import time
from typing import Any

from twoprompt.runners.local_base import LocalExperimentRunner
from twoprompt.parsing.types import PARSE_OK, ParseResult
from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt

logger = logging.getLogger(__name__)

_OPTION_LETTERS = ["A", "B", "C", "D"]

class AnswerCalibrationRunner(LocalExperimentRunner):
    """Answer-level calibration runner.

    Estimates a per-option bias prior by scoring each option in a
    content-free (neutral) context, then subtracts that prior from
    the real-context option scores before picking the answer.

    Two phases:
      1. Calibration — score options against a neutral prompt (no real
         question) to build a bias estimate.
      2. Inference — score options against the real question, subtract
         the calibration prior, pick the highest corrected score.

    One backend call per option per question (plus calibration calls).
    """

    def setup(self, question_rows) -> None:
        """Run calibration phase once before inference begins."""
        self._prior = self._run_calibration()
        self._calibration_ready = True

    def _run_calibration(self) -> dict[str, float]:
        """Score each option letter in a neutral context to estimate label bias.

        Returns a dict mapping option letter to its log-prob prior.
        """
        prompt = self._build_neutral_prompt()
        try:
            result = self.backend.score_options(prompt, _OPTION_LETTERS)
            return result.scores
        except Exception as exc:
            logger.warning("AnswerCalibration: calibration call failed — %s. Using zero prior.", exc)
            return {letter: 0.0 for letter in _OPTION_LETTERS}


    def _build_neutral_prompt(self) -> str:
        """Build a content-free prompt used during calibration.

        The prompt contains no real question — just enough structure to
        elicit a first-token option-letter response.
        """
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question="N/A",
            options=["N/A"] * 4
        )

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Score all options for one question, apply calibration correction, return result."""
        prompt = self._build_prompt(question_row)

        start = time.perf_counter()
        try:
            score_result_obj = self.backend.score_options(prompt, _OPTION_LETTERS)
            raw_scores = score_result_obj.scores
        except Exception as exc:
            latency = time.perf_counter() - start
            logger.warning(
                "AnswerCalibration: score_options failed for question %s — %s",
                question_row["question_id"],
                exc,
            )
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
            row["calibration_adjusted_choice"] = None
            row["prior_logprob_json"] = None
            row["option_logprob_json"] = None
            return row

        latency = time.perf_counter() - start

        chosen = self._apply_correction(raw_scores, self._prior)

        parse_result = ParseResult(
            final_choice=chosen,
            status=PARSE_OK,
            raw_text=None,
            normalized_text=chosen,
            reason="answer_calibration",
        )
        score_result = self._score(parse_result, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            generation_result=None,
            latency_seconds=latency,
            parsed_result=parse_result,
            score_result=score_result,
            error=None,
        )

        row["model_status"] = "success"
        row["error_type"] = None
        row["error_message"] = None
        row["error_stage"] = None
        row["error_retryable"] = None

        row["calibration_adjusted_choice"] = chosen
        row["prior_logprob_json"] = json.dumps(self._prior)
        row["option_logprob_json"] = json.dumps(raw_scores)

        return row

    def _build_prompt(self, question_row: Any) -> str:
        """Build the standard MCQ prompt for a question."""
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=list(self._build_options(question_row).values())
        )

    def _apply_correction(
        self,
        raw_scores: dict[str, float],
        prior: dict[str, float],
    ) -> str:
        """Subtract prior bias from raw scores and return the winning option letter."""
        adjusted_scores = {}
        for option, raw in raw_scores.items():
            adjusted_scores[option] = raw - prior[option]
        return max(adjusted_scores, key=adjusted_scores.get)
